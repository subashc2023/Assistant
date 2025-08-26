# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic",
#     "google",
#     "litellm",
#     "mcp",
#     "openai",
#     "prompt_toolkit",
# ]
# ///
import os
import sys
import json
import asyncio
import logging
import shutil
import pathlib
import argparse
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from contextlib import AsyncExitStack

import litellm
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import ANSI

from config import AppConfig, Metrics, check_provider_auth

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MCP_CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"
HISTORY_FILE = SCRIPT_DIR / ".chat_history"

logger = logging.getLogger(__name__)

class AnsiTheme:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._codes = (
            {'reset': '\x1b[0m','bold': '\x1b[1m','red': '\x1b[31m','green': '\x1b[32m',
             'yellow': '\x1b[33m','blue': '\x1b[34m','cyan': '\x1b[36m','gray': '\x1b[90m','magenta':'\x1b[35m'}
            if enabled else {k: '' for k in ['reset','bold','red','green','yellow','blue','cyan','gray','magenta']}
        )

    def style(self, text: str, *styles: str) -> str:
        if not self.enabled or not text:
            return text
        codes = ''.join(self._codes.get(s, '') for s in styles)
        return f"{codes}{text}{self._codes['reset']}"

    def label(self, text: str, color: str = 'cyan') -> str:
        return self.style(text, 'bold', color)

    def sep(self, title: Optional[str] = None, char: str = "─") -> str:
        width = get_terminal_width()
        if title:
            title_text = f" {title} "
            side = (width - len(title_text)) // 2
            line = char * side + title_text + char * (width - side - len(title_text))
            return self.style(line, 'gray')
        return self.style(char * width, 'gray')

def get_terminal_width() -> int:
    try:
        width = shutil.get_terminal_size(fallback=(80, 20)).columns
        return max(20, min(width, 120))
    except Exception:
        return 80

def format_exception(exc: BaseException, limit: int = 1024) -> str:
    name = type(exc).__name__
    text = str(exc)[:limit]
    return f"{name}: {text}" if text else name

def generate_tool_call_id(index: int, prefix: str = "call") -> str:
    return f"{prefix}_{index}"

def configure_logging(config: AppConfig) -> None:
    level = getattr(logging, config.log_level, logging.INFO)
    if config.log_json:
        fmt = '{"level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(levelname)s %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)
    for lib in ['httpx', 'anyio', 'mcp', 'urllib3', 'openai', 'LiteLLM', 'litellm']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    litellm.set_verbose = False

def create_theme(config: AppConfig) -> AnsiTheme:
    enabled = config.use_color and sys.stdout.isatty()
    return AnsiTheme(enabled=enabled)

class MCPServer:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    async def initialize(self) -> None:
        cmd = self._resolve_command(self.config.get("command", ""))
        args = self.config.get("args", [])
        env = self.config.get("env")
        self._validate_config(cmd, args, env)
        params = StdioServerParameters(command=cmd, args=args, env=env)

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            logger.info("Server '%s' initialized", self.name)
        except (OSError, ConnectionError) as e:
            logger.error("Error initializing server '%s': %r", self.name, e)
            await self.cleanup()
            raise

    def _resolve_command(self, cmd: str) -> str:
        if not cmd or not cmd.strip():
            raise ValueError("Invalid command in config")
        cmd = cmd.strip()
        if os.path.isabs(cmd) or os.path.sep in cmd:
            if not os.path.exists(cmd):
                raise ValueError(f"Command not found: {cmd}")
            return cmd
        resolved = shutil.which(cmd)
        if not resolved:
            raise ValueError(f"Command '{cmd}' not found in PATH")
        return resolved

    def _validate_config(self, cmd: str, args: List, env: Optional[Dict]) -> None:
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError(f"[{self.name}] 'args' must be a list of strings")
        if env is not None:
            if not isinstance(env, dict):
                raise ValueError(f"[{self.name}] 'env' must be a dictionary")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise ValueError(f"[{self.name}] env keys and values must be strings")

    async def list_tools(self) -> Any:
        if not self.session:
            raise RuntimeError("Server not initialized")
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not self.session:
            raise RuntimeError("Server not initialized")
        return await self.session.call_tool(tool_name, arguments=arguments)

    async def cleanup(self) -> None:
        try:
            await self.exit_stack.aclose()
        except (RuntimeError, OSError, ConnectionError) as e:
            msg = str(e)
            benign = any(s in msg for s in (
                "cancel scope", "Event loop is closed", "already closed",
                "attached to a different loop", "cannot schedule new futures",
                "I/O operation on closed file", "Proactor"
            ))
            if not benign:
                logger.warning("Cleanup warning for '%s': %r", self.name, e)
        finally:
            self.session = None

class MCPRouter:
    def __init__(self):
        self.servers: List[MCPServer] = []
        self.tool_specs: List[Dict[str, Any]] = []
        self.tool_to_server: Dict[str, MCPServer] = {}
        self.tool_name_map: Dict[str, str] = {}

    async def load_and_start(self, config_path: pathlib.Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        raw_servers = config.get("mcpServers", {})
        if not isinstance(raw_servers, dict):
            raise ValueError("mcp_config.json must contain 'mcpServers' object (dict) if present")

        discovered = self._discover_local_servers(config_path)
        for name, cfg in discovered.items():
            if name not in raw_servers:
                raw_servers[name] = cfg

        if not raw_servers:
            raise ValueError("No MCP servers found (neither in mcp_config.json nor in ./servers)")

        self.servers = [MCPServer(name, cfg) for name, cfg in raw_servers.items()]

        self.tool_specs.clear()
        self.tool_to_server.clear()
        self.tool_name_map.clear()

        results = await asyncio.gather(*[self._init_and_list(s) for s in self.servers])
        for server, tools_resp in results:
            self._register_server_tools(server, tools_resp)

        if not self.tool_specs:
            logger.warning("No tools found from any MCP server")
        else:
            logger.info("Loaded %d tools from %d servers", len(self.tool_specs), len(self.servers))

    async def _init_and_list(self, server: MCPServer) -> Tuple[MCPServer, Any]:
        await server.initialize()
        tools = await server.list_tools()
        return server, tools

    def _register_server_tools(self, server: MCPServer, tools_resp: Any) -> None:
        for tool in tools_resp.tools:
            schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
            schema = self._validate_schema(schema, server.name, tool.name)
            base_name = f"{server.name}_{tool.name}"
            namespaced = self._get_unique_name(base_name)

            self.tool_to_server[namespaced] = server
            self.tool_name_map[namespaced] = tool.name

            self.tool_specs.append({
                "type": "function",
                "function": {
                    "name": namespaced,
                    "description": tool.description or "",
                    "parameters": schema
                }
            })

    def _get_unique_name(self, base_name: str) -> str:
        name = base_name
        suffix = 2
        while name in self.tool_to_server:
            name = f"{base_name}_{suffix}"
            suffix += 1
        return name

    def _validate_schema(self, schema: Any, server_name: str, tool_name: str) -> Dict:
        if schema is None:
            return {"type": "object", "properties": {}}
        if not isinstance(schema, dict):
            raise ValueError(f"[{server_name}] Tool '{tool_name}' schema must be object")
        if schema.get("type") != "object":
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}
        elif not isinstance(schema["properties"], dict):
            raise ValueError(f"[{server_name}] Tool '{tool_name}' properties must be object")
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        return schema

    async def call_tool(self, namespaced: str, arguments: Dict[str, Any]) -> Any:
        server = self.tool_to_server.get(namespaced)
        if not server:
            raise RuntimeError(f"No server owns tool '{namespaced}'")
        original_name = self.tool_name_map.get(namespaced, namespaced)
        return await server.call_tool(original_name, arguments)

    async def cleanup(self) -> None:
        for server in self.servers:
            try:
                await server.cleanup()
            except Exception as e:
                logger.warning("Cleanup error for '%s': %r", server.name, e)

    def get_server_tools_map(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for namespaced, server in self.tool_to_server.items():
            prefix = f"{server.name}_"
            display = namespaced[len(prefix):] if namespaced.startswith(prefix) else namespaced
            result.setdefault(server.name, []).append(display)
        return result

    def _discover_local_servers(self, config_path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
        servers: Dict[str, Dict[str, Any]] = {}
        candidates = [config_path.parent / "servers", SCRIPT_DIR / "servers"]
        seen = set()
        for d in candidates:
            try:
                rp = d.resolve()
                if rp in seen or not rp.exists() or not rp.is_dir():
                    continue
                seen.add(rp)
                for py in d.glob("*.py"):
                    if py.name in ("__init__.py", ) or py.name.startswith(("._", "~")):
                        continue
                    name = py.stem
                    cmd = sys.executable or "python"
                    servers.setdefault(name, {"command": cmd, "args": [str(py.resolve())]})
            except (OSError, PermissionError) as e:
                logger.debug("Local server discovery error in %s: %r", str(d), e)
        if servers:
            logger.info("Discovered %d local MCP servers in ./servers", len(servers))
        return servers

@dataclass
class ToolResult:
    name: str
    tool_call_id: str
    content: str
    success: bool
    duration: float
    lines: int = 0
    truncated: bool = False

    def to_message(self) -> Dict[str, Any]:
        meta = (f"[TOOL META] name={self.name} ok={str(self.success).lower()} "
                f"duration={self.duration:.2f}s lines={self.lines} "
                f"truncated={str(self.truncated).lower()}\n")
        return {"role": "tool", "tool_call_id": self.tool_call_id, "name": self.name, "content": meta + self.content}

class ToolExecutor:
    def __init__(self, router: MCPRouter, config: AppConfig):
        self.router = router
        self.config = config

    async def execute_batch(self, tool_calls: List[Dict]) -> List[ToolResult]:
        results: List[ToolResult] = []
        limit = max(1, self.config.max_parallel_tools or 1)
        sem = asyncio.Semaphore(limit)

        tasks = []
        for i, tc in enumerate(tool_calls):
            name = tc.get("function", {}).get("name", "")
            args_str = tc.get("function", {}).get("arguments", "{}")
            tool_call_id = tc.get("id", generate_tool_call_id(i, "tc"))
            tasks.append(self._execute_single(name, args_str, tool_call_id, sem))

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in task_results:
                if isinstance(r, Exception):
                    logger.error("Tool execution error: %r", r)
                else:
                    results.append(r)
        return results

    async def _execute_single(self, name: str, args_str: Any, tool_call_id: str, sem: asyncio.Semaphore) -> ToolResult:
        try:
            if isinstance(args_str, str):
                args = json.loads(args_str) if args_str.strip() else {}
            elif isinstance(args_str, dict):
                args = args_str
            else:
                args = {}
        except json.JSONDecodeError as e:
            return ToolResult(name, tool_call_id, f"ERROR: Invalid JSON arguments: {str(e)[:100]}", False, 0.0)

        async with sem:
            start = time.monotonic()
            try:
                result = await asyncio.wait_for(self.router.call_tool(name, args), timeout=self.config.tool_timeout_seconds)
                duration = time.monotonic() - start
                content, truncated, lines = self._format_result(result)
                return ToolResult(name, tool_call_id, content, True, duration, lines, truncated)
            except asyncio.TimeoutError:
                return ToolResult(name, tool_call_id, f"ERROR: Timeout after {self.config.tool_timeout_seconds}s", False, self.config.tool_timeout_seconds)
            except Exception as e:
                return ToolResult(name, tool_call_id, f"ERROR: {format_exception(e)}", False, time.monotonic() - start)

    def _format_result(self, result: Any) -> Tuple[str, bool, int]:
        try:
            if hasattr(result, 'content') and isinstance(result.content, list):
                parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        parts.append(item.text)
                    else:
                        parts.append(json.dumps(item, indent=2, ensure_ascii=False))
                text = '\n'.join(parts)
            else:
                text = json.dumps(result, indent=2, ensure_ascii=False)
            lines = len(text.splitlines())
            if len(text) > self.config.tool_result_max_chars:
                text = text[:self.config.tool_result_max_chars]
                return text or "[empty response]", True, lines
            return text or "[empty response]", False, lines
        except (TypeError, ValueError):
            text = str(result)
            lines = len(text.splitlines())
            if len(text) > self.config.tool_result_max_chars:
                text = text[:self.config.tool_result_max_chars]
                return text, True, lines
            return text, False, lines

class StreamParser:
    def __init__(self):
        self.content: List[str] = []
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.tool_args: Dict[int, List[str]] = {}

    def process_chunk(self, chunk: Any) -> Optional[str]:
        try:
            delta = chunk.choices[0].delta

            if content := getattr(delta, 'content', None):
                self.content.append(content)
                return content

            if tcs := getattr(delta, 'tool_calls', None):
                for tc in tcs:
                    idx = getattr(tc, 'index', len(self.tool_calls))
                    if idx not in self.tool_calls:
                        self.tool_calls[idx] = {
                            'id': getattr(tc, 'id', None),
                            'type': getattr(tc, 'type', 'function'),
                            'function': {'name': None, 'arguments': ''}
                        }
                        self.tool_args[idx] = []

                    if tc_id := getattr(tc, 'id', None):
                        self.tool_calls[idx]['id'] = tc_id
                    if func := getattr(tc, 'function', None):
                        if name := getattr(func, 'name', None):
                            self.tool_calls[idx]['function']['name'] = name
                        if args := getattr(func, 'arguments', None):
                            self.tool_args[idx].append(args)
            return None
        except (AttributeError, IndexError, KeyError):
            return None

    def get_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            'role': 'assistant',
            'content': ''.join(self.content).strip() or None
        }
        if self.tool_calls:
            tcs: List[Dict[str, Any]] = []
            for idx in sorted(self.tool_calls.keys()):
                tc = self.tool_calls[idx].copy()
                tc['function']['arguments'] = ''.join(self.tool_args.get(idx, []))
                tc['id'] = tc.get('id') or generate_tool_call_id(idx)
                tcs.append(tc)
            message['tool_calls'] = tcs
        return message

class LLMOrchestrator:
    def __init__(self, config: AppConfig, router: MCPRouter, metrics: Metrics, theme: AnsiTheme):
        self.config = config
        self.router = router
        self.metrics = metrics
        self.theme = theme
        self.executor = ToolExecutor(router, config)
        self.conversation: List[Dict[str, Any]] = []
        self._register_metrics_callback()

    def _register_metrics_callback(self) -> None:
        async def track_usage(kwargs, response_obj, start_time, end_time):
            try:
                source = kwargs.get('complete_streaming_response') or response_obj
                usage = getattr(source, 'usage', None) or (source.get('usage') if isinstance(source, dict) else None) or {}
                prompt = (getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0) or usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0) or 0)
                completion = (getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0) or usage.get('completion_tokens', 0) or usage.get('output_tokens', 0) or 0)
                cost = kwargs.get('response_cost', 0) or getattr(response_obj, 'cost', 0) or 0.0
                self.metrics.update(prompt, completion, float(cost))
            except Exception:
                pass
        litellm.success_callback = [track_usage]

    async def run_turn(self, user_input: str) -> str:
        self.conversation.append({"role": "user", "content": user_input})
        final_response: List[str] = []
        tool_cycles = 0

        while tool_cycles < self.config.max_tool_hops:
            assistant_msg = await self._call_llm_streaming(show_header=True)
            if assistant_msg.get("content"):
                final_response.append(assistant_msg["content"])
            self.conversation.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls:
                break

            print(self.theme.sep("Tools request"))
            for i, tc in enumerate(tool_calls, 1):
                name = tc.get("function", {}).get("name", "")
                args = tc.get("function", {}).get("arguments", "{}")
                try:
                    scrub = re.sub(r'(?i)("?(api[_-]?key|authorization|password|token|secret|bearer)"?\s*:\s*")([^"]+)"',
                                   r'\1[REDACTED]"', args)
                except Exception:
                    scrub = args
                print(f"  [{i:02d}] {self.theme.style(name, 'bold', 'blue')} {self.theme.style(scrub[:120], 'gray')}", flush=True)

            results = await self.executor.execute_batch(tool_calls)
            id_order = {tc.get("id", generate_tool_call_id(i, "tc")): i for i, tc in enumerate(tool_calls)}
            results_sorted = sorted(results, key=lambda r: id_order.get(r.tool_call_id, 10**9))

            print(self.theme.sep("Tools results"))
            for i, r in enumerate(results_sorted, 1):
                status = "ok" if r.success else "error"
                status_colored = self.theme.style(status, 'green' if r.success else 'red')
                idx_label = self.theme.style(f"[{i:02d}]", 'bold')
                print(f"← {idx_label} {self.theme.style(r.name, 'bold', 'blue')}: {status_colored} ({r.duration:.2f}s) [{r.lines} lines]", flush=True)
                if self.config.tool_preview_lines > 0 and r.success:
                    preview = '\n'.join(r.content.splitlines()[:self.config.tool_preview_lines])
                    if preview:
                        print(self.theme.style(preview, 'gray'), flush=True)

            for r in results_sorted:
                self.conversation.append(r.to_message())

            tool_cycles += 1

        if tool_cycles >= self.config.max_tool_hops and 'tool_calls' in locals() and tool_calls:
            final_msg = await self._call_llm_streaming(show_header=True, allow_tools=False)
            if final_msg.get("content"):
                final_response.append(final_msg["content"])
                self.conversation.append({"role": "assistant", "content": final_msg["content"]})
        return "\n".join(final_response)

    async def _call_llm_streaming(self, show_header: bool, allow_tools: bool = True) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self.conversation)

        effective_model = self.config.resolve_effective_model(self.config.model)
        kwargs = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if allow_tools and self.router.tool_specs:
            kwargs["tools"] = self.router.tool_specs
            kwargs["tool_choice"] = "auto"

        retryable = (
            asyncio.TimeoutError,
            ConnectionError,
            litellm.APIError,
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
        )
        for attempt, delay in enumerate(self.config.retry_delays + [None]):
            try:
                return await self._stream_response(kwargs, show_header)
            except retryable as e:
                if delay is None:
                    raise
                short = format_exception(e)
                print(self.theme.style(f"[Retry {attempt+1}] {short} – retrying in {delay:.1f}s", 'yellow'))
                await asyncio.sleep(delay)

    async def _stream_response(self, kwargs: Dict, show_header: bool) -> Dict[str, Any]:
        parser = StreamParser()
        stream = await litellm.acompletion(**kwargs)
        header_shown = False
        label = self.theme.label("[Assistant]", "cyan") + " "
        try:
            async for chunk in stream:
                content = parser.process_chunk(chunk)
                if content:
                    if show_header and not header_shown:
                        print("\n" + label, end="", flush=True)
                        header_shown = True
                    print(content, end="", flush=True)
            if header_shown or (show_header and parser.content):
                print(flush=True)
        finally:
            try:
                await stream.aclose()
            except (RuntimeError, AttributeError):
                logger.debug("Stream close error", exc_info=True)
        return parser.get_message()

class PromptUI:
    def __init__(self, theme: AnsiTheme, config: AppConfig, metrics: Metrics, commands: List[str]):
        self.theme = theme
        self.config = config
        self.metrics = metrics
        self._preset_text: Optional[str] = None

        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(commands, ignore_case=True, WORD=True),
        )

        self.kb = KeyBindings()
        self._bind_keys()
        self._undo_callback: Optional[Callable[[], Optional[str]]] = None
        self._redo_callback: Optional[Callable[[], bool]] = None

    def _bind_keys(self) -> None:
        @self.kb.add('c-z')
        def _(event):
            if self._undo_callback:
                text = self._undo_callback() or ""
                buf = event.app.current_buffer
                buf.text = text
                buf.cursor_position = len(text)

        @self.kb.add('c-y')
        def _(event):
            if self._redo_callback:
                self._redo_callback()

    def set_callbacks(self, undo_cb: Callable[[], Optional[str]], redo_cb: Callable[[], bool]) -> None:
        self._undo_callback = undo_cb
        self._redo_callback = redo_cb

    def set_preset_text(self, text: Optional[str]) -> None:
        self._preset_text = text

    def bottom_toolbar(self) -> Any:
        model = self.config.resolve_effective_model(self.config.model)
        tok = f"{self.metrics.prompt_tokens}/{self.metrics.completion_tokens}/{self.metrics.total_tokens}"
        return ANSI(self.theme.style(f" [Model: {model}] [Tokens p/c/t: {tok}] ", 'gray'))

    async def prompt(self, label_text: str) -> str:
        default = self._preset_text or ""
        self._preset_text = None
        return await asyncio.to_thread(
            self.session.prompt,
            ANSI(label_text),
            default=default,
            key_bindings=self.kb,
            bottom_toolbar=self.bottom_toolbar
        )

class CommandHandler:
    COMMANDS = ['/quit', '/exit', '/new', '/tools', '/model', '/reload', '/clean', '/undo', '/redo']

    def __init__(self, orchestrator: LLMOrchestrator, router: MCPRouter, config: AppConfig, theme: AnsiTheme, ui: PromptUI):
        self.orchestrator = orchestrator
        self.router = router
        self.config = config
        self.theme = theme
        self.ui = ui

        self._turn_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []

        self.ui.set_callbacks(self._undo_immediate, self._redo_immediate)

    def _undo_immediate(self) -> Optional[str]:
        """Undo without printing; return prefill text for UI keybinding."""
        if not self._turn_stack:
            return None
        record = self._turn_stack.pop()
        msgs = record["messages"]
        remove_n = len(msgs)
        if remove_n > 0 and len(self.orchestrator.conversation) >= remove_n:
            del self.orchestrator.conversation[-remove_n:]
        self._redo_stack.append(record)
        return record["user_text"]

    def _redo_immediate(self) -> bool:
        """Redo without printing; return True if done."""
        if not self._redo_stack:
            return False
        record = self._redo_stack.pop()
        self.orchestrator.conversation.extend(record["messages"])
        self._turn_stack.append(record)
        return True

    async def handle(self, user_input: str) -> bool:
        if not user_input.strip():
            return True
        if user_input.startswith('/'):
            return await self._handle_command(user_input)

        start_len = len(self.orchestrator.conversation)
        try:
            with patch_stdout(raw=True):
                await self.orchestrator.run_turn(user_input)
        except KeyboardInterrupt:
            print("\n" + self.theme.label("[Info]", "blue") + " Response interrupted")
        except Exception as e:
            logger.debug("Chat error", exc_info=True)
            print(self.theme.style(f"[Error] {format_exception(e)}", 'red'))
            import traceback; traceback.print_exc()
        else:
            end_len = len(self.orchestrator.conversation)
            turn_msgs = self.orchestrator.conversation[start_len:end_len]
            self._turn_stack.append({
                "messages": [m.copy() if isinstance(m, dict) else m for m in turn_msgs],
                "user_text": user_input
            })
            self._redo_stack.clear()
        return True

    async def _handle_command(self, cmd: str) -> bool:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command not in self.COMMANDS:
            print(f"Unknown command: {command}")
            print("Commands: " + ", ".join(self.COMMANDS))
            return True

        if command in ['/quit', '/exit']:
            print("👋 Bye.")
            return False

        elif command == '/new':
            self.orchestrator.conversation.clear()
            self._turn_stack.clear()
            self._redo_stack.clear()
            print("[Info] Conversation reset")

        elif command == '/tools':
            for server, tools in sorted(self.router.get_server_tools_map().items()):
                print(f"  - {server}: {sorted(tools)}")

        elif command == '/model':
            if not args:
                effective = self.config.resolve_effective_model(self.config.model)
                print(f"Current model: {self.config.model} -> {effective}")
                alias_map = getattr(litellm, 'model_alias_map', {}) or {}
                if alias_map:
                    print("Available aliases:")
                    for alias, full in sorted(alias_map.items()):
                        print(f"  {alias} -> {full}")
            else:
                model = args
                effective_model = self.config.resolve_effective_model(model)
                ok, missing = check_provider_auth(effective_model)
                if not ok:
                    print(f"[Error] Missing API keys: {', '.join(missing)}")
                else:
                    self.config.model = model
                    self.config.max_tokens = self.config._apply_token_cap(effective_model, self.config.max_tokens)
                    print(f"[Info] Switched to {model} -> {effective_model}")

        elif command == '/reload':
            print(self.theme.label("[Info]", "blue") + " Reloading MCP servers...")
            try:
                await self.router.cleanup()
                await self.router.load_and_start(MCP_CONFIG_FILE)
                print(f"{self.theme.style('✅ Reloaded', 'green')}: {len(self.router.servers)} servers, {len(self.router.tool_specs)} tools")
            except Exception as e:
                logger.debug("Reload failed", exc_info=True)
                print(self.theme.style(f"[Error] Reload failed: {format_exception(e)}", 'red'))

        elif command == '/clean':
            print("[Info] Cleanup requested")
            return False

        elif command == '/undo':
            text = self._undo_immediate()
            if text is None:
                print(self.theme.style("[Info] Nothing to undo", 'yellow'))
            else:
                self.ui.set_preset_text(text)
                print(self.theme.style("[Info] Undid last turn. Text prefilled for editing.", 'gray'))

        elif command == '/redo':
            if self._redo_immediate():
                print(self.theme.style("[Info] Redid last undone turn.", 'gray'))
            else:
                print(self.theme.style("[Info] Nothing to redo", 'yellow'))

        return True

async def amain(args: argparse.Namespace) -> None:
    cli_args = {
        'model': args.model,
        'provider': None,
        'max_tokens': args.max_tokens,
        'max_tool_hops': args.max_tool_hops,
        'tool_result_max_chars': args.tool_result_max_chars,
        'tool_timeout_seconds': args.tool_timeout_seconds,
        'max_parallel_tools': args.max_parallel_tools,
        'tool_preview_lines': args.tool_preview_lines,
        'system_prompt': args.system_prompt,
        'system_prompt_file': args.system_prompt_file,
        'log_level': args.log_level,
        'log_json': args.log_json,
        'use_color': False if getattr(args, 'no_color', False) else None,
    }
    if args.provider_openai: cli_args['provider'] = 'openai'
    elif args.provider_anthropic: cli_args['provider'] = 'anthropic'
    elif args.provider_gemini: cli_args['provider'] = 'gemini'
    elif args.provider_groq: cli_args['provider'] = 'groq'

    config = AppConfig.load(cli_args=cli_args)
    configure_logging(config)
    litellm.disable_fallback = True
    if hasattr(config, 'model_aliases') and config.model_aliases:
        litellm.model_alias_map = dict(config.model_aliases)
    theme = create_theme(config)

    effective_model = config.resolve_effective_model(config.model)
    ok, missing = check_provider_auth(effective_model)
    if not ok:
        print(f"[Fatal] Missing API credentials: {', '.join(missing)}")
        sys.exit(1)

    print(theme.sep("LiteLLM MCP CLI Chat"))
    print(f"{theme.label('[Model]', 'magenta')} {config.model} -> {effective_model}")
    print(f"{theme.label('[Config]', 'magenta')} {args.config}")
    print(theme.style("Commands: /new, /tools, /model, /reload, /undo (Ctrl+Z), /redo (Ctrl+Y), /quit", 'gray'))

    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"[Fatal] Config file not found: {config_path}")
        print("Create mcp_config.json with your MCP server configuration")
        sys.exit(1)

    metrics = Metrics()
    router = MCPRouter()

    try:
        await router.load_and_start(config_path)
        print(f"{theme.style('✅ Connected', 'green')}: {len(router.servers)} servers, {len(router.tool_specs)} tools")
        for server, tools in sorted(router.get_server_tools_map().items()):
            print(f"  - {theme.style(server, 'cyan')}: {sorted(tools)}")

        if config.system_prompt:
            preview = config.system_prompt[:120] + ("…" if len(config.system_prompt) > 120 else "")
            print("\n" + theme.label("[System]", "yellow") + f" {preview}")

        orchestrator = LLMOrchestrator(config, router, metrics, theme)
        ui = PromptUI(theme, config, metrics, commands=CommandHandler.COMMANDS)
        handler = CommandHandler(orchestrator, router, config, theme, ui)

        while True:
            try:
                user_input = await ui.prompt("\n" + theme.label("[You]", "green") + " ")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye.")
                break

            keep_going = await handler.handle(user_input)
            if not keep_going:
                break

    finally:
        print("\n" + theme.label("[Info]", "blue") + " Cleaning up MCP connections...")
        try:
            await router.cleanup()
        except Exception as e:
            logger.warning("Cleanup error: %r", e)
        try:
            await asyncio.sleep(0.05)
        except (asyncio.CancelledError, RuntimeError):
            pass
        print(f"{theme.label('[Metrics]', 'magenta')} {metrics.summary()}")

def main():
    parser = argparse.ArgumentParser(description="MCP chat (LiteLLM + prompt_toolkit)")
    parser.add_argument("--config", default=str(MCP_CONFIG_FILE), help="Path to mcp_config.json")
    parser.add_argument("--model", "-model", "-m", "--m", help="Model name or alias")
    parser.add_argument("-o", dest="provider_openai", action="store_true", help="Use OpenAI")
    parser.add_argument("-a", dest="provider_anthropic", action="store_true", help="Use Anthropic")
    parser.add_argument("-g", dest="provider_gemini", action="store_true", help="Use Google Gemini")
    parser.add_argument("-q", dest="provider_groq", action="store_true", help="Use Groq")
    parser.add_argument("--max-tokens", type=int, help="Max response tokens")
    parser.add_argument("--max-tool-hops", type=int, help="Max tool iterations per turn")
    parser.add_argument("--tool-result-max-chars", type=int, help="Truncate tool results to N characters")
    parser.add_argument("--tool-timeout-seconds", type=float, help="Tool execution timeout")
    parser.add_argument("--max-parallel-tools", type=int, help="Max parallel tool executions (0=serial)")
    parser.add_argument("--tool-preview-lines", type=int, help="Preview first N lines of tool results")
    parser.add_argument("--system-prompt", help="System prompt text")
    parser.add_argument("--system-prompt-file", help="Path to system prompt file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log-json", action="store_true", help="Output JSON logs to stderr")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")

    args = parser.parse_args()
    try:
        asyncio.run(amain(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Info] Terminated")

if __name__ == "__main__":
    main()
