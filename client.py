# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic",
#     "google",
#     "litellm",
#     "mcp",
#     "openai",
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
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import AsyncExitStack

import litellm
from litellm.integrations.custom_logger import CustomLogger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import AppConfig, Metrics, check_provider_auth

# Constants
DEFAULT_TERMINAL_WIDTH = 80
MAX_EXCEPTION_LENGTH = 160
PREVIEW_LENGTH = 120
DEFAULT_TIMEOUT = 30.0
RETRY_DELAYS = [1.0, 3.0, 5.0]

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MCP_CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

logger = logging.getLogger(__name__)

class AnsiTheme:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.RESET = "\x1b[0m" if enabled else ""
        self.BOLD = "\x1b[1m" if enabled else ""
        self.DIM = "\x1b[2m" if enabled else ""
        self.RED = "\x1b[31m" if enabled else ""
        self.GREEN = "\x1b[32m" if enabled else ""
        self.YELLOW = "\x1b[33m" if enabled else ""
        self.BLUE = "\x1b[34m" if enabled else ""
        self.MAGENTA = "\x1b[35m" if enabled else ""
        self.CYAN = "\x1b[36m" if enabled else ""
        self.GRAY = "\x1b[90m" if enabled else ""

    def _wrap(self, text: str, code: str) -> str:
        if not self.enabled or not text:
            return text
        return f"{code}{text}{self.RESET}"

    def bold(self, text: str) -> str:
        return self._wrap(text, self.BOLD)

    def dim(self, text: str) -> str:
        return self._wrap(text, self.DIM)

    def red(self, text: str) -> str:
        return self._wrap(text, self.RED)

    def green(self, text: str) -> str:
        return self._wrap(text, self.GREEN)

    def yellow(self, text: str) -> str:
        return self._wrap(text, self.YELLOW)

    def blue(self, text: str) -> str:
        return self._wrap(text, self.BLUE)

    def magenta(self, text: str) -> str:
        return self._wrap(text, self.MAGENTA)

    def cyan(self, text: str) -> str:
        return self._wrap(text, self.CYAN)

    def gray(self, text: str) -> str:
        return self._wrap(text, self.GRAY)

    def label(self, text: str, color: str = "cyan") -> str:
        colorer = getattr(self, color, self.cyan)
        return colorer(self.bold(text))

    def sep(self, title: Optional[str] = None, char: str = "─") -> str:
        width = get_terminal_width()
        if title:
            title_text = f" {title} "
            side = (width - len(title_text)) // 2
            line = char * side + title_text + char * (width - side - len(title_text))
            return self.gray(line)
        return self.gray(char * width)

def get_terminal_width() -> int:
    """Get terminal width with consistent fallback."""
    try:
        width = shutil.get_terminal_size(fallback=(DEFAULT_TERMINAL_WIDTH, 20)).columns
        return max(20, min(width, 120))
    except Exception:
        return DEFAULT_TERMINAL_WIDTH

def format_exception(exc: BaseException, limit: int = MAX_EXCEPTION_LENGTH) -> str:
    """Format exception with length limit."""
    name = type(exc).__name__
    text = str(exc)[:limit]
    return f"{name}: {text}" if text else name

def configure_logging(config: AppConfig) -> None:
    level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        force=True
    )

    for lib in ['httpx', 'anyio', 'mcp', 'urllib3', 'openai', 'LiteLLM', 'litellm']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    litellm.set_verbose = False

def create_theme(config: AppConfig) -> AnsiTheme:
    """Create theme based on config and terminal capabilities."""
    enabled = bool(getattr(config, 'use_color', True) and sys.stdout.isatty())
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
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(params)
            )
            read, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
            logger.info("Server '%s' initialized", self.name)
        except Exception as e:
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
        except Exception as e:
            msg = str(e)
            benign = (
                isinstance(e, (RuntimeError, OSError, ConnectionError)) and any(s in msg for s in (
                    "cancel scope",
                    "Event loop is closed",
                    "already closed",
                    "attached to a different loop",
                    "cannot schedule new futures",
                    "I/O operation on closed file",
                    "Proactor"
                ))
            )
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

        results = await asyncio.gather(
            *[self._init_and_list(s) for s in self.servers]
        )
        
        for server, tools_resp in results:
            self._register_server_tools(server, tools_resp)
        
        if not self.tool_specs:
            logger.warning("No tools found from any MCP server")
        else:
            logger.info("Loaded %d tools from %d servers", 
                       len(self.tool_specs), len(self.servers))
    
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
        result = {}
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
                    servers.setdefault(name, {
                        "command": cmd,
                        "args": [str(py.resolve())],
                    })
            except Exception as e:
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
        
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": meta + self.content
        }

class ToolExecutor:
    def __init__(self, router: MCPRouter, config: AppConfig):
        self.router = router
        self.config = config
    
    async def execute_batch(self, tool_calls: List[Dict]) -> List[ToolResult]:
        results = []
        limit = max(1, self.config.max_parallel_tools or 1)
        sem = asyncio.Semaphore(limit)
        
        tasks = []
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            args_str = tc.get("function", {}).get("arguments", "{}")
            tool_call_id = tc.get("id", f"tc_{len(tasks)}")
            tasks.append(self._execute_single(name, args_str, tool_call_id, sem))

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in task_results:
                if isinstance(result, Exception):
                    logger.error("Tool execution error: %r", result)
                else:
                    results.append(result)
        
        return results
    
    async def _execute_single(self, name: str, args_str: Any,
                             tool_call_id: str, sem: asyncio.Semaphore) -> ToolResult:
        # Parse arguments
        try:
            if isinstance(args_str, str):
                args = json.loads(args_str) if args_str.strip() else {}
            elif isinstance(args_str, dict):
                args = args_str
            else:
                args = {}
        except json.JSONDecodeError:
            return ToolResult(
                name=name,
                tool_call_id=tool_call_id,
                content=f"ERROR: Invalid JSON arguments: {args_str[:200]}",
                success=False,
                duration=0.0
            )
        
        async with sem:
            start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self.router.call_tool(name, args),
                    timeout=self.config.tool_timeout_seconds
                )
                duration = time.monotonic() - start

                content, truncated, lines = self._format_result(result)
                
                return ToolResult(
                    name=name,
                    tool_call_id=tool_call_id,
                    content=content,
                    success=True,
                    duration=duration,
                    lines=lines,
                    truncated=truncated
                )
                
            except asyncio.TimeoutError:
                return ToolResult(
                    name=name,
                    tool_call_id=tool_call_id,
                    content=f"ERROR: Timeout after {self.config.tool_timeout_seconds}s",
                    success=False,
                    duration=self.config.tool_timeout_seconds
                )
            except Exception as e:
                return ToolResult(
                    name=name,
                    tool_call_id=tool_call_id,
                    content=f"ERROR: {e!r}",
                    success=False,
                    duration=time.monotonic() - start
                )
    
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
            
        except Exception:
            text = str(result)
            lines = len(text.splitlines())
            if len(text) > self.config.tool_result_max_chars:
                text = text[:self.config.tool_result_max_chars]
                return text, True, lines
            return text, False, lines

class StreamParser:
    def __init__(self):
        self.content_buffer: List[str] = []
        self.tool_calls: Dict[int, Dict] = {}
        self.tool_args_buffer: Dict[int, List[str]] = {}
    
    def process_chunk(self, chunk: Any) -> Optional[str]:
        try:
            delta = chunk.choices[0].delta

            # Handle content
            content = self._get_attr(delta, 'content')
            if content:
                self.content_buffer.append(content)
                return content

            # Handle tool calls
            tool_calls = self._get_attr(delta, 'tool_calls')
            if tool_calls:
                self._process_tool_calls(tool_calls)
            
            return None
                
        except (AttributeError, IndexError, KeyError):
            return None
    
    def _get_attr(self, obj: Any, attr: str) -> Any:
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr)
        return None
    
    def _process_tool_calls(self, tool_calls: List) -> None:
        for tc in tool_calls:
            idx = self._get_attr(tc, 'index')
            if idx is None:
                idx = len(self.tool_calls)
            
            if idx not in self.tool_calls:
                self.tool_calls[idx] = {
                    'id': None,
                    'type': 'function',
                    'function': {'name': None, 'arguments': ''}
                }
                self.tool_args_buffer[idx] = []

            tc_id = self._get_attr(tc, 'id')
            if tc_id:
                self.tool_calls[idx]['id'] = tc_id

            tc_type = self._get_attr(tc, 'type')
            if tc_type:
                self.tool_calls[idx]['type'] = tc_type

            func = self._get_attr(tc, 'function')
            if func:
                name = self._get_attr(func, 'name')
                if name:
                    self.tool_calls[idx]['function']['name'] = name
                
                args = self._get_attr(func, 'arguments')
                if args is not None:
                    if isinstance(args, str):
                        self.tool_args_buffer[idx].append(args)
                    else:
                        try:
                            self.tool_args_buffer[idx].append(json.dumps(args, ensure_ascii=False))
                        except Exception:
                            self.tool_args_buffer[idx].append(str(args))
    
    def get_message(self) -> Dict[str, Any]:
        message = {
            'role': 'assistant',
            'content': ''.join(self.content_buffer).strip() or None
        }

        if self.tool_calls:
            tool_calls = []
            for idx in sorted(self.tool_calls.keys()):
                tc = self.tool_calls[idx].copy()
                if 'function' not in tc:
                    tc['function'] = {'name': None, 'arguments': ''}
                
                # Join argument chunks
                args_parts = self.tool_args_buffer.get(idx, [])
                tc['function']['arguments'] = ''.join(args_parts)
                
                # Ensure ID exists
                if not tc.get('id'):
                    tc['id'] = f'call_{idx}'
                
                tool_calls.append(tc)
            
            message['tool_calls'] = tool_calls
        
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
        orchestrator_metrics = self.metrics

        class _MetricsLogger(CustomLogger):
            def __init__(self, metrics: Metrics):
                self.metrics = metrics

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                try:
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_cost = 0.0

                    # Prefer complete_streaming_response for streaming calls
                    full_stream_resp = kwargs.get('complete_streaming_response')
                    usage_source = full_stream_resp or response_obj

                    # 1) Try usage on object
                    try:
                        usage_obj = getattr(usage_source, 'usage', None)
                        if usage_obj:
                            prompt_tokens = getattr(usage_obj, 'prompt_tokens', 0) or getattr(usage_obj, 'input_tokens', 0) or 0
                            completion_tokens = getattr(usage_obj, 'completion_tokens', 0) or getattr(usage_obj, 'output_tokens', 0) or 0
                    except Exception:
                        pass

                    # 2) Try dict-style usage
                    if (prompt_tokens == 0 and completion_tokens == 0) and isinstance(usage_source, dict):
                        usage_dict = usage_source.get('usage') or {}
                        if isinstance(usage_dict, dict):
                            prompt_tokens = usage_dict.get('prompt_tokens') or usage_dict.get('input_tokens') or 0
                            completion_tokens = usage_dict.get('completion_tokens') or usage_dict.get('output_tokens') or 0

                    # 3) Try kwargs fallbacks (LiteLLM sets these on success)
                    if prompt_tokens == 0:
                        prompt_tokens = kwargs.get('prompt_tokens') or kwargs.get('input_tokens') or 0
                    if completion_tokens == 0:
                        completion_tokens = kwargs.get('completion_tokens') or kwargs.get('output_tokens') or 0

                    # 4) Cost from kwargs/response
                    total_cost = kwargs.get('response_cost') or getattr(response_obj, 'cost', 0) or 0.0

                    # 5) Fallback – compute prompt tokens locally if still zero
                    if prompt_tokens == 0:
                        try:
                            model_for_count = kwargs.get('model')
                            messages_for_count = kwargs.get('messages') or []
                            if model_for_count and messages_for_count:
                                prompt_tokens = litellm.token_counter(model=model_for_count, messages=messages_for_count) or 0
                        except Exception:
                            pass

                    # 6) Fallback – compute completion tokens from full streamed content if available
                    if completion_tokens == 0:
                        try:
                            model_for_count = kwargs.get('model')
                            assistant_text = None
                            if isinstance(usage_source, dict):
                                # OpenAI-style
                                try:
                                    choices = usage_source.get('choices') or []
                                    if choices:
                                        choice0 = choices[0] or {}
                                        msg = choice0.get('message') or {}
                                        assistant_text = msg.get('content') or assistant_text
                                        if not assistant_text:
                                            delta = choice0.get('delta') or {}
                                            assistant_text = delta.get('content') or assistant_text
                                except Exception:
                                    pass
                                # Anthropic/Groq-style
                                if assistant_text is None:
                                    content = usage_source.get('content')
                                    if isinstance(content, list) and content:
                                        # [{'type': 'text', 'text': '...'}]
                                        first = content[0]
                                        assistant_text = first.get('text') if isinstance(first, dict) else None
                                    elif isinstance(content, str):
                                        assistant_text = content
                            # Compute token count for assistant text
                            if model_for_count and assistant_text:
                                completion_tokens = litellm.token_counter(model=model_for_count, text=assistant_text) or \
                                                    litellm.token_counter(model=model_for_count, messages=[{"role": "assistant", "content": assistant_text}]) or 0
                        except Exception:
                            pass

                    if prompt_tokens or completion_tokens or total_cost:
                        orchestrator_metrics.update(prompt_tokens, completion_tokens, float(total_cost or 0.0))
                except Exception:
                    pass

            async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
                # no-op; could record failures if desired
                return

        # Register the custom logger (overrides previous callbacks to avoid duplicates)
        try:
            litellm.callbacks = [_MetricsLogger(orchestrator_metrics)]
        except Exception:
            # Fallback to success_callback with async function if CustomLogger import fails
            async def async_callback(kwargs, response_obj, start_time, end_time):
                try:
                    usage = (getattr(response_obj, 'usage', None) or
                             (response_obj.get('usage') if isinstance(response_obj, dict) else None)) or {}
                    prompt = getattr(usage, 'prompt_tokens', 0) or (usage.get('prompt_tokens') if isinstance(usage, dict) else 0) or 0
                    completion = getattr(usage, 'completion_tokens', 0) or (usage.get('completion_tokens') if isinstance(usage, dict) else 0) or 0
                    if prompt == 0:
                        try:
                            prompt = litellm.token_counter(model=kwargs.get('model'), messages=kwargs.get('messages') or []) or 0
                        except Exception:
                            pass
                    cost = kwargs.get('response_cost') or getattr(response_obj, 'cost', 0) or 0.0
                    orchestrator_metrics.update(prompt, completion, float(cost or 0.0))
                except Exception:
                    pass
            litellm.success_callback = [async_callback]
    
    async def run_turn(self, user_input: str) -> str:
        self.conversation.append({"role": "user", "content": user_input})

        final_response = []
        tool_cycles = 0

        while tool_cycles < self.config.max_tool_hops:
            show_header = True
            assistant_msg = await self._call_llm_streaming(show_header)

            if assistant_msg.get("content"):
                final_response.append(assistant_msg["content"])

            self.conversation.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls:
                break

            # Display tool requests
            print("\n" + self.theme.sep("Tools request"))
            for i, tc in enumerate(tool_calls, 1):
                name = tc.get("function", {}).get("name", "")
                args = tc.get("function", {}).get("arguments", "{}")
                print(f"  [{i:02d}] {self.theme.bold(self.theme.blue(name))} {self.theme.gray(args)}", flush=True)
            
            # Execute tools
            results = await self.executor.execute_batch(tool_calls)

            # Sort results
            id_order = {tc.get("id", f"call_{i}"): i for i, tc in enumerate(tool_calls)}
            results_sorted = sorted(results, key=lambda r: id_order.get(r.tool_call_id, 10**9))

            # Display results
            print(self.theme.sep("Tools results"))
            for i, r in enumerate(results_sorted, 1):
                status = "ok" if r.success else "error"
                status_colored = self.theme.green(status) if r.success else self.theme.red(status)
                idx_label = self.theme.bold(f"[{i:02d}]")
                print(f"← {idx_label} {self.theme.bold(self.theme.blue(r.name))}: {status_colored} ({r.duration:.2f}s) [{r.lines} lines]", flush=True)

                if self.config.tool_preview_lines > 0 and r.success:
                    preview = '\n'.join(r.content.splitlines()[:self.config.tool_preview_lines])
                    if preview:
                        print(self.theme.gray(preview), flush=True)

            for result in results_sorted:
                self.conversation.append(result.to_message())

            tool_cycles += 1

        if tool_cycles >= self.config.max_tool_hops and tool_calls:
            final_msg = await self._call_llm_streaming(True, allow_tools=False)
            if final_msg.get("content"):
                final_response.append(final_msg["content"])
                self.conversation.append({"role": "assistant", "content": final_msg["content"]})
        
        return "\n".join(final_response)
    
    async def _call_llm_streaming(self, show_header: bool, allow_tools: bool = True) -> Dict[str, Any]:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self.conversation)

        # Resolve model aliases locally to avoid provider detection issues upstream
        effective_model = self.config.resolve_effective_model(self.config.model)

        kwargs = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            # Ensure providers that support it (e.g., OpenAI) include usage on the final stream chunk
            "stream_options": {"include_usage": True}
        }

        if allow_tools and self.router.tool_specs:
            kwargs["tools"] = self.router.tool_specs
            kwargs["tool_choice"] = "auto"

        for attempt, delay in enumerate(RETRY_DELAYS + [None]):
            try:
                return await self._stream_response(kwargs, show_header)
            except Exception as e:
                if delay is None:
                    raise
                short = format_exception(e)
                print(self.theme.yellow(f"[Retry {attempt+1}] {short} – retrying in {delay:.1f}s"))
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
                    # Just print content directly as it streams
                    print(content, end="", flush=True)

            # Final newline if content was shown
            if header_shown or (show_header and parser.content_buffer):
                print(flush=True)
                
        finally:
            try:
                await stream.aclose()
            except Exception:
                logger.debug("Stream close error", exc_info=True)
        
        return parser.get_message()

class CommandHandler:
    def __init__(self, orchestrator: LLMOrchestrator, router: MCPRouter, config: AppConfig, theme: AnsiTheme):
        self.orchestrator = orchestrator
        self.router = router
        self.config = config
        self.theme = theme
        
        # Valid commands for input validation
        self.valid_commands = {
            '/quit', '/exit', '/new', '/tools', '/model', '/reload', '/clean'
        }
    
    async def handle(self, user_input: str) -> bool:
        if not user_input.strip():
            return True

        if user_input.startswith('/'):
            return await self._handle_command(user_input)

        try:
            await self.orchestrator.run_turn(user_input)
        except KeyboardInterrupt:
            print("\n" + self.theme.label("[Info]", "blue") + " Response interrupted")
        except Exception as e:
            logger.debug("Chat error", exc_info=True)
            print(self.theme.red(f"[Error] {format_exception(e)}"))
        
        return True
    
    async def _handle_command(self, cmd: str) -> bool:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Validate command
        if command not in self.valid_commands:
            print(f"Unknown command: {command}")
            print("Commands: /new, /tools, /model, /reload, /quit")
            return True

        if command in ['/quit', '/exit']:
            print("👋 Bye.")
            return False

        elif command == '/new':
            self.orchestrator.conversation.clear()
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
                # Just set the model directly - LiteLLM will resolve aliases
                model = args
                # For auth checking, resolve alias to check the right provider
                effective_model = self.config.resolve_effective_model(model)
                ok, missing = check_provider_auth(effective_model)
                if not ok:
                    print(f"[Error] Missing API keys: {', '.join(missing)}")
                else:
                    self.config.model = model  # Store the alias/model as-is
                    # Apply token cap based on resolved model
                    self.config.max_tokens = self.config._apply_token_cap(effective_model, self.config.max_tokens)
                    print(f"[Info] Switched to {model} -> {effective_model}")
        
        elif command == '/reload':
            print(self.theme.label("[Info]", "blue") + " Reloading MCP servers...")
            try:
                await self.router.cleanup()
                await self.router.load_and_start(MCP_CONFIG_FILE)
                print(f"{self.theme.green('✅ Reloaded')}: {len(self.router.servers)} servers, {len(self.router.tool_specs)} tools")
            except Exception as e:
                logger.debug("Reload failed", exc_info=True)
                print(self.theme.red(f"[Error] Reload failed: {format_exception(e)}"))
        
        elif command == '/clean':
            print("[Info] Cleanup requested")
            return False
        
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

    if args.provider_openai:
        cli_args['provider'] = 'openai'
    elif args.provider_anthropic:
        cli_args['provider'] = 'anthropic'
    elif args.provider_gemini:
        cli_args['provider'] = 'gemini'
    elif args.provider_groq:
        cli_args['provider'] = 'groq'

    config = AppConfig.load(cli_args=cli_args)
    configure_logging(config)
    
    # Set model alias map immediately after loading config
    # This ensures LiteLLM will resolve aliases in all subsequent calls
    if hasattr(config, 'model_aliases') and config.model_aliases:
        litellm.model_alias_map = dict(config.model_aliases)
    
    # Create theme based on config
    theme = create_theme(config)

    # Check auth using the resolved model
    effective_model = config.resolve_effective_model(config.model)
    ok, missing = check_provider_auth(effective_model)
    if not ok:
        print(f"[Fatal] Missing API credentials: {', '.join(missing)}")
        sys.exit(1)

    print(theme.sep("LiteLLM MCP CLI Chat"))
    print(f"{theme.label('[Model]', 'magenta')} {config.model} -> {effective_model}")
    print(f"{theme.label('[Config]', 'magenta')} {args.config}")
    print(theme.gray("Type '/quit' or '/exit' to exit. Commands: /new, /history, /tools, /model, /reload, /quit"))

    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"[Fatal] Config file not found: {config_path}")
        print("Create mcp_config.json with your MCP server configuration")
        sys.exit(1)

    metrics = Metrics()
    router = MCPRouter()
    
    try:
        await router.load_and_start(config_path)
        print(f"{theme.green('✅ Connected')}: {len(router.servers)} servers, {len(router.tool_specs)} tools")

        for server, tools in sorted(router.get_server_tools_map().items()):
            print(f"  - {theme.cyan(server)}: {sorted(tools)}")

        if config.system_prompt:
            preview = config.system_prompt[:PREVIEW_LENGTH]
            if len(config.system_prompt) > PREVIEW_LENGTH:
                preview += "…"
            print("\n" + theme.label("[System]", "yellow") + f" {preview}")

        orchestrator = LLMOrchestrator(config, router, metrics, theme)
        handler = CommandHandler(orchestrator, router, config, theme)

        while True:
            try:
                user_input = input("\n" + theme.label("[You]", "green") + " ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye.")
                break
            
            if not await handler.handle(user_input):
                break

    finally:
        print("\n" + theme.label("[Info]", "blue") + " Cleaning up MCP connections...")
        try:
            await router.cleanup()
        except Exception as e:
            logger.warning("Cleanup error: %r", e)
        # Give background tasks/transports a brief moment to settle on shutdown (Windows/proactor)
        try:
            await asyncio.sleep(0.05)
        except Exception:
            pass

        print(f"{theme.label('[Metrics]', 'magenta')} {metrics.summary()}")

def main():
    parser = argparse.ArgumentParser(description="Minimal MCP client with LiteLLM integration")
    parser.add_argument("--config", default=str(MCP_CONFIG_FILE), help="Path to mcp_config.json")
    parser.add_argument("--model", help="Model name or alias")
    parser.add_argument("-o", dest="provider_openai", action="store_true", help="Use OpenAI (gpt-4o-mini)")
    parser.add_argument("-a", dest="provider_anthropic", action="store_true", help="Use Anthropic (Claude Sonnet)")
    parser.add_argument("-g", dest="provider_gemini", action="store_true", help="Use Google Gemini")
    parser.add_argument("-q", dest="provider_groq", action="store_true", help="Use Groq (Llama 70B)")
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