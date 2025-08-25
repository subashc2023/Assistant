#!/usr/bin/env python3
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
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import AsyncExitStack

import litellm
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import AppConfig, Metrics, check_provider_auth

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MCP_CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"
APP_CONFIG_FILE = SCRIPT_DIR / "tinyclient_config.yaml"

logger = logging.getLogger(__name__)

def configure_logging(config: AppConfig) -> None:
    level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        force=True
    )

    for lib in ['httpx', 'anyio', 'mcp', 'urllib3', 'openai', 'LiteLLM', 'litellm']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    try:
        litellm.set_verbose = False
    except Exception:
        pass

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
            if not (isinstance(e, RuntimeError) and "cancel scope" in str(e)):
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
        if not isinstance(raw_servers, dict) or not raw_servers:
            raise ValueError("mcp_config.json must contain 'mcpServers' object")
        
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
        self.seen_calls: set = set()
    
    async def execute_batch(self, tool_calls: List[Dict]) -> List[ToolResult]:
        results = []
        tasks = []
        self.seen_calls.clear()

        sem = None
        if self.config.max_parallel_tools > 0:
            sem = asyncio.Semaphore(self.config.max_parallel_tools)
        
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            args_str = tc.get("function", {}).get("arguments", "{}")
            tool_call_id = tc.get("id", f"tc_{len(tasks)}")


            call_key = (name, args_str)
            if call_key in self.seen_calls:
                results.append(ToolResult(
                    name=name,
                    tool_call_id=tool_call_id,
                    content="ERROR: Duplicate tool call prevented",
                    success=False,
                    duration=0.0
                ))
                continue
            
            self.seen_calls.add(call_key)
            tasks.append(self._execute_single(name, args_str, tool_call_id, sem))

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in task_results:
                if isinstance(result, Exception):
                    logger.error("Tool execution error: %r", result)
                else:
                    results.append(result)
        
        return results
    
    async def _execute_single(self, name: str, args_str: str,
                             tool_call_id: str, sem: Optional[asyncio.Semaphore]) -> ToolResult:
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            return ToolResult(
                name=name,
                tool_call_id=tool_call_id,
                content="ERROR: Invalid JSON arguments",
                success=False,
                duration=0.0
            )
        
        async def _run():
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

        if sem:
            async with sem:
                return await _run()
        else:
            return await _run()
    
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
                return text, True, lines
            
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

            content = self._get_attr(delta, 'content')
            if content:
                self.content_buffer.append(content)
                return content

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
            idx = self._get_attr(tc, 'index') or 0

            if idx not in self.tool_calls:
                self.tool_calls[idx] = {
                    'id': None,
                    'type': 'function',
                    'function': {'name': None, 'arguments': ''}
                }
                self.tool_args_buffer[idx] = []

            if tc_id := self._get_attr(tc, 'id'):
                self.tool_calls[idx]['id'] = tc_id

            if tc_type := self._get_attr(tc, 'type'):
                self.tool_calls[idx]['type'] = tc_type

            if func := self._get_attr(tc, 'function'):
                if name := self._get_attr(func, 'name'):
                    self.tool_calls[idx]['function']['name'] = name
                if args := self._get_attr(func, 'arguments'):
                    self.tool_args_buffer[idx].append(args)
    
    def get_message(self) -> Dict[str, Any]:
        message = {
            'role': 'assistant',
            'content': ''.join(self.content_buffer)
        }

        if self.tool_calls:
            tool_calls = []
            for idx in sorted(self.tool_calls.keys()):
                tc = self.tool_calls[idx].copy()
                tc['id'] = tc.get('id') or f'call_{idx}'
                tc['function']['arguments'] = ''.join(self.tool_args_buffer.get(idx, []))
                tool_calls.append(tc)
            
            message['tool_calls'] = tool_calls
        
        return message


class LLMOrchestrator:

    def __init__(self, config: AppConfig, router: MCPRouter, metrics: Metrics):
        self.config = config
        self.router = router
        self.metrics = metrics
        self.executor = ToolExecutor(router, config)
        self.conversation: List[Dict[str, Any]] = []

        self._register_metrics_callback()
    
    def _register_metrics_callback(self) -> None:
        def callback(kwargs, response, start_time, end_time):
            try:
                usage = getattr(response, 'usage', None) or response.get('usage')
                if usage:
                    prompt = getattr(usage, 'prompt_tokens', 0) or usage.get('prompt_tokens', 0)
                    completion = getattr(usage, 'completion_tokens', 0) or usage.get('completion_tokens', 0)
                    cost = kwargs.get('response_cost') or getattr(response, 'cost', 0)
                    self.metrics.update(prompt, completion, cost)
            except Exception:
                pass
        
        try:
            callbacks = getattr(litellm, 'success_callback', [])
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            callbacks.append(callback)
            litellm.success_callback = callbacks
        except Exception:
            pass
    
    async def run_turn(self, user_input: str) -> str:
        self.conversation.append({"role": "user", "content": user_input})

        final_response = []
        tool_cycles = 0

        while tool_cycles < self.config.max_tool_hops:
            show_header = (tool_cycles == 0)
            assistant_msg = await self._call_llm_streaming(show_header)

            if assistant_msg.get("content"):
                final_response.append(assistant_msg["content"])

            self.conversation.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls:
                break

            print("\n[Tools] request:", flush=True)
            for i, tc in enumerate(tool_calls, 1):
                name = tc.get("function", {}).get("name", "")
                args = tc.get("function", {}).get("arguments", "{}")
                print(f"  [{i}] {name} {args}", flush=True)
            
            results = await self.executor.execute_batch(tool_calls)

            print("[Tools] results:", flush=True)
            for r in results:
                status = "ok" if r.success else "error"
                print(f"← {r.name}: {status} ({r.duration:.2f}s) [{r.lines} lines]", flush=True)

                if self.config.tool_preview_lines > 0 and r.success:
                    preview = '\n'.join(r.content.splitlines()[:self.config.tool_preview_lines])
                    if preview:
                        print(preview, flush=True)

            for result in results:
                self.conversation.append(result.to_message())

            print("\n[Assistant]", end=" ", flush=True)
            tool_cycles += 1

        if tool_cycles >= self.config.max_tool_hops and tool_calls:
            final_msg = await self._call_llm_streaming(False, allow_tools=False)
            if final_msg.get("content"):
                final_response.append(final_msg["content"])
                self.conversation.append({"role": "assistant", "content": final_msg["content"]})
        
        return "\n".join(final_response)
    
    async def _call_llm_streaming(self, show_header: bool, allow_tools: bool = True) -> Dict[str, Any]:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self.conversation)

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": True
        }

        if allow_tools and self.router.tool_specs:
            kwargs["tools"] = self.router.tool_specs
            kwargs["tool_choice"] = "auto"

        retry_delays = [1.0, 3.0, 5.0]
        for attempt, delay in enumerate(retry_delays + [None]):
            try:
                return await self._stream_response(kwargs, show_header)
            except Exception as e:
                if delay is None:  # Last attempt
                    raise
                logger.warning("LLM error (attempt %d): %r - retrying in %.1fs", 
                             attempt + 1, e, delay)
                await asyncio.sleep(delay)
    
    async def _stream_response(self, kwargs: Dict, show_header: bool) -> Dict[str, Any]:
        parser = StreamParser()
        stream = await litellm.acompletion(**kwargs)
        header_shown = False
        suppress_content = False

        try:
            async for chunk in stream:
                content = parser.process_chunk(chunk)

                if parser.tool_calls and not suppress_content:
                    suppress_content = True

                if content and not suppress_content:
                    if show_header and not header_shown:
                        print("\n[Assistant]", end=" ", flush=True)
                        header_shown = True
                    print(content, end="", flush=True)

            if header_shown or (show_header and not suppress_content and not parser.tool_calls):
                if not header_shown and show_header:
                    print("\n[Assistant]", end=" ", flush=True)
                print(flush=True)
                
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass
        
        return parser.get_message()

class CommandHandler:
    
    def __init__(self, orchestrator: LLMOrchestrator, router: MCPRouter, config: AppConfig):
        self.orchestrator = orchestrator
        self.router = router
        self.config = config
    
    async def handle(self, user_input: str) -> bool:
        if not user_input.strip():
            return True

        if user_input.startswith('/'):
            return await self._handle_command(user_input)

        try:
            await self.orchestrator.run_turn(user_input)
        except KeyboardInterrupt:
            print("\n[Info] Response interrupted")
        except Exception as e:
            logger.error("Chat error: %r", e, exc_info=True)
            print(f"[Error] {e}")
        
        return True
    
    async def _handle_command(self, cmd: str) -> bool:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ['/quit', '/exit']:
            print("👋 Bye.")
            return False

        elif command == '/new':
            self.orchestrator.conversation.clear()
            print("[Info] Conversation reset")

        elif command == '/history':
            print(json.dumps(self.orchestrator.conversation, indent=2, ensure_ascii=False))

        elif command == '/tools':
            for server, tools in sorted(self.router.get_server_tools_map().items()):
                print(f"  - {server}: {sorted(tools)}")

        elif command == '/model':
            if not args:
                print(f"Current model: {self.config.model}")
                if self.config.model_aliases:
                    print("Available aliases:")
                    for alias, full in sorted(self.config.model_aliases.items()):
                        print(f"  {alias} -> {full}")
            else:
                model = self.config.model_aliases.get(args, args)
                ok, missing = check_provider_auth(model)
                if not ok:
                    print(f"[Error] Missing API keys: {', '.join(missing)}")
                else:
                    self.config.model = model
                    self.config.max_tokens = self.config._apply_token_cap(model, self.config.max_tokens)
                    print(f"[Info] Switched to {model}")
        
        elif command == '/reload':
            print("[Info] Reloading MCP servers...")
            try:
                await self.router.cleanup()
                await self.router.load_and_start(MCP_CONFIG_FILE)
                print(f"✅ Reloaded {len(self.router.servers)} servers, {len(self.router.tool_specs)} tools")
            except Exception as e:
                logger.error("Reload failed: %r", e, exc_info=True)
                print(f"[Error] Reload failed: {e}")
        
        elif command == '/clean':
            print("[Info] Cleanup requested")
            return False
        
        else:
            print(f"Unknown command: {command}")
            print("Commands: /new, /history, /tools, /model, /reload, /quit")
        
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
    }

    if args.provider_openai:
        cli_args['provider'] = 'openai'
    elif args.provider_anthropic:
        cli_args['provider'] = 'anthropic'
    elif args.provider_gemini:
        cli_args['provider'] = 'gemini'
    elif args.provider_groq:
        cli_args['provider'] = 'groq'

    config = AppConfig.load(
        yaml_path=APP_CONFIG_FILE,
        cli_args=cli_args,
        legacy_prompt_path=SCRIPT_DIR / "prompt.txt"
    )

    configure_logging(config)

    ok, missing = check_provider_auth(config.model)
    if not ok:
        print(f"[Fatal] Missing API credentials: {', '.join(missing)}")
        sys.exit(1)

    print("=== LiteLLM MCP CLI Chat ===")
    print(f"Model: {config.model}")
    print(f"Config: {args.config}")
    print("Type 'quit' to exit. Commands: /new, /history, /tools, /model, /reload\n")

    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"[Fatal] Config file not found: {config_path}")
        print("Create mcp_config.json with your MCP server configuration")
        sys.exit(1)

    metrics = Metrics()
    router = MCPRouter()
    
    try:
        await router.load_and_start(config_path)
        print(f"✅ Connected: {len(router.servers)} servers, {len(router.tool_specs)} tools")

        for server, tools in sorted(router.get_server_tools_map().items()):
            print(f"  - {server}: {sorted(tools)}")

        if config.system_prompt:
            preview = config.system_prompt[:120]
            if len(config.system_prompt) > 120:
                preview += "…"
            print(f"\n[System] {preview}")

        orchestrator = LLMOrchestrator(config, router, metrics)
        handler = CommandHandler(orchestrator, router, config)

        while True:
            try:
                user_input = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye.")
                break
            
            if not await handler.handle(user_input):
                break

    finally:
        print("\n[Info] Cleaning up MCP connections...")
        try:
            await router.cleanup()
        except Exception as e:
            logger.warning("Cleanup error: %r", e)

        print(f"[Metrics] {metrics.summary()}")

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
    args = parser.parse_args()
    try:
        asyncio.run(amain(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Info] Terminated")


if __name__ == "__main__":
    main()