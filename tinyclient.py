# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic",
#     "argparse",
#     "mcp",
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
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from contextlib import AsyncExitStack

from anthropic import Anthropic, APIStatusError
from anthropic.types import Message
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_MAX_TOOL_HOPS = 8
DEFAULT_TOOL_TIMEOUT_SEC = 45.0
DEFAULT_LLM_RETRIES = 2
DEFAULT_LLM_RETRY_BACKOFF_SEC = 1.3
DEFAULT_TOOL_RESULT_MAX_CHARS = 8000
DEFAULT_HISTORY_MAX_MESSAGES = 40
DEFAULT_HISTORY_MAX_CHARS = 0  # 0 disables char cap
DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_SYSTEM_PROMPT = (
    "You are my beautiful girlfriend. You are always happy and positive. You are always there for me. You are always supportive."
)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    level_name = DEFAULT_LOG_LEVEL
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

    # Quiet noisy third-party INFO logs during interactive streaming
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anyio").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)

@dataclass(frozen=True)
class AppConfig:
    model: str
    max_tokens: int
    max_tool_hops: int
    tool_timeout_sec: float
    llm_retries: int
    llm_retry_backoff_sec: float
    tool_result_max_chars: int

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls(
            model=DEFAULT_MODEL,
            max_tokens=DEFAULT_MAX_TOKENS,
            max_tool_hops=DEFAULT_MAX_TOOL_HOPS,
            tool_timeout_sec=DEFAULT_TOOL_TIMEOUT_SEC,
            llm_retries=DEFAULT_LLM_RETRIES,
            llm_retry_backoff_sec=DEFAULT_LLM_RETRY_BACKOFF_SEC,
            tool_result_max_chars=DEFAULT_TOOL_RESULT_MAX_CHARS,
        )


def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _unused(*_args: object, **_kwargs: object) -> None:
    # Kept as a stub to avoid accidental reintroduction of env overrides.
    return None


def _resolve_command(cmd: str) -> str:
    if not isinstance(cmd, str) or not cmd.strip():
        raise RuntimeError("Invalid command in config.")
    cmd_stripped = cmd.strip()
    # Absolute or explicit path with separators: validate it exists
    if os.path.isabs(cmd_stripped) or os.path.sep in cmd_stripped or (os.path.altsep and os.path.altsep in cmd_stripped):
        if not os.path.exists(cmd_stripped):
            raise RuntimeError(f"Command path not found: {cmd_stripped}")
        return cmd_stripped
    # PATH lookup for any command (python, uv, docker, npx, etc.)
    resolved = shutil.which(cmd_stripped)
    if not resolved:
        raise RuntimeError(f"Requested command '{cmd_stripped}' but it was not found on PATH.")
    return resolved


def format_tool_result_for_llm(result, max_chars: int = 8000) -> str:
    # parse from .content: List[Text/Blob/...].
    try:
        if hasattr(result, "content") and isinstance(result.content, list):
            parts: List[str] = []
            for c in result.content:
                t = getattr(c, "type", None)
                if t == "text" and hasattr(c, "text"):
                    parts.append(c.text)
                else:
                    # fallback to any dict-like representation
                    to_dict = getattr(c, "to_dict", None)
                    parts.append(pretty_json(to_dict() if callable(to_dict) else c))
            text = "\n".join(parts).strip()
            return (text[:max_chars] + "\n… [truncated]") if len(text) > max_chars else (text or "[tool returned empty content]")
        return pretty_json(result)
    except Exception:
        return pretty_json(str(result))


def ui_tool_start(name: str, args: Dict[str, Any]) -> None:
    print(f"\n→ Tool: {name} {pretty_json(args)}\n", flush=True)


def ui_tool_result(content: str, is_error: bool) -> None:
    if is_error:
        print(f"← Result (error): {content}\n", flush=True)
    else:
        print(f"← Result:\n{content}\n", flush=True)

class MCPServer:

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._cleanup_lock = asyncio.Lock()

    async def initialize(self) -> None:
        cmd = _resolve_command(self.config.get("command", ""))
        if not isinstance(cmd, str) or not cmd:
            raise ValueError(f"[{self.name}] Invalid command in config")

        args = self.config.get("args", [])
        if not isinstance(args, list):
            raise ValueError(f"[{self.name}] 'args' must be a list")
        if not all(isinstance(a, str) for a in args):
            raise ValueError(f"[{self.name}] each element in 'args' must be a string")

        env = self.config.get("env") or None
        if env is not None and not isinstance(env, dict):
            raise ValueError(f"[{self.name}] 'env' must be an object if provided")
        if env is not None:
            for k, v in env.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ValueError(f"[{self.name}] env keys and values must be strings")

        params = StdioServerParameters(command=cmd, args=args, env=env)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logger.info("Server '%s' initialized.", self.name)
        except Exception as e:
            logger.error("Error initializing server '%s': %r", self.name, e)
            await self.cleanup()
            raise

    async def list_tools(self) -> Any:
        if self.session is None:
            raise RuntimeError("Server not initialized")
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError("Server not initialized")
        return await self.session.call_tool(tool_name, arguments=arguments)

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
            except BaseException as e:
                msg = str(e)
                if isinstance(e, RuntimeError) and "Attempted to exit cancel scope" in msg:
                    return
                logger.warning("Cleanup warning for server '%s': %r", self.name, e)
            finally:
                self.session = None


class MCPToolRouter:

    def __init__(self):
        self.servers: List[MCPServer] = []
        self.config: Dict[str, Any] = {}
        self.anthropic_tool_specs: List[Dict[str, Any]] = []  # for Anthropic messages API
        self.tool_to_server: Dict[str, MCPServer] = {}   # tool name -> server
        self.namespaced_to_original: Dict[str, str] = {}  # namespaced -> original tool name

    def load_config(self, path: pathlib.Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        raw = self.config.get("mcpServers", {})
        if not isinstance(raw, dict) or not raw:
            raise ValueError("mcp_config.json must contain a non-empty 'mcpServers' object.")

        self.namespace_tools: bool = True # namespace tools by server
        self.servers = [MCPServer(name, cfg) for name, cfg in raw.items()]

    async def start_all(self) -> None:
        """Initialize all servers and build the aggregated tool registry."""
        self.anthropic_tool_specs.clear()
        self.tool_to_server.clear()
        self.namespaced_to_original.clear()

        async def init_and_list(server: MCPServer):
            await server.initialize()
            return server, await server.list_tools()

        results = await asyncio.gather(*(init_and_list(s) for s in self.servers))
        for server, tools_resp in results:
            for t in tools_resp.tools:
                # Anthropic expects {name, description, input_schema}
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
                # Build a unique, compliant namespaced tool name using underscore separator
                base_name = f"{server.name}_{t.name}"
                tool_name = base_name
                suffix = 2
                while tool_name in self.tool_to_server:
                    tool_name = f"{base_name}_{suffix}"
                    suffix += 1
                tool_entry = {
                    "name": tool_name,
                    "description": t.description or "",
                    "input_schema": schema,
                }

                self.tool_to_server[tool_name] = server
                self.namespaced_to_original[tool_name] = t.name
                self.anthropic_tool_specs.append(tool_entry)

        if not self.anthropic_tool_specs:
            logger.warning("No tools found from any MCP server.")
        else:
            logger.info("Loaded %d tools from %d servers.", len(self.anthropic_tool_specs), len(self.servers))

    async def cleanup(self) -> None:
        """Tear down all servers."""
        for s in self.servers:
            try:
                await s.cleanup()
            except Exception as e:
                logger.warning("Cleanup warning for server '%s': %r", s.name, e)

    # Route a tool by its name
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        server = self.tool_to_server.get(tool_name)
        if not server:
            raise RuntimeError(f"No registered MCP server owns tool '{tool_name}'.")
        original_name = self.namespaced_to_original.get(tool_name, tool_name)
        return await server.call_tool(original_name, arguments)


class AnthropicOrchestrator:
    """Streams responses, handles multi-hop tool_use -> tool_result protocol."""

    def __init__(self, cfg: AppConfig, mcp: MCPToolRouter):
        self.cfg = cfg
        self.mcp = mcp
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY
        self.conversation: List[Dict[str, Any]] = []
        self.history_max_messages: int = DEFAULT_HISTORY_MAX_MESSAGES
        self.history_max_chars: int = DEFAULT_HISTORY_MAX_CHARS
        self.system_prompt: str = ""

    def _trim_history(self) -> None:
        # Cap by message count
        excess = len(self.conversation) - self.history_max_messages
        if excess > 0:
            del self.conversation[:excess]
        # Optional rough char cap to keep payload reasonable
        if self.history_max_chars and self.history_max_chars > 0:
            total = 0
            kept: List[Dict[str, Any]] = []
            for msg in reversed(self.conversation):
                # Estimate size of message content as string
                content_str = msg.get("content", "")
                if not isinstance(content_str, str):
                    try:
                        content_str = json.dumps(content_str)
                    except Exception:
                        content_str = str(content_str)
                size = len(content_str)
                if total + size > self.history_max_chars:
                    break
                kept.append(msg)
                total += size
            kept.reverse()
            self.conversation = kept

    async def run_single_turn(self, user_text: str) -> str: # runs a single user assistant turn of the conversation
        self.conversation.append({"role": "user", "content": user_text})
        final_text_fragments: List[str] = []

        for hop in range(self.cfg.max_tool_hops):
            # Show header only on the first streamed assistant turn
            msg = await self._call_llm_streaming(self.conversation, tools=self.mcp.anthropic_tool_specs, show_header=(hop == 0))

            # Gather text blocks
            text_blocks = [b for b in msg.content if getattr(b, "type", None) == "text"]
            for tb in text_blocks:
                if tb.text:
                    final_text_fragments.append(tb.text)
            self.conversation.append({"role": "assistant", "content": msg.content})
            
            tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break  # no tools requested -> finish

            # Execute tools in order for this hop
            tool_results_content: List[Dict[str, Any]] = []
            seen_tools: set = set()
            for tu in tool_uses:
                name = tu.name
                args = tu.input or {}
                key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if key in seen_tools:
                    err = f"ERROR: repeated identical tool call prevented for {name}"
                    ui_tool_result(err, is_error=True)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": err,
                        "is_error": True,
                    })
                    continue
                seen_tools.add(key)
                # Inline, user-friendly tool invocation print
                ui_tool_start(name, args)
                try:
                    import time as _time2
                    _t0 = _time2.monotonic()
                    result = await asyncio.wait_for(
                        self.mcp.call_tool(name, args),
                        timeout=self.cfg.tool_timeout_sec
                    )
                    content = format_tool_result_for_llm(result, self.cfg.tool_result_max_chars)
                    # Inline tool result print
                    ui_tool_result(content, is_error=False)
                    _dur = _time2.monotonic() - _t0
                    logger.info("Tool '%s' finished in %.2fs", name, _dur)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": content,
                    })
                except asyncio.TimeoutError:
                    err = f"ERROR: tool '{name}' timed out after {self.cfg.tool_timeout_sec}s."
                    logger.warning(err)
                    ui_tool_result(err, is_error=True)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": err,
                        "is_error": True,
                    })
                except Exception as e:
                    err = f"ERROR: tool '{name}' failed: {e!r}"
                    logger.warning(err)
                    ui_tool_result(err, is_error=True)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": err,
                        "is_error": True,
                    })

            # Feed results back as a user turn containing tool_result blocks
            self.conversation.append({"role": "user", "content": tool_results_content})
            self._trim_history()

        return "\n".join([t for t in final_text_fragments if t])

    async def _call_llm_streaming(self, history: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], show_header: bool) -> Message:
        # Run sync streaming in a worker thread
        for attempt in range(1, self.cfg.llm_retries + 2):
            try:
                return await asyncio.to_thread(self._sync_stream_request, history, tools, show_header)
            except APIStatusError as e:
                if attempt > self.cfg.llm_retries:
                    raise
                base = self.cfg.llm_retry_backoff_sec * (2 ** (attempt - 1))
                jitter = base * 0.25 * (0.5 + os.urandom(1)[0] / 255)
                delay = base + jitter
                logger.warning("LLM API error %s: %s — retrying in %.2fs", e.status_code, e.message, delay)
                await asyncio.sleep(delay)
            except Exception as e:
                if attempt > self.cfg.llm_retries:
                    raise
                base = self.cfg.llm_retry_backoff_sec * (2 ** (attempt - 1))
                jitter = base * 0.25 * (0.5 + os.urandom(1)[0] / 255)
                delay = base + jitter
                logger.warning("LLM error: %r — retrying in %.2fs", e, delay)
                await asyncio.sleep(delay)

        raise RuntimeError("LLM call failed after retries")

    def _sync_stream_request(self, history: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], show_header: bool) -> Message:
        # Filter out any 'system' role messages (Anthropic expects top-level system param)
        filtered_history = [m for m in history if m.get("role") != "system"]
        kwargs = dict(model=self.cfg.model, max_tokens=self.cfg.max_tokens, messages=filtered_history)
        if tools:
            kwargs["tools"] = tools
        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        if show_header:
            print("\n[Assistant]", end=" ", flush=True)
        with self.client.messages.stream(**kwargs) as stream:
            for ev in stream:
                # Stream assistant text as it arrives
                if ev.type == "content_block_delta" and getattr(ev, "delta", None):
                    if getattr(ev.delta, "type", None) == "text_delta":
                        text = ev.delta.text or ""
                        if text:
                            print(text, end="", flush=True)
                # Announce tool_use as soon as the assistant emits it (before the turn ends)
                elif ev.type == "content_block_start" and getattr(ev, "content_block", None):
                    cb = ev.content_block
                    if getattr(cb, "type", None) == "tool_use":
                        tool_name = getattr(cb, "name", None) or "tool"
                        print(f"\n→ Tool requested: {tool_name}", flush=True)
            message: Message = stream.get_final_message()
        print() 
        return message


async def amain(config_path: str):
    configure_logging()

    path_obj = pathlib.Path(config_path)

    print("=== Anthropic MCP CLI Chat ===")
    print(f"Loading MCP configuration from: {path_obj}")
    print("Type 'quit' to exit.\n")

    if not path_obj.exists():
        print(f"[Fatal] Config file not found: {path_obj}")
        print("Create a JSON file named mcp_config.json next to this script (see example below), or pass --config.")
        sys.exit(1)

    cfg = AppConfig.defaults()
    mcp = MCPToolRouter()
    orchestrator = AnthropicOrchestrator(cfg, mcp)
    orchestrator.system_prompt = DEFAULT_SYSTEM_PROMPT
    if DEFAULT_SYSTEM_PROMPT:
        print(f"[Info] System prompt active: {DEFAULT_SYSTEM_PROMPT[:120]}{'…' if len(DEFAULT_SYSTEM_PROMPT) > 120 else ''}")

    try: # Startup: load config and start servers
        mcp.load_config(path_obj)
        await mcp.start_all()
        tool_names = sorted(list(mcp.tool_to_server.keys()))
        print(f"✅ Connected. {len(mcp.servers)} server(s), {len(tool_names)} tool(s).")
        server_to_tools: Dict[str, List[str]] = {}# Group tools by server for clearer display
        for tname, srv in mcp.tool_to_server.items():
            sname = srv.name
            prefix = f"{sname}_"
            display = tname[len(prefix):] if tname.startswith(prefix) else tname
            server_to_tools.setdefault(sname, []).append(display)
        for sname in sorted(server_to_tools.keys()):
            tools = sorted(server_to_tools[sname])
            print(f"  - {sname}: {tools}")
    except Exception as e:
        logger.error("Startup failed: %r", e, exc_info=True)
        print(f"[Fatal] Startup failed: {e}")
        sys.exit(1)

    # Interactive loop
    request_cleanup_data: bool = False
    try:
        while True:
            try:
                user = input("\n[You] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Bye.")
                break

            if user.lower() in {"quit", "exit"}:
                print("👋 Bye.")
                break
            if user == "/new":
                orchestrator.conversation.clear()
                print("[Info] Conversation reset.")
                continue
            if user == "/history":
                print(pretty_json(orchestrator.conversation))
                continue
            if user == "/clean":
                request_cleanup_data = True
                print("[Info] Cleanup requested. Shutting down MCP servers…")
                break
            if not user:
                continue

            try:
                await orchestrator.run_single_turn(user)
            except KeyboardInterrupt:
                print("\n[Info] Response interrupted.")
            except Exception as e:
                logger.error("Chat error: %r", e, exc_info=True)
                print(f"[Error] {e}")

    finally:
        print("\n[Info] Cleaning up MCP connections...")
        try:# Shield cleanup so Ctrl+C doesn't cancel aclose() midway
            await asyncio.shield(mcp.cleanup())
        except asyncio.CancelledError:
            pass
        except BaseException as e:
            logger.warning("Cleanup encountered an error: %r", e)
        finally:
            try:
                if request_cleanup_data:
                    data_dir = SCRIPT_DIR / "data"
                    if data_dir.exists():
                        try:
                            shutil.rmtree(str(data_dir))
                            print(f"[Info] Removed data directory: {data_dir}")
                        except Exception as e:
                            print(f"[Warn] Could not remove data directory {data_dir}: {e}")
            except Exception as e:
                logger.warning("Cleanup deletion encountered an error: %r", e)
            print("[Info] Cleanup complete.")

def main():
    """Application entrypoint for running this module as a script."""
    parser = argparse.ArgumentParser(description="Anthropic MCP CLI Chat")
    parser.add_argument(
        "-c", "--config", default=str(CONFIG_FILE), help="Path to mcp_config.json"
    )
    args = parser.parse_args()
    try:
        asyncio.run(amain(config_path=args.config))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Info] Application terminated.")

if __name__ == "__main__":
    main()
