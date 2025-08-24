#!/usr/bin/env python3
import os
import sys
import json
import time
import asyncio
import logging
import shutil
import pathlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from anthropic import Anthropic, APIStatusError
from anthropic.types import Message
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------
# Env & logging
# ---------------------------
# Load .env first so it can influence logging configuration
load_dotenv()

# Treat empty LOG_LEVEL as unset and default to INFO
_LOG_LEVEL_NAME = (os.getenv("LOG_LEVEL") or "").strip().upper() or "INFO"
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Quiet noisy third-party INFO logs during interactive streaming
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anyio").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"


# ---------------------------
# App config
# ---------------------------
@dataclass(frozen=True)
class AppConfig:
    model: str = None  # type: ignore[assignment]
    max_tokens: int = 0  # type: ignore[assignment]
    max_tool_hops: int = 0  # type: ignore[assignment]
    tool_timeout_sec: float = 0.0  # type: ignore[assignment]
    llm_retries: int = 0  # type: ignore[assignment]
    llm_retry_backoff_sec: float = 0.0  # type: ignore[assignment]

    # Post-init defaults using env with empty-string fallback to defaults
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        object.__setattr__(self, "model", getenv_str("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
        object.__setattr__(self, "max_tokens", getenv_int("MAX_TOKENS", 1500))
        object.__setattr__(self, "max_tool_hops", getenv_int("MAX_TOOL_HOPS", 8))
        object.__setattr__(self, "tool_timeout_sec", getenv_float("TOOL_TIMEOUT_SEC", 45.0))
        object.__setattr__(self, "llm_retries", getenv_int("LLM_RETRIES", 2))
        object.__setattr__(self, "llm_retry_backoff_sec", getenv_float("LLM_RETRY_BACKOFF_SEC", 1.3))


# ---------------------------
# Utilities
# ---------------------------
def pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def getenv_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except Exception:
        return default


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except Exception:
        return default

def _resolve_command(cmd: str) -> str:
    """
    Resolve command with support for 'npx' convenience, otherwise leave as-is.
    If cmd == 'npx', try shutil.which('npx'), else use the literal string (e.g., 'python', 'node', '/usr/bin/foo').
    """
    if cmd == "npx":
        resolved = shutil.which("npx")
        if not resolved:
            raise RuntimeError("Requested command 'npx' but it was not found on PATH.")
        return resolved
    return cmd


def _normalize_tool_result(result) -> str:
    """
    MCP tool call result typically has .content: List[Text/Blob/...].
    Convert to a reasonably concise string for the LLM.
    """
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
                    parts.append(pretty(to_dict() if callable(to_dict) else c))
            text = "\n".join(parts).strip()
            return (text[:8000] + "\n… [truncated]") if len(text) > 8000 else (text or "[tool returned empty content]")
        return pretty(result)
    except Exception:
        return pretty(str(result))


# ---------------------------
# MCP Server wrapper
# ---------------------------
class MCPServer:
    """
    One MCP server instance.
    Config fields expected:
      {
        "command": "python" | "node" | "npx" | "/path/to/cmd",
        "args": ["path/to/server.py", "..."],
        "env": { "KEY": "VALUE", ... }   # optional
      }
    """

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

        env = self.config.get("env") or None
        if env is not None and not isinstance(env, dict):
            raise ValueError(f"[{self.name}] 'env' must be an object if provided")

        params = StdioServerParameters(command=cmd, args=args, env=env)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info("Server '%s' initialized.", self.name)
        except Exception as e:
            logging.error("Error initializing server '%s': %r", self.name, e)
            await self.cleanup()
            raise

    async def list_tools(self):
        assert self.session is not None, "Server not initialized"
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        assert self.session is not None, "Server not initialized"
        return await self.session.call_tool(tool_name, arguments=arguments)

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
            except BaseException as e:
                msg = str(e)
                # Suppress benign anyio cancel-scope task-mismatch warnings on shutdown
                if isinstance(e, RuntimeError) and "Attempted to exit cancel scope" in msg:
                    return
                logging.warning("Cleanup warning for server '%s': %r", self.name, e)
            finally:
                self.session = None


# ---------------------------
# MCP Client: multi-server manager + tool registry
# ---------------------------
class MCPMultiClient:
    """
    Loads servers from mcp_config.json, starts them, aggregates tool schemas for Anthropic,
    and routes tool calls to the owning server by tool name.
    """

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.servers: List[MCPServer] = []
        self.config: Dict[str, Any] = {}
        self.anthropic_tools: List[Dict[str, Any]] = []  # for Anthropic messages API
        self.tool_to_server: Dict[str, MCPServer] = {}   # tool name -> server

    def load_config(self, path: pathlib.Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        raw = self.config.get("mcpServers", {})
        if not isinstance(raw, dict) or not raw:
            raise ValueError("mcp_config.json must contain a non-empty 'mcpServers' object.")

        self.servers = [MCPServer(name, cfg) for name, cfg in raw.items()]

    async def start(self) -> None:
        """Initialize all servers and build the aggregated tool registry."""
        self.anthropic_tools.clear()
        self.tool_to_server.clear()

        for server in self.servers:
            await server.initialize()
            tools_resp = await server.list_tools()
            for t in tools_resp.tools:
                # Anthropic expects {name, description, input_schema}
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
                tool_entry = {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": schema,
                }

                # warn on collisions; last writer wins (simple policy)
                if t.name in self.tool_to_server:
                    prev = self.tool_to_server[t.name].name
                    logging.warning("Tool name collision: '%s' from server '%s' overrides server '%s'",
                                    t.name, server.name, prev)

                self.tool_to_server[t.name] = server
                self.anthropic_tools.append(tool_entry)

        if not self.anthropic_tools:
            logging.warning("No tools found from any MCP server.")
        else:
            logging.info("Loaded %d tools from %d servers.", len(self.anthropic_tools), len(self.servers))

    async def cleanup(self) -> None:
        """Tear down all servers."""
        for s in self.servers:
            try:
                await s.cleanup()
            except Exception as e:
                logging.warning("Cleanup warning for server '%s': %r", s.name, e)

    # Route a tool by its name
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        server = self.tool_to_server.get(tool_name)
        if not server:
            raise RuntimeError(f"No registered MCP server owns tool '{tool_name}'.")
        return await server.call_tool(tool_name, arguments)


# ---------------------------
# Anthropic Orchestrator
# ---------------------------
class AnthropicOrchestrator:
    """Streams responses, handles multi-hop tool_use -> tool_result protocol."""

    def __init__(self, cfg: AppConfig, mcp: MCPMultiClient):
        self.cfg = cfg
        self.mcp = mcp
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY

    async def chat_once(self, user_text: str) -> str:
        """
        Single question/answer turn:
          - ask Claude with tools catalog
          - execute any tool_use blocks (multiple hops)
          - return concatenated assistant text
        """
        history: List[Dict[str, Any]] = [{"role": "user", "content": user_text}]
        final_text_fragments: List[str] = []

        for hop in range(self.cfg.max_tool_hops):
            # Show header only on the first streamed assistant turn
            msg = await self._call_llm_streaming(history, tools=self.mcp.anthropic_tools, show_header=(hop == 0))

            # Gather text blocks
            text_blocks = [b for b in msg.content if getattr(b, "type", None) == "text"]
            for tb in text_blocks:
                if tb.text:
                    final_text_fragments.append(tb.text)

            # Always append the assistant message that may contain tool_use blocks
            # so that subsequent tool_result blocks correctly reference the previous message
            history.append({"role": "assistant", "content": msg.content})

            # Gather tool_use blocks
            tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break  # no tools requested -> finish

            # Execute tools in order for this hop
            tool_results_content: List[Dict[str, Any]] = []
            for tu in tool_uses:
                name = tu.name
                args = tu.input or {}
                # Inline, user-friendly tool invocation print
                print(f"\n→ Tool: {name} {pretty(args)}\n", flush=True)
                try:
                    result = await asyncio.wait_for(
                        self.mcp.call_tool(name, args),
                        timeout=self.cfg.tool_timeout_sec
                    )
                    content = _normalize_tool_result(result)
                    # Inline tool result print
                    print(f"← Result:\n{content}\n", flush=True)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": content,
                    })
                except asyncio.TimeoutError:
                    err = f"ERROR: tool '{name}' timed out after {self.cfg.tool_timeout_sec}s."
                    logging.warning(err)
                    print(f"← Result (error): {err}\n", flush=True)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": err,
                        "is_error": True,
                    })
                except Exception as e:
                    err = f"ERROR: tool '{name}' failed: {e!r}"
                    logging.warning(err)
                    print(f"← Result (error): {err}\n", flush=True)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": err,
                        "is_error": True,
                    })

            # Feed results back as a user turn containing tool_result blocks
            history.append({"role": "user", "content": tool_results_content})

        return "\n".join([t for t in final_text_fragments if t])

    async def _call_llm_streaming(self, history: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], show_header: bool) -> Message:
        # Run sync streaming in a worker thread
        for attempt in range(1, self.cfg.llm_retries + 2):
            try:
                return await asyncio.to_thread(self._sync_stream_request, history, tools, show_header)
            except APIStatusError as e:
                if attempt > self.cfg.llm_retries:
                    raise
                backoff = self.cfg.llm_retry_backoff_sec * attempt
                logging.warning("LLM API error %s: %s — retrying in %.1fs", e.status_code, e.message, backoff)
                await asyncio.sleep(backoff)
            except Exception as e:
                if attempt > self.cfg.llm_retries:
                    raise
                backoff = self.cfg.llm_retry_backoff_sec * attempt
                logging.warning("LLM error: %r — retrying in %.1fs", e, backoff)
                await asyncio.sleep(backoff)

        raise RuntimeError("LLM call failed after retries")

    def _sync_stream_request(self, history: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], show_header: bool) -> Message:
        kwargs = dict(model=self.cfg.model, max_tokens=self.cfg.max_tokens, messages=history)
        if tools:
            kwargs["tools"] = tools

        if show_header:
            print("\n[Assistant]", end=" ", flush=True)
        with self.client.messages.stream(**kwargs) as stream:
            for ev in stream:
                if ev.type == "content_block_delta" and getattr(ev, "delta", None):
                    if getattr(ev.delta, "type", None) == "text_delta":
                        text = ev.delta.text or ""
                        if text:
                            print(text, end="", flush=True)
            message: Message = stream.get_final_message()
        print()  # newline after streaming turn
        return message


# ---------------------------
# CLI & lifecycle
# ---------------------------
async def amain():
    print("=== Anthropic MCP CLI Chat ===")
    print(f"Loading MCP configuration from: {CONFIG_FILE}")
    print("Type 'quit' to exit.\n")

    if not CONFIG_FILE.exists():
        print(f"[Fatal] Config file not found: {CONFIG_FILE}")
        print("Create a JSON file named mcp_config.json next to this script (see example below).")
        sys.exit(1)

    cfg = AppConfig()
    mcp = MCPMultiClient()
    orchestrator = AnthropicOrchestrator(cfg, mcp)

    try:
        # Startup: load config and start servers
        mcp.load_config(CONFIG_FILE)
        await mcp.start()
        tool_names = list(mcp.tool_to_server.keys())
        print(f"✅ Connected. {len(mcp.servers)} server(s), {len(tool_names)} tool(s): {tool_names or '—'}")
    except Exception as e:
        logging.error("Startup failed: %r", e, exc_info=True)
        print(f"[Fatal] Startup failed: {e}")
        sys.exit(1)

    # Interactive loop
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
            if not user:
                continue

            try:
                answer = await orchestrator.chat_once(user)
                # We already streamed the assistant turns inline; no second print
            except KeyboardInterrupt:
                print("\n[Info] Response interrupted.")
            except Exception as e:
                logging.error("Chat error: %r", e, exc_info=True)
                print(f"[Error] {e}")

    finally:
        # Cleanup: always attempt to close servers, but protect against cancellation
        print("\n[Info] Cleaning up MCP connections...")
        try:
            # Shield cleanup so Ctrl+C doesn't cancel aclose() midway
            await asyncio.shield(mcp.cleanup())
        except asyncio.CancelledError:
            # Swallow cancellation during cleanup
            pass
        except BaseException as e:
            logging.warning("Cleanup encountered an error: %r", e)
        finally:
            print("[Info] Cleanup complete.")


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Graceful shutdown without traceback on Ctrl+C
        print("\n[Info] Application terminated.")
