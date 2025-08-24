# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "litellm",
#     "argparse",
#     "google",
#     "google-genai",
#     "anthropic",
#     "openai",
#     "groq",
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
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import AsyncExitStack

import litellm
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"

DEFAULT_MODEL_OPENAI = "gpt-4.1-mini"
DEFAULT_MODEL_ANTHROPIC = "claude-3-7-sonnet-20250219"
DEFAULT_MODEL_GEMINI = "gemini-2.5-pro"
DEFAULT_MODEL_GROQ = "llama-3.3-70b-versatile" # or "gpt-oss-120b"
DEFAULT_MODEL = DEFAULT_MODEL_GROQ

DEFAULT_MAX_TOKENS = 32768
DEFAULT_MAX_TOOL_HOPS = 50
TOOL_TIMEOUT_SEC = 30.0
DEFAULT_TOOL_RESULT_MAX_CHARS = 8000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_PARALLEL_TOOLS = 4
DEFAULT_TOOL_PREVIEW_LINES = 0

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

logger = logging.getLogger(__name__)

TOTAL_PROMPT_TOKENS: int = 0
TOTAL_COMPLETION_TOKENS: int = 0
TOTAL_COST: float = 0.0

LOG_JSON: bool = False


def log_event(event: str, **fields: Any) -> None:
    """Emit optional structured JSON logs for ops."""
    if not LOG_JSON:
        return
    # Avoid dumping huge blobs
    trimmed = {}
    for k, v in fields.items():
        try:
            s = json.dumps(v, ensure_ascii=False)
        except Exception:
            s = str(v)
        if len(s) > 2000:
            trimmed[k] = s[:2000] + "…"
        else:
            trimmed[k] = v
    payload = {"event": event, "ts": time.time(), **trimmed}
    try:
        # Send to stderr to avoid interleaving with streamed assistant text
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
    except Exception:
        pass


def _infer_provider_env_vars(model_name: str) -> Tuple[str, List[str]]:
    m = (model_name or "").lower()
    if "claude" in m:
        return ("Anthropic", ["ANTHROPIC_API_KEY"])
    if "gemini" in m or "google" in m:
        return ("Gemini", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    if m.startswith("gpt-") or "openai" in m or m.startswith("o1") or m.startswith("o3"):
        return ("OpenAI", ["OPENAI_API_KEY"])
    if "groq" in m or "llama" in m:
        return ("Groq", ["GROQ_API_KEY"])
    return ("", [])


def configure_logging(level_name: Optional[str] = None) -> None:
    level_name = level_name or DEFAULT_LOG_LEVEL
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anyio").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    try:
        litellm.set_verbose = False
    except Exception:
        pass
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    try:
        _register_litellm_callbacks()
    except Exception:
        pass


def _register_litellm_callbacks() -> None:
    try:
        existing = getattr(litellm, "success_callback", [])
        if not isinstance(existing, list):
            existing = [existing]
        if _litellm_success_callback not in existing:
            existing.append(_litellm_success_callback)  # type: ignore[arg-type]
        litellm.success_callback = existing  # type: ignore[attr-defined]
    except Exception:
        pass


def _litellm_success_callback(kwargs: Dict[str, Any], completion_response: Any, start_time: Any, end_time: Any) -> None:
    try:
        model = str(kwargs.get("model", ""))
        latency_s: Optional[float] = None
        try:
            latency_s = float(end_time) - float(start_time)  # type: ignore[arg-type]
        except Exception:
            latency_s = None

        def _get(obj: Any, *keys: str) -> Any:
            cur = obj
            for k in keys:
                if cur is None:
                    return None
                try:
                    if isinstance(cur, dict):
                        cur = cur.get(k)
                    else:
                        cur = getattr(cur, k)
                except Exception:
                    return None
            return cur

        usage = _get(completion_response, "usage")
        prompt_tokens = _get(usage, "prompt_tokens") or _get(usage, "input_tokens") or 0
        completion_tokens = _get(usage, "completion_tokens") or _get(usage, "output_tokens") or 0
        total_tokens = _get(usage, "total_tokens") or (prompt_tokens or 0) + (completion_tokens or 0)
        cost = kwargs.get("response_cost")
        if cost is None:
            cost = _get(completion_response, "cost") or _get(completion_response, "response_cost")
        logger.info(
            "[metrics] model=%s latency=%.2fs tokens[in=%s, out=%s, total=%s] cost=%s",
            model,
            (latency_s if latency_s is not None else -1.0),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
        )
        try:
            global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_COST
            TOTAL_PROMPT_TOKENS += int(prompt_tokens or 0)
            TOTAL_COMPLETION_TOKENS += int(completion_tokens or 0)
            if cost is not None:
                TOTAL_COST += float(cost)
        except Exception:
            pass
    except Exception:
        pass


@dataclass(frozen=True)
class AppConfig:
    model: str
    max_tokens: int
    max_tool_hops: int
    tool_result_max_chars: int

    @classmethod
    def defaults(cls) -> "AppConfig":
        return cls(
            model=DEFAULT_MODEL,
            max_tokens=DEFAULT_MAX_TOKENS,
            max_tool_hops=DEFAULT_MAX_TOOL_HOPS,
            tool_result_max_chars=DEFAULT_TOOL_RESULT_MAX_CHARS,
        )


def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(obj)


def ui_tool_start(tool_name: str, arguments: dict | None = None) -> None:
    print(f"→ {tool_name} {compact_json(arguments or {})}", flush=True)


def ui_tool_result(tool_name: str, text: str, is_error: bool = False, duration_s: float | None = None) -> None:
    status = "error" if is_error else "ok"
    suffix = "" if duration_s is None else f" ({duration_s:.2f}s)"
    header = f"← {tool_name}: {status}{suffix}"
    if text:
        print(f"{header}\n{text}", flush=True)
    else:
        print(header, flush=True)


def _resolve_command(cmd: str) -> str:
    if not isinstance(cmd, str) or not cmd.strip():
        raise RuntimeError("Invalid command in config.")
    cmd_stripped = cmd.strip()
    if os.path.isabs(cmd_stripped) or os.path.sep in cmd_stripped or (os.path.altsep and os.path.altsep in cmd_stripped):
        if not os.path.exists(cmd_stripped):
            raise RuntimeError(f"Command path not found: {cmd_stripped}")
        return cmd_stripped
    resolved = shutil.which(cmd_stripped)
    if not resolved:
        raise RuntimeError(f"Requested command '{cmd_stripped}' but it was not found on PATH.")
    return resolved


def format_tool_result_for_llm(result, max_chars: int = 8000) -> Tuple[str, bool, int]:
    try:
        if hasattr(result, "content") and isinstance(result.content, list):
            parts: List[str] = []
            for c in result.content:
                t = getattr(c, "type", None)
                if t == "text" and hasattr(c, "text"):
                    parts.append(c.text)
                else:
                    to_dict = getattr(c, "to_dict", None)
                    parts.append(pretty_json(to_dict() if callable(to_dict) else c))
            text = "\n".join(parts).strip()
            total = len(text)
            if total > max_chars:
                return (text[:max_chars], True, total)
            return (text or "[tool returned empty content]", False, total)
        text = pretty_json(result)
        total = len(text)
        if total > max_chars:
            return (text[:max_chars], True, total)
        return (text, False, total)
    except Exception:
        s = pretty_json(str(result))
        total = len(s)
        if total > max_chars:
            return (s[:max_chars], True, total)
        return (s, False, total)


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
        self.openai_tool_specs: List[Dict[str, Any]] = []
        self.tool_to_server: Dict[str, MCPServer] = {}
        self.namespaced_to_original: Dict[str, str] = {}

    def load_config(self, path: pathlib.Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        raw = self.config.get("mcpServers", {})
        if not isinstance(raw, dict) or not raw:
            raise ValueError("mcp_config.json must contain a non-empty 'mcpServers' object.")
        self.servers = [MCPServer(name, cfg) for name, cfg in raw.items()]

    async def start_all(self) -> None:
        self.openai_tool_specs.clear()
        self.tool_to_server.clear()
        self.namespaced_to_original.clear()

        async def init_and_list(server: MCPServer):
            await server.initialize()
            return server, await server.list_tools()

        results = await asyncio.gather(*(init_and_list(s) for s in self.servers))
        for server, tools_resp in results:
            for t in tools_resp.tools:
                schema_raw = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
                schema = self._validate_input_schema(schema_raw, server.name, t.name)
                base_name = f"{server.name}_{t.name}"
                tool_name = base_name
                suffix = 2
                while tool_name in self.tool_to_server:
                    tool_name = f"{base_name}_{suffix}"
                    suffix += 1
                tool_entry = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": t.description or "",
                        "parameters": schema,
                    },
                }
                self.tool_to_server[tool_name] = server
                self.namespaced_to_original[tool_name] = t.name
                self.openai_tool_specs.append(tool_entry)
        if not self.openai_tool_specs:
            logger.warning("No tools found from any MCP server.")
        else:
            logger.info("Loaded %d tools from %d servers.", len(self.openai_tool_specs), len(self.servers))

    def _validate_input_schema(self, schema: Any, server_name: str, tool_name: str) -> Dict[str, Any]:
        if schema is None:
            return {"type": "object", "properties": {}}
        if not isinstance(schema, dict):
            raise ValueError(f"[{server_name}] Tool '{tool_name}' input_schema must be an object")

        typ = schema.get("type")
        if typ is None:
            schema = {**schema, "type": "object"}
        elif typ != "object":
            raise ValueError(f"[{server_name}] Tool '{tool_name}' input_schema.type must be 'object'")

        props = schema.get("properties")
        if props is None:
            schema["properties"] = {}
        elif not isinstance(props, dict):
            raise ValueError(f"[{server_name}] Tool '{tool_name}' input_schema.properties must be an object")

        if "required" in schema:
            req = schema["required"]
            if not isinstance(req, list) or not all(isinstance(x, str) for x in req):
                raise ValueError(f"[{server_name}] Tool '{tool_name}' input_schema.required must be a list of strings")

        allowed = {
            "type", "properties", "required", "additionalProperties",
            "description", "title", "$schema", "definitions"
        }
        unknown = [k for k in schema.keys() if k not in allowed]
        if unknown:
            logger.warning("[%s] Tool '%s' input_schema has unknown keys: %s", server_name, tool_name, unknown)
        return schema

    async def cleanup(self) -> None:
        for s in self.servers:
            try:
                await s.cleanup()
            except Exception as e:
                logger.warning("Cleanup warning for server '%s': %r", s.name, e)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        server = self.tool_to_server.get(tool_name)
        if not server:
            raise RuntimeError(f"No registered MCP server owns tool '{tool_name}'.")
        original_name = self.namespaced_to_original.get(tool_name, tool_name)
        return await server.call_tool(original_name, arguments)


class LLMOrchestrator:
    def __init__(self, cfg: AppConfig, mcp: MCPToolRouter, *, tool_preview_lines: int = 0):
        self.cfg = cfg
        self.mcp = mcp
        self.conversation: List[Dict[str, Any]] = []
        self.system_prompt: str = ""
        self.max_parallel_tools: Optional[int] = DEFAULT_MAX_PARALLEL_TOOLS
        self.tool_preview_lines = tool_preview_lines
        self._print_lock = asyncio.Lock()

    async def run_single_turn(self, user_text: str) -> str:
        self.conversation.append({"role": "user", "content": user_text})
        final_text_fragments: List[str] = []

        for hop in range(self.cfg.max_tool_hops):
            assistant_msg = await self._call_llm_streaming(
                self.conversation,
                tools=self.mcp.openai_tool_specs,
                show_header=(hop == 0),
            )
            if assistant_msg.get("content"):
                final_text_fragments.append(assistant_msg["content"])
            self.conversation.append({
                "role": "assistant",
                "content": assistant_msg.get("content", ""),
                **({"tool_calls": assistant_msg.get("tool_calls")} if assistant_msg.get("tool_calls") else {}),
            })

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                break

            tool_results_msgs: List[Optional[Dict[str, Any]]] = [None] * len(tool_calls)
            result_meta: List[Optional[Dict[str, Any]]] = [None] * len(tool_calls)
            result_blocks: List[Optional[str]] = [None] * len(tool_calls)
            seen_tools: set = set()

            async def _run_one(idx: int, name: str, args: Dict[str, Any], tool_call_id: str, sem: Optional[asyncio.Semaphore]):
                label = str(idx + 1)

                async def _body():
                    async with self._print_lock:
                        print(f"→ [{label}] {name} {compact_json(args)}", flush=True)
                    log_event("tool_start", name=name, index=idx, args=args)
                    try:
                        import time as _time2
                        _t0 = _time2.monotonic()
                        result = await asyncio.wait_for(self.mcp.call_tool(name, args), timeout=TOOL_TIMEOUT_SEC)
                        content_text, was_trunc, total_len = format_tool_result_for_llm(
                            result, self.cfg.tool_result_max_chars
                        )
                        _dur = _time2.monotonic() - _t0
                        try:
                            num_lines = len(content_text.splitlines()) if content_text else 0
                        except Exception:
                            num_lines = 0
                        block = f"← [{label}] {name}: ok ({_dur:.2f}s) [{num_lines} lines]"
                        result_blocks[idx] = block
                        logger.info("Tool '%s' finished in %.2fs", name, _dur)
                        log_event("tool_finish", name=name, index=idx, ok=True, duration=_dur, lines=num_lines)
                        if self.tool_preview_lines and content_text:
                            preview = "\n".join(content_text.splitlines()[: self.tool_preview_lines])
                            async with self._print_lock:
                                print(preview, flush=True)
                        result_meta[idx] = {"name": name, "ok": True, "duration": _dur}
                        meta = f"[TOOL META] name={name} ok=true duration={_dur:.2f}s lines={num_lines} truncated={str(was_trunc).lower()} chars={min(self.cfg.tool_result_max_chars, total_len) if was_trunc else total_len}/{total_len}\n"
                        tool_results_msgs[idx] = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "content": meta + (content_text or ""),
                        }
                    except asyncio.TimeoutError:
                        err = f"ERROR: tool '{name}' timed out after {TOOL_TIMEOUT_SEC}s."
                        logger.warning(err)
                        log_event("tool_finish", name=name, index=idx, ok=False, error="timeout")
                        block = f"← [{label}] {name}: error\n{err}"
                        result_blocks[idx] = block
                        result_meta[idx] = {"name": name, "ok": False, "error": "timeout"}
                        tool_results_msgs[idx] = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "content": err,
                        }
                    except Exception as e:
                        err = f"ERROR: tool '{name}' failed: {e!r}"
                        logger.warning(err)
                        log_event("tool_finish", name=name, index=idx, ok=False, error="exception", detail=str(e))
                        block = f"← [{label}] {name}: error\n{err}"
                        result_blocks[idx] = block
                        result_meta[idx] = {"name": name, "ok": False, "error": "exception"}
                        tool_results_msgs[idx] = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "content": err,
                        }

                if sem is None:
                    await _body()
                else:
                    async with sem:
                        await _body()

            tasks: List[asyncio.Task] = []
            # 0 = serial, >0 = bounded concurrency, None = unlimited
            sem: Optional[asyncio.Semaphore] = None
            if self.max_parallel_tools is None:
                sem = None
            elif self.max_parallel_tools == 0:
                sem = asyncio.Semaphore(1)
            elif self.max_parallel_tools > 0:
                sem = asyncio.Semaphore(self.max_parallel_tools)
            else:
                sem = None

            for idx, tc in enumerate(tool_calls):
                name = tc.get("function", {}).get("name") or tc.get("name")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {"_raw": raw_args}
                key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if key in seen_tools:
                    err = f"ERROR: repeated identical tool call prevented for {name}"
                    label = str(idx + 1)
                    block = f"← [{label}] {name}: error\n{err}"
                    result_blocks[idx] = block
                    result_meta[idx] = {"name": name, "ok": False, "error": "duplicate"}
                    tool_results_msgs[idx] = {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"tc_{idx}"),
                        "name": name,
                        "content": err,
                    }
                    continue
                seen_tools.add(key)
                tasks.append(asyncio.create_task(_run_one(idx, name, args, tc.get("id", f"tc_{idx}"), sem)))

            if tasks:
                try:
                    await asyncio.gather(*tasks)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

            async with self._print_lock:
                for blk in result_blocks:
                    if blk:
                        print(blk, flush=True)
            for msg in tool_results_msgs:
                if msg is not None:
                    self.conversation.append(msg)

        return "\n".join([t for t in final_text_fragments if t])

    async def _call_llm_streaming(self, history: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], show_header: bool) -> Dict[str, Any]:
        retry_delays = [1.0, 3.0, 5.0]
        attempt = 0
        while True:
            try:
                messages: List[Dict[str, Any]] = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.extend(history)
                kwargs = dict(model=self.cfg.model, max_tokens=self.cfg.max_tokens, messages=messages, stream=True)
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if show_header:
                    print("\n[Assistant]", end=" ", flush=True)

                log_event("llm_request", model=self.cfg.model, max_tokens=self.cfg.max_tokens, tools=len(tools or []))
                chunks: List[Any] = []
                stream = await litellm.acompletion(**kwargs)
                saw_toolcall_notice: bool = False
                suppress_content: bool = False

                try:
                    async for chunk in stream:
                        chunks.append(chunk)
                        try:
                            choice = chunk.choices[0]
                            delta = choice.delta
                            # Try to read text delta
                            text = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else None)

                            # Detect tool deltas
                            has_tool_delta = False
                            if isinstance(delta, dict):
                                has_tool_delta = bool(delta.get("tool_calls") or delta.get("function_call"))
                            else:
                                try:
                                    has_tool_delta = bool(getattr(delta, "tool_calls", None) or getattr(delta, "function_call", None))
                                except Exception:
                                    has_tool_delta = False

                            if has_tool_delta and not saw_toolcall_notice:
                                saw_toolcall_notice = True
                                suppress_content = True
                                print(" … [requesting tools]", end="", flush=True)

                            if text and not suppress_content:
                                print(text, end="", flush=True)

                            fr = getattr(choice, "finish_reason", None)
                            if fr is None and isinstance(choice, dict):
                                fr = choice.get("finish_reason")
                            if fr == "tool_calls":
                                break
                        except Exception:
                            pass
                finally:
                    try:
                        await stream.aclose()
                    except Exception:
                        pass

                final_resp = self._manually_build_from_chunks(chunks)
                msg = final_resp['choices'][0]['message']
                out: Dict[str, Any] = {"content": msg.get("content", "")}
                tc = msg.get("tool_calls")
                if tc:
                    # Convert pydantic objects (if any) to dicts; here they are already dicts we built.
                    out["tool_calls"] = tc
                print()
                log_event("llm_response", has_tool_calls=bool(tc), content_len=len(out["content"] or ""))
                return out

            except Exception as e:
                if attempt >= len(retry_delays):
                    raise
                delay = retry_delays[attempt]
                attempt += 1
                logger.warning("LLM error: %r — retrying in %.2fs", e, delay)
                log_event("llm_retry", attempt=attempt, delay=delay, error=str(e))
                await asyncio.sleep(delay)

    def _manually_build_from_chunks(self, chunks: List[Any]) -> Dict[str, Any]:
        final_content: List[str] = []
        calls_by_index: Dict[int, Dict[str, Any]] = {}
        argbuf_by_index: Dict[int, List[str]] = {}

        for chunk in chunks:
            try:
                delta = chunk.choices[0].delta
                if not delta:
                    continue

                # Content stream
                content = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else None)
                if content:
                    final_content.append(content)

                # Tool-calls stream
                raw_tool_calls = getattr(delta, "tool_calls", None) or (delta.get("tool_calls") if isinstance(delta, dict) else None)
                if raw_tool_calls:
                    for tc_chunk in raw_tool_calls:
                        idx = getattr(tc_chunk, 'index', None)
                        if idx is None:
                            # Some providers omit 'index'—assume 0
                            idx = 0
                        if idx not in calls_by_index:
                            calls_by_index[idx] = {"type": None, "id": None, "function": {"name": None, "arguments": ""}}
                            argbuf_by_index[idx] = []

                        # id/type
                        if hasattr(tc_chunk, 'id') and tc_chunk.id:
                            calls_by_index[idx]['id'] = tc_chunk.id
                        if hasattr(tc_chunk, 'type') and tc_chunk.type:
                            calls_by_index[idx]['type'] = tc_chunk.type

                        # function block
                        fn = getattr(tc_chunk, 'function', None)
                        if fn:
                            # name
                            if hasattr(fn, 'name') and fn.name:
                                calls_by_index[idx]['function']['name'] = fn.name
                            # arguments (append chunks)
                            if hasattr(fn, 'arguments') and fn.arguments:
                                argbuf_by_index[idx].append(fn.arguments)
                        elif isinstance(tc_chunk, dict):
                            fnd = tc_chunk.get("function")
                            if fnd:
                                name = fnd.get("name")
                                if name:
                                    calls_by_index[idx]['function']['name'] = name
                                arguments = fnd.get("arguments")
                                if arguments:
                                    argbuf_by_index[idx].append(arguments)

            except (AttributeError, IndexError, KeyError):
                continue

        # Build message
        message = {'role': 'assistant', 'content': "".join(final_content)}

        if calls_by_index:
            tool_calls: List[Dict[str, Any]] = []
            for idx in sorted(calls_by_index.keys()):
                base = calls_by_index[idx]
                arg_str = "".join(argbuf_by_index.get(idx, [])) if idx in argbuf_by_index else ""
                # Attempt a single parse for verification (we still pass the string along)
                json_ok = True
                try:
                    if arg_str:
                        json.loads(arg_str)
                except Exception:
                    json_ok = False
                # Fill fields with a well-formed dict that the rest of the pipeline expects
                tc_out = {
                    "id": base.get("id") or f"call_{idx}",
                    "type": base.get("type") or "function",
                    "function": {
                        "name": base.get("function", {}).get("name"),
                        "arguments": arg_str or "{}",
                    },
                }
                # We keep a flag for debugging; not used elsewhere
                if not json_ok:
                    tc_out["_arguments_json_error"] = True
                tool_calls.append(tc_out)

            message['tool_calls'] = tool_calls

        return {'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls_by_index else 'stop'}]}


def clamp_max_tokens_for_provider(model_name: str, requested: int) -> int:
    provider, _ = _infer_provider_env_vars(model_name)
    # Very conservative defaults; adjust if you know your exact model limits.
    if provider in ("OpenAI", "Anthropic", "Gemini"):
        CAP = 8192
        if requested > CAP:
            logger.warning("Lowering max_tokens from %d to %d for provider %s (safety cap)", requested, CAP, provider)
            return CAP
    return requested


async def amain(
    config_path: str,
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_tool_hops: Optional[int] = None,
    tool_result_max_chars: Optional[int] = None,
    tool_timeout_seconds: Optional[float] = None,
    system_prompt: Optional[str] = None,
    system_prompt_file: Optional[str] = None,
    log_level: Optional[str] = None,
    allow_prompt_txt: bool = True,
    provider_openai: bool = False,
    provider_gemini: bool = False,
    provider_anthropic: bool = False,
    provider_groq: bool = False,
    max_parallel_tools: Optional[int] = None,
    tool_preview_lines: int = DEFAULT_TOOL_PREVIEW_LINES,
    log_json: bool = False,
):
    global LOG_JSON
    LOG_JSON = bool(log_json)

    configure_logging(log_level)
    path_obj = pathlib.Path(config_path)
    print("=== LiteLLM MCP CLI Chat ===")
    print(f"Loading MCP configuration from: {path_obj}")
    print("Type 'quit' to exit. Commands: /new, /history, /tools, /reload, /clean\n")

    if not path_obj.exists():
        print(f"[Fatal] Config file not found: {path_obj}")
        print("Create a JSON file named mcp_config.json next to this script (see example below), or pass --config.")
        sys.exit(1)

    cfg_defaults = AppConfig.defaults()
    selected_model: Optional[str] = model or DEFAULT_MODEL
    if provider_openai:
        selected_model = DEFAULT_MODEL_OPENAI
    elif provider_anthropic:
        selected_model = DEFAULT_MODEL_ANTHROPIC
    elif provider_groq:
        selected_model = DEFAULT_MODEL_GROQ
    elif provider_gemini:
        selected_model = DEFAULT_MODEL_GEMINI

    model_for_llm = selected_model
    try:
        # Prefix mapping for providers if user passed short names
        if model_for_llm and "/" not in model_for_llm:
            if "gemini" in model_for_llm.lower():
                model_for_llm = f"gemini/{model_for_llm}"
            elif provider_groq:
                if model_for_llm == "gpt-oss-120b":
                    model_for_llm = "groq/openai/gpt-oss-120b"
                else:
                    model_for_llm = f"groq/{model_for_llm}"
    except Exception:
        pass

    provider_name, env_keys = _infer_provider_env_vars(model_for_llm or "")
    missing = [k for k in env_keys if not os.getenv(k)]
    if provider_name and missing:
        print(f"[Fatal] Missing API credentials for {provider_name}. Please set: {', '.join(missing)}")
        sys.exit(1)

    effective_max_tokens = cfg_defaults.max_tokens if max_tokens is None else max_tokens
    effective_max_tokens = clamp_max_tokens_for_provider(model_for_llm or "", effective_max_tokens)

    cfg = AppConfig(
        model=model_for_llm,
        max_tokens=effective_max_tokens,
        max_tool_hops=cfg_defaults.max_tool_hops if max_tool_hops is None else max_tool_hops,
        tool_result_max_chars=cfg_defaults.tool_result_max_chars if tool_result_max_chars is None else tool_result_max_chars,
    )

    if tool_timeout_seconds is not None and tool_timeout_seconds > 0:
        try:
            global TOOL_TIMEOUT_SEC
            TOOL_TIMEOUT_SEC = float(tool_timeout_seconds)
        except Exception:
            pass

    mcp = MCPToolRouter()
    orchestrator = LLMOrchestrator(cfg, mcp, tool_preview_lines=tool_preview_lines)
    if max_parallel_tools is not None and max_parallel_tools >= 0:
        orchestrator.max_parallel_tools = max_parallel_tools

    resolved_prompt: Optional[str] = None
    if system_prompt is not None:
        resolved_prompt = system_prompt
    elif system_prompt_file:
        p = pathlib.Path(system_prompt_file)
        if p.exists():
            try:
                resolved_prompt = p.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Could not read system prompt file %s: %r", p, e)
    elif allow_prompt_txt:
        p = SCRIPT_DIR / "prompt.txt"
        if p.exists():
            try:
                resolved_prompt = p.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Could not read prompt.txt: %r", e)

    if resolved_prompt is None or not resolved_prompt.strip():
        resolved_prompt = DEFAULT_SYSTEM_PROMPT

    orchestrator.system_prompt = resolved_prompt or ""
    if orchestrator.system_prompt:
        sp = orchestrator.system_prompt
        print(f"[Info] System prompt active: {sp[:120]}{'…' if len(sp) > 120 else ''}")

    # Start MCP
    try:
        mcp.load_config(path_obj)
        await mcp.start_all()
        tool_names = sorted(list(mcp.tool_to_server.keys()))
        print(f"✅ Connected. {len(mcp.servers)} server(s), {len(tool_names)} tool(s).")
        server_to_tools: Dict[str, List[str]] = {}
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
            if user == "/tools":
                for sname in sorted(server_to_tools.keys()):
                    tools = sorted(server_to_tools[sname])
                    print(f"  - {sname}: {tools}")
                continue
            if user == "/reload":
                print("[Info] Reloading MCP servers and tools...")
                try:
                    await mcp.cleanup()
                    mcp.load_config(path_obj)
                    await mcp.start_all()
                    tool_names = sorted(list(mcp.tool_to_server.keys()))
                    print(f"✅ Reloaded. {len(mcp.servers)} server(s), {len(tool_names)} tool(s).")
                    server_to_tools: Dict[str, List[str]] = {}
                    for tname, srv in mcp.tool_to_server.items():
                        sname = srv.name
                        prefix = f"{sname}_"
                        display = tname[len(prefix) :] if tname.startswith(prefix) else tname
                        server_to_tools.setdefault(sname, []).append(display)
                    for sname in sorted(server_to_tools.keys()):
                        tools = sorted(server_to_tools[sname])
                        print(f"  - {sname}: {tools}")
                except Exception as e:
                    logger.error("Reload failed: %r", e, exc_info=True)
                    print(f"[Error] Reload failed: {e}")
                continue
            if user == "/clean":
                request_cleanup_data = True
                print("[Info] Cleanup requested. Shutting down MCP servers… (may remove ./data)")
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
        try:
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
            try:
                total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS
                print(f"{TOTAL_PROMPT_TOKENS}/{TOTAL_COMPLETION_TOKENS}/{total_tokens} (${TOTAL_COST:.4f})")
            except Exception:
                pass
            print("[Info] Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Tiny MCP CLI Chat (supports /new, /history, /tools, /reload, /clean — note: /clean may remove ./data)"
    )
    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to mcp_config.json")
    parser.add_argument("--model", default=None, help="Model name (e.g., claude-3-7-sonnet-20250219)")
    parser.add_argument("-o", dest="provider_openai", action="store_true", help="Use OpenAI default model")
    parser.add_argument("-g", dest="provider_gemini", action="store_true", help="Use Gemini default model")
    parser.add_argument("-a", dest="provider_anthropic", action="store_true", help="Use Anthropic default model")
    parser.add_argument("-q", dest="provider_groq", action="store_true", help="Use Groq default model")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens per response (may be capped safely per provider)")
    parser.add_argument("--max-tool-hops", type=int, default=None, help="Max tool-use hops per user turn")
    parser.add_argument("--max-parallel-tools", type=int, default=DEFAULT_MAX_PARALLEL_TOOLS, help="Max parallel tool calls (0 = serial, None = unlimited)")
    parser.add_argument("--tool-preview-lines", type=int, default=DEFAULT_TOOL_PREVIEW_LINES, help="Print first N lines of each tool result")
    parser.add_argument("--tool-result-max-chars", type=int, default=None, help="Truncate tool results to this many chars")
    parser.add_argument("--tool-timeout-seconds", type=float, default=None, help="Per tool-call timeout in seconds")
    parser.add_argument("--system-prompt", default=None, help="Inline system prompt text (use --system-prompt '' to clear)")
    parser.add_argument("--system-prompt-file", default=None, help="Path to a system prompt text file")
    parser.add_argument("--no-prompt-txt", action="store_true", help="Do not auto-load prompt.txt next to the script")
    parser.add_argument("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-json", action="store_true", help="Emit structured JSON logs for key events to stderr")

    args = parser.parse_args()
    try:
        asyncio.run(amain(
            config_path=args.config,
            model=args.model,
            max_tokens=args.max_tokens,
            max_tool_hops=args.max_tool_hops,
            max_parallel_tools=args.max_parallel_tools,
            tool_preview_lines=args.tool_preview_lines,
            tool_result_max_chars=args.tool_result_max_chars,
            tool_timeout_seconds=args.tool_timeout_seconds,
            system_prompt=args.system_prompt,
            system_prompt_file=args.system_prompt_file,
            log_level=args.log_level,
            allow_prompt_txt=not args.no_prompt_txt,
            provider_openai=args.provider_openai,
            provider_gemini=args.provider_gemini,
            provider_anthropic=args.provider_anthropic,
            provider_groq=args.provider_groq,
            log_json=args.log_json,
        ))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Info] Application terminated.")


if __name__ == "__main__":
    main()
