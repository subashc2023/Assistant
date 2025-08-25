import io
import os
import sys
import json
import tempfile
import pathlib
import contextlib
import unittest
import types as _types

if "litellm" not in sys.modules:
    litellm = _types.ModuleType("litellm")
    async def _acompletion(**_kwargs):
        class _Stream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration
        return _Stream()
    def _stream_chunk_builder(_chunks, messages=None):
        class _MsgObj:
            def __init__(self):
                self.message = {"content": ""}
        class _Resp:
            def __init__(self):
                self.choices = [_MsgObj()]
        return _Resp()
    litellm.acompletion = _acompletion
    litellm.stream_chunk_builder = _stream_chunk_builder
    sys.modules["litellm"] = litellm
if "anthropic" not in sys.modules:
    anthropic = _types.ModuleType("anthropic")
    class _APIStatusError(Exception):
        def __init__(self, status_code=500, message="error"):  # compat surface
            super().__init__(message)
            self.status_code = status_code
            self.message = message
    class _Anthropic:
        def __init__(self):
            pass
        class messages:  # dummy attribute to satisfy attribute access in code paths not hit here
            @staticmethod
            def stream(**kwargs):
                raise RuntimeError("stream() not available in tests")
    anthropic.Anthropic = _Anthropic
    anthropic.APIStatusError = _APIStatusError
    anthropic_types = _types.ModuleType("anthropic.types")
    class _Message:
        pass
    anthropic_types.Message = _Message
    sys.modules["anthropic"] = anthropic
    sys.modules["anthropic.types"] = anthropic_types

if "mcp" not in sys.modules:
    mcp = _types.ModuleType("mcp")
    class _ClientSession:
        def __init__(self, *_args, **_kwargs):
            pass
        async def initialize(self):
            return None
    class _StdioServerParameters:
        def __init__(self, **_kwargs):
            pass
    mcp.ClientSession = _ClientSession
    mcp.StdioServerParameters = _StdioServerParameters
    mcp_client = _types.ModuleType("mcp.client")
    mcp_client_stdio = _types.ModuleType("mcp.client.stdio")
    async def _stdio_client(_params):
        raise RuntimeError("stdio_client not used in tests")
    mcp_client_stdio.stdio_client = _stdio_client
    sys.modules["mcp"] = mcp
    sys.modules["mcp.client"] = mcp_client
    sys.modules["mcp.client.stdio"] = mcp_client_stdio

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import client as tc


class TestResolveCommand(unittest.TestCase):

    def test_absolute_existing_path_returns_same(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            resolved = tc._resolve_command(tmp_path)
            self.assertEqual(resolved, tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

    def test_invalid_command_raises(self):
        with self.assertRaises(RuntimeError):
            tc._resolve_command("this-command-should-not-exist-1234567890")


class TestFormatToolResultForLLM(unittest.TestCase):

    class _Text:
        def __init__(self, text: str):
            self.type = "text"
            self.text = text

    class _Blob:
        def __init__(self, payload):
            self.type = "blob"
            self._payload = payload
        def to_dict(self):
            return {"blob": self._payload}

    class _Result:
        def __init__(self, content):
            self.content = content

    def test_text_and_blob_aggregation(self):
        res = self._Result([
            self._Text("hello"),
            self._Blob({"k": "v"}),
            self._Text("world"),
        ])
        text, was_trunc, total = tc.format_tool_result_for_llm(res, max_chars=1000)
        self.assertIn("hello", text)
        self.assertIn("world", text)
        self.assertIn("\"blob\"", text)
        self.assertIn("\"k\": \"v\"", text)

    def test_truncation(self):
        long_text = "x" * 500
        res = self._Result([self._Text(long_text)])
        text, was_trunc, total = tc.format_tool_result_for_llm(res, max_chars=100)
        self.assertTrue(len(text) <= 100)
        self.assertTrue(was_trunc)


class TestUIToolPrinting(unittest.TestCase):

    def test_ui_tool_start_and_result_print(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tc.ui_tool_start("example_tool", {"a": 1})
            tc.ui_tool_result("example_tool", "ok", is_error=False, duration_s=0.0)
            tc.ui_tool_result("example_tool", "bad", is_error=True)
        text = buf.getvalue()
        self.assertIn("→ example_tool", text)
        self.assertIn("← example_tool: ok", text)
        self.assertIn("← example_tool: error", text)


class TestMCPToolRouter(unittest.IsolatedAsyncioTestCase):

    class _FakeTool:
        def __init__(self, name: str, description: str = "", input_schema: dict | None = None):
            self.name = name
            self.description = description
            self.inputSchema = input_schema or {"type": "object"}

    class _FakeToolsResp:
        def __init__(self, tools):
            self.tools = tools

    class _FakeServer:
        def __init__(self, name: str):
            self.name = name
            self.config = {}
            self.initialized = False
        async def initialize(self):
            self.initialized = True
        async def list_tools(self):
            return TestMCPToolRouter._FakeToolsResp([
                TestMCPToolRouter._FakeTool("list"),
                TestMCPToolRouter._FakeTool("list"),  # duplicate name to test dedup suffixing
                TestMCPToolRouter._FakeTool("info"),
            ])
        async def cleanup(self):  # not used here
            return None

    async def test_start_all_namespacing_and_registry(self):
        router = tc.MCPToolRouter()
        router.servers = [self._FakeServer("srvA"), self._FakeServer("srvB")]
        await router.start_all()
        self.assertEqual(len(router.openai_tool_specs), 6)
        names = {spec["function"]["name"] for spec in router.openai_tool_specs}
        self.assertIn("srvA_list", names)
        self.assertIn("srvA_list_2", names)
        self.assertIn("srvA_info", names)
        self.assertIs(router.tool_to_server["srvA_list"], router.servers[0])
        self.assertEqual(router.namespaced_to_original["srvA_list"], "list")

class TestLoadConfig(unittest.TestCase):

    def test_load_config_validates_mcpServers(self):
        router = tc.MCPToolRouter()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = pathlib.Path(tmpdir) / "cfg.json"
            p.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                router.load_config(p)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = pathlib.Path(tmpdir) / "cfg.json"
            cfg = {
                "mcpServers": {
                    "filesystem": {
                        "command": sys.executable,
                        "args": ["-c", "print('ok')"],
                    }
                }
            }
            p.write_text(json.dumps(cfg), encoding="utf-8")
            router.load_config(p)
            self.assertEqual(len(router.servers), 1)
            self.assertEqual(router.servers[0].name, "filesystem")


class TestRouterCallRouting(unittest.IsolatedAsyncioTestCase):

    class _Srv:
        def __init__(self, name: str):
            self.name = name
            self.called = []
        async def call_tool(self, tool_name: str, arguments: dict):
            self.called.append((tool_name, arguments))
            return {"ok": True, "tool": tool_name, "args": arguments}

    async def test_routing_to_original_tool_name(self):
        router = tc.MCPToolRouter()
        srv = self._Srv("alpha")
        router.tool_to_server = {"alpha_list": srv}
        router.namespaced_to_original = {"alpha_list": "list"}
        res = await router.call_tool("alpha_list", {"k": 1})
        self.assertEqual(srv.called, [("list", {"k": 1})])
        self.assertEqual(res["ok"], True)


class TestOrchestratorToolFlow(unittest.IsolatedAsyncioTestCase):

    class _Router:
        def __init__(self, result_delay: float = 0.0):
            self.openai_tool_specs = [{
                "type": "function",
                "function": {"name": "alpha_list", "description": "", "parameters": {"type": "object"}},
            }]
            self.calls = []
            self.result_delay = result_delay
        async def call_tool(self, name: str, arguments: dict):
            import asyncio as _asyncio
            self.calls.append((name, arguments))
            if self.result_delay:
                await _asyncio.sleep(self.result_delay)
            return {"ok": True, "name": name, "args": arguments}

    async def test_no_tool_use_returns_text(self):
        cfg = tc.AppConfig.defaults()
        router = self._Router()
        orch = tc.LLMOrchestrator(cfg, router)

        async def _fake_stream(history, tools, show_header):
            return {"content": "hello"}

        orch._call_llm_streaming = _fake_stream  # type: ignore
        out = await orch.run_single_turn("hi")
        self.assertIn("hello", out)
        self.assertEqual(router.calls, [])

    async def test_single_tool_use_then_finish(self):
        cfg = tc.AppConfig.defaults()
        router = self._Router()
        orch = tc.LLMOrchestrator(cfg, router)

        state = {"step": 0}
        async def _fake_stream(history, tools, show_header):
            if state["step"] == 0:
                state["step"] += 1
                return {
                    "content": "working",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "alpha_list", "arguments": json.dumps({"path": "/"})}}
                    ],
                }
            else:
                return {"content": "done"}

        orch._call_llm_streaming = _fake_stream  # type: ignore
        cap = io.StringIO()
        with contextlib.redirect_stdout(cap):
            out = await orch.run_single_turn("list please")
        self.assertIn("working", out)
        self.assertIn("done", out)
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(router.calls[0][0], "alpha_list")
        self.assertEqual(router.calls[0][1], {"path": "/"})

    async def test_duplicate_tool_calls_blocked(self):
        cfg0 = tc.AppConfig.defaults()
        cfg = tc.AppConfig(
            model=cfg0.model,
            max_tokens=cfg0.max_tokens,
            max_tool_hops=1,
            tool_result_max_chars=cfg0.tool_result_max_chars,
        )
        router = self._Router()
        orch = tc.LLMOrchestrator(cfg, router)

        async def _fake_stream(history, tools, show_header):
            return {
                "content": "",
                "tool_calls": [
                    {"id": "t1", "function": {"name": "alpha_list", "arguments": json.dumps({"x": 1})}},
                    {"id": "t2", "function": {"name": "alpha_list", "arguments": json.dumps({"x": 1})}},
                ],
            }

        orch._call_llm_streaming = _fake_stream  # type: ignore
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            await orch.run_single_turn("run twice")
        self.assertEqual(len(router.calls), 1)
        self.assertIn("repeated identical tool call prevented", buf.getvalue())

    async def test_tool_timeout_sets_error(self):
        cfg0 = tc.AppConfig.defaults()
        cfg = tc.AppConfig(
            model=cfg0.model,
            max_tokens=cfg0.max_tokens,
            max_tool_hops=1,
            tool_result_max_chars=cfg0.tool_result_max_chars,
        )
        import client as _tc
        old_timeout = getattr(_tc, "TOOL_TIMEOUT_SEC", 30.0)
        _tc.TOOL_TIMEOUT_SEC = 0.01
        router = self._Router(result_delay=0.05)
        orch = tc.LLMOrchestrator(cfg, router)

        async def _fake_stream(history, tools, show_header):
            return {
                "content": "",
                "tool_calls": [
                    {"id": "t1", "function": {"name": "alpha_list", "arguments": json.dumps({})}},
                ],
            }

        orch._call_llm_streaming = _fake_stream  # type: ignore
        buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err_buf):
            await orch.run_single_turn("slow tool")
        self.assertIn("timed out", buf.getvalue())
        _tc.TOOL_TIMEOUT_SEC = old_timeout


if __name__ == "__main__":
    unittest.main()


