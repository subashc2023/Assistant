
import os
import sys
import json
import pathlib
import tempfile
import types as _types
import builtins
import contextlib
import io
import asyncio
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import client as tc


class _FakeRouter:
    def __init__(self):
        self.servers = []
        self.tool_to_server = {}
        self.namespaced_to_original = {}
        self.openai_tool_specs = []

    def load_config(self, path: pathlib.Path) -> None:
        self.servers = [_types.SimpleNamespace(name="fake")]  # type: ignore

    async def start_all(self) -> None:
        self.tool_to_server = {"fake_list": self.servers[0]}
        self.namespaced_to_original = {"fake_list": "list"}
        self.openai_tool_specs = [{
            "type": "function",
            "function": {"name": "fake_list", "description": "", "parameters": {"type": "object"}},
        }]

    async def cleanup(self) -> None:
        return None


class _FakeOrchestrator:
    def __init__(self, cfg: tc.AppConfig, mcp: _FakeRouter, *, tool_preview_lines: int = 0):
        self.cfg = cfg
        self.mcp = mcp
        self.tool_preview_lines = tool_preview_lines
        self.max_parallel_tools = None
        self.system_prompt = ""
        self.conversation = []
        tc._last_orchestrator = self  # type: ignore[attr-defined]

    async def run_single_turn(self, user_text: str) -> str:
        self.conversation.append({"role": "user", "content": user_text})
        return ""


class TestRuntimeConfig(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self._old_router_cls = tc.MCPToolRouter
        self._old_orch_cls = tc.LLMOrchestrator
        tc.MCPToolRouter = _FakeRouter  # type: ignore
        tc.LLMOrchestrator = _FakeOrchestrator  # type: ignore
        os.environ.setdefault("GROQ_API_KEY", "test-key")

        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self.tmpdir.name)
        self.config_path = self.tmp_path / "mcp_config.json"
        self.config_path.write_text(json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": ["-c", "print('ok')"]}}}), encoding="utf-8")
        self._old_app_cfg = tc.APP_CONFIG_FILE
        tc.APP_CONFIG_FILE = self.tmp_path / "tinyclient_config.yaml"

    def tearDown(self):
        tc.MCPToolRouter = self._old_router_cls  # type: ignore
        tc.LLMOrchestrator = self._old_orch_cls  # type: ignore
        tc.APP_CONFIG_FILE = self._old_app_cfg
        self.tmpdir.cleanup()
        if hasattr(tc, "_last_orchestrator"):
            delattr(tc, "_last_orchestrator")

    async def _run_with_inputs(self, inputs):
        it = iter(inputs)
        def _fake_input(_prompt=""):
            try:
                return next(it)
            except StopIteration:
                return "quit"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            old_input = builtins.input
            builtins.input = _fake_input
            try:
                await tc.amain(str(self.config_path))
            finally:
                builtins.input = old_input
        return buf.getvalue()

    async def test_yaml_precedence_on_start(self):
        tc.APP_CONFIG_FILE.write_text(
            "\n".join([
                "log_level: DEBUG",
                "log_json: false",
                "max_tokens: 1234",
                "max_tool_hops: 7",
                "tool_result_max_chars: 333",
                "tool_timeout_seconds: 0.5",
                "max_parallel_tools: 2",
                "tool_preview_lines: 1",
                "system_prompt: |",
                "  test prompt",
                "model_aliases:",
                "  llamaL: groq/llama-3.3-70b-versatile",
            ]),
            encoding="utf-8",
        )
        await self._run_with_inputs(["quit"])
        orch = getattr(tc, "_last_orchestrator")
        self.assertEqual(orch.cfg.max_tokens, 1234)
        self.assertEqual(orch.cfg.max_tool_hops, 7)
        self.assertEqual(orch.cfg.tool_result_max_chars, 333)
        self.assertEqual(orch.tool_preview_lines, 1)
        self.assertEqual(orch.max_parallel_tools, 2)
        self.assertTrue(isinstance(tc.TOOL_TIMEOUT_SEC, float) and abs(tc.TOOL_TIMEOUT_SEC - 0.5) < 1e-6)
        self.assertTrue(orch.system_prompt.startswith("test prompt"))

    async def test_reload_reapplies_yaml(self):
        tc.APP_CONFIG_FILE.write_text(
            "\n".join([
                "max_tokens: 1000",
                "max_tool_hops: 5",
            ]),
            encoding="utf-8",
        )
        async def _change_yaml_then_reload():
            tc.APP_CONFIG_FILE.write_text(
                "\n".join([
                    "max_tokens: 2000",
                    "max_tool_hops: 9",
                ]),
                encoding="utf-8",
            )
        out_task = asyncio.create_task(self._run_with_inputs(["/history", "/reload", "quit"]))
        await _change_yaml_then_reload()
        await out_task
        orch = getattr(tc, "_last_orchestrator")
        self.assertEqual(orch.cfg.max_tokens, 2000)
        self.assertEqual(orch.cfg.max_tool_hops, 9)

    async def test_model_switch_and_history_preserved(self):
        tc.APP_CONFIG_FILE.write_text(
            "\n".join([
                "model_aliases:",
                "  llamaL: groq/llama-3.3-70b-versatile",
            ]),
            encoding="utf-8",
        )
        await self._run_with_inputs(["hi", "/model llamaL", "/reload", "quit"])
        orch = getattr(tc, "_last_orchestrator")
        self.assertEqual(orch.cfg.model, "groq/llama-3.3-70b-versatile")
        self.assertEqual(len(orch.conversation), 1)
        self.assertEqual(orch.conversation[0]["role"], "user")
        self.assertEqual(orch.conversation[0]["content"], "hi")
