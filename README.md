Assistant (LiteLLM + MCP CLI)
================================

Minimal, fast CLI that streams responses from an LLM (via LiteLLM) and executes Model Context Protocol (MCP) tools over stdio. It auto-discovers tools from configured MCP servers and exposes them to the model as OpenAI-style function tools. Built for reliability: parallel tool calls, timeouts, truncation, and duplicate-call prevention.

Why this exists
----------------
- Simple: a single Python script wired to LiteLLM and MCP
- Productive: live streaming output, tooling discovery, and batch tool execution
- Practical: provider auto-detect, token caps, structured tool results, and sane defaults

Features
--------
- Streaming assistant output with tool-call suppression to avoid half-answers
- MCP server discovery from `mcp_config.json` (stdio transport)
- Namespaced tool exposure to the LLM (e.g., `filesystem_readFile`)
- Parallel tool execution with semaphore control and per-call timeout
- Tool result truncation and preview printing
- Duplicate tool-call prevention per batch
- Provider detection (OpenAI, Anthropic, Gemini, Groq) and `max_tokens` capping
- Basic usage metrics (tokens and total cost if available)

Quickstart
----------

Prereqs
- Python 3.13+
- One or more provider API keys (depending on your chosen model)
- For the sample MCP servers:
  - Node.js (for `npx @modelcontextprotocol/server-filesystem`)
  - Docker (for the sample SQLite MCP container)

# Install uv with 
Windows
``` 
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
unix - Use wget or curl
```
< wget -q0- > or < curl -LsSf > https://astral.sh/uv/install.sh | sh
```
or if you already have pip/python
```
pip install uv
```

# Configure API Keys
## Example 

(PowerShell - temporary):
```
$env:<PROVIDER>_API_KEY = "<API_KEY>"
```
(PowerShell - persistent):
```
Add-Content $PROFILE '$env:VARIABLE_NAME = "value"'
. $PROFILE #to reload shell
```

(CMD - temporary):
```
set <PROVIDER>_API_KEY = <API_KEY>
```
(CMD - persistent):
```
setx <PROVIDER>_API_KEY <API_KEY>  #not reflected in current prompt
```

(BASH - temporary):
```
export <PROVIDER>_API_KEY = <API_KEY>
```
(BASH - persistent):
```
echo 'export <PROVIDER>_API_KEY=<API_KEY>' >> ~/.bashrc
source ~/.bashrc
```

# Run
```
uv run client.py                                # Default model
uv run client.py -o                             # OpenAI default model
uv run client.py --config mcp_config.json --model openai/gpt-4o-mini 
```


Configure MCP servers
Ensure `mcp_config.json` exists in the repo root (a starter is provided):
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\workspace\\environment\\Assistant"]
    },
    "sqlite": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "./data:/mcp",
        "mcp/sqlite",
        "--db-path", "/mcp/test.db"
      ]
    }
  }
}
```



You should see something like:
```
=== LiteLLM MCP CLI Chat ===
Model: groq/llama-3.3-70b-versatile
Config: mcp_config.json
Type 'quit' to exit. Commands: /new, /history, /tools, /model, /reload
✅ Connected: 2 servers, N tools
  - filesystem: [...]
  - sqlite: [...]

[You]
```

Usage
-----
Type a prompt at `[You]`. The assistant will stream output. 
Note : LiteLLM Does not stream tool calls, as some of its constituent providers do not support this feature. :(

Commands
- `/new` reset the conversation
- `/history` print conversation JSON
- `/tools` list discovered tools per MCP server
- `/model` show current model and (if configured) aliases
- `/model <alias|full>` switch model (env must be set for that provider)
- `/reload` reconnect to MCP servers and rebuild tool list
- `/quit` or `/exit` exit

CLI flags
```
uv run client.py --help

--config PATH                    Path to mcp_config.json
--model NAME                     Model name or alias
-o | -a | -g | -q                OpenAI | Anthropic | Gemini | Groq defaults
--max-tokens N                   Max response tokens (capped per provider)
--max-tool-hops N                Max tool iterations per user turn (default 50)
--tool-result-max-chars N        Truncate tool results (default 8000)
--tool-timeout-seconds S         Per-tool timeout (default 30s)
--max-parallel-tools N           Parallel tool calls (default 4; 0=serial)
--tool-preview-lines N           Print first N lines of tool results to console
--system-prompt TEXT             System prompt inline
--system-prompt-file PATH        System prompt from file
--log-level LEVEL                DEBUG|INFO|WARNING|ERROR
--log-json                       (reserved) emit JSON logs (see roadmap)
```

Configuration details
---------------------

Providers and models
- Defaults: see `config.py:DEFAULTS`
- Provider detection: prefix/keyword heuristics (`detect_provider`)
- Token caps per provider: `config.py:PROVIDERS[*].max_tokens_cap`; enforced by `_apply_token_cap`

Environment variables
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Groq: `GROQ_API_KEY`

MCP servers (`mcp_config.json`)
- Each entry must specify `command` and `args`; optional `env` with string values
- Commands resolve either as absolute paths or via `PATH`
- Tool names are exposed as `<server>_<tool>` to the model

How it works (internals)
------------------------
- `MCPRouter` starts all MCP servers (stdio) and collects their tools and JSON schemas
- Tools are exposed to LiteLLM as OpenAI-style `tools` entries
- `LLMOrchestrator` streams completions, captures partial tool_calls, and batches execution via `ToolExecutor`
- Tool outputs are formatted and appended back to the conversation as `role=tool` messages
- The loop continues up to `max_tool_hops` or until no tools are requested

Troubleshooting
---------------
- Missing API credentials
  - On start: `[Fatal] Missing API credentials: ...` → export the env var(s) for your provider
- No MCP tools discovered
  - Run `/tools` to verify; check `mcp_config.json`; ensure `npx`/`docker` on PATH
- Docker volume path on Windows
  - `./data:/mcp` is relative to current working directory; create `data` folder or use an absolute path
- Streaming shows nothing after `[Assistant]`
  - The model emitted tool_calls; output is suppressed until tools finish. Increase `--tool-timeout-seconds` or check server health
- Tool output too long
  - Increase `--tool-result-max-chars` or enable `--tool-preview-lines` to see a snippet


Code layout
- `client.py` runtime, CLI, MCP orchestration, streaming, tool execution
- `config.py` defaults, provider detection, token caps, metrics helpers
- `mcp_config.json` sample MCP server setup (filesystem + sqlite)
