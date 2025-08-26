Assistant (LiteLLM + MCP CLI)
================================

Minimal, fast CLI that streams responses from an LLM (via LiteLLM) and executes Model Context Protocol (MCP) tools over stdio. It auto-discovers tools from configured MCP servers and exposes them to the model as OpenAI-style function tools. Built to be simple and reliable, with parallel calls, timeouts, and result truncation for Functions/Tools

Supports MCP in two ways:
- Local Python MCP servers discovered from the `servers/` directory
- Configured MCP servers via `mcp_config.json` (e.g., npx and Docker commands)

Why this exists
----------------
- Simple: a single Python script wired to LiteLLM and MCP - in ~1000 LoC
- Productive: live streaming output, tooling discovery, and batch tool execution
- Practical: provider auto-detect, token caps, structured tool results, and sane defaults
- Efficient: Starts MCP Server Docker Containers, and automatically cleans up on quit/termination

Features
--------
- Streaming assistant output with tool-call suppression to avoid half-answers
- MCP server discovery from `mcp_config.json` and from local `servers/*.py` (stdio transport)
- Namespaced tool exposure to the LLM (e.g., `filesystem_readFile`)
- Parallel tool execution with semaphore control and per-call timeout
- Tool result truncation and preview printing
- Undo/Redo of conversation turns
- Redaction of secrets (API keys, tokens) in tool call previews
- Provider detection (OpenAI, Anthropic, Gemini, Groq) and `max_tokens` capping
- Basic usage metrics (tokens and total cost if available)
- Executes duplicate tool calls when requested (no dedup) — supports non-idempotent tools

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

PowerShell (temporary):
```
$env:<PROVIDER>_API_KEY = "<API_KEY>"
```
PowerShell (persistent):
```
Add-Content $PROFILE '$env:VARIABLE_NAME = "value"'
. $PROFILE #to reload shell
```

CMD (temporary):
```
set <PROVIDER>_API_KEY = <API_KEY>
```
CMD (persistent):
```
setx <PROVIDER>_API_KEY <API_KEY>  #not reflected in current prompt
```

BASH (temporary):
```
export <PROVIDER>_API_KEY = <API_KEY>
```
BASH (persistent):
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
Ensure `mcp_config.json` exists in the repo root (a starter is already provided):
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

Local Python MCP servers (auto-discovery)
-----------------------------------------
Place Python MCP servers in the `servers/` folder at the repo root. Each `*.py` file (excluding `__init__.py`) will be auto-discovered and launched with your current Python interpreter via stdio.

Rules and behavior:
- The server name is the filename stem. Example: `servers/weather.py` -> server name `weather`.
- Discovered servers are merged with `mcp_config.json`. If a name conflict exists, the config entry wins.
- Discovery looks in `./servers` (next to `mcp_config.json`) and also next to `client.py`.
- No extra config needed; environment variables are inherited.

Example layout:
```
servers/
  weather.py   # exposes tools with FastMCP/stdio; becomes `weather_*` tools
```

You can run the client and see discovered servers in the startup summary and via `/tools`.

You should see something like:
```
=== LiteLLM MCP CLI Chat ===
Model: groq/llama-3.3-70b-versatile
Config: mcp_config.json
Commands: /new, /tools, /model, /reload, /undo, /redo, /quit
✅ Connected: 2 servers, N tools
  - filesystem: [...]
  - sqlite: [...]

[You] ...
```

Usage
-----
Type a prompt at `[You]`. The assistant will stream output. 

Note : LiteLLM Does not stream tool calls, as some of its constituent providers do not support this feature. :(

Commands
- `/new` reset the conversation
- `/tools` list discovered tools per MCP server
- `/model` show current model and aliases
- `/model <alias|full>` switch model (env must be set for that provider)
- `/reload` reconnect to MCP servers and rebuild tool list
- `/undo` Reverts the last conversation turn. Your previous input is prefilled.
- `/redo` Restores the last undone turn.
- `/clear` Deletes the entire `data/` directory (including `.chat_history` and any SQLite DB) and exits.
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
--log-json                       Output structured JSON logs to stderr
```

Data directory and history
--------------------------
- Chat history is stored at `data/.chat_history` and is created at startup.
- MCP server sample(s) may create `data/test.db` (via the SQLite Docker container).
- Use `/clear` to delete the entire `data/` directory in one go, while quitting (removes the DB and history).

Built-in model aliases
----------------------
Aliases are built-in and loaded into LiteLLM automatically. Use `/model <alias>` to switch.

Aliases are defined in config.py



Notes:
- Provider auth and token caps are validated against the resolved target model.
- `/model` without args lists the current model and all aliases in config.

Configuration details
---------------------

Providers and models
- Defaults: see `config.py:DEFAULTS`
- Provider detection: prefix/keyword heuristics (`detect_provider`)
- Token caps per provider: `config.py:PROVIDERS[*].max_tokens_cap`; enforced by `_apply_token_cap`

Environment variables - need at least one of the following
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Groq: `GROQ_API_KEY`

MCP servers (`mcp_config.json`)
- Each entry must specify `command` and `args`; optional `env` with string values
- Commands resolve either as absolute paths or via `PATH`
- Tool names are exposed as `<server>_<tool>` to the model
- Tool input schemas are automatically made strict (`additionalProperties: false`) unless overridden.

Local MCP servers (`servers/*.py`)
- Auto-discovered and executed with `sys.executable` over stdio
- Name is derived from file stem; tools are exposed as `<server>_<tool>`
- Overrides: entries in `mcp_config.json`

# TODO
  - /dump to dump raw json to file
  - /save to dump reader friendly chat history to file
  - smart truncation of tool result jsons to save to file if its too big, and show schema, instead of filling up context window.
  - smart history management with rolling window, to drop old tool calls, thinking sessions, and really old messages, that probably wont be needed
  - even smarter history management that uses another LLM to summarize message history on the fly
  - Continous Heirarchical Architecture, with 2 models running at all times, A small one managing state, context, etc. with another large one actually performing actions as necessary.