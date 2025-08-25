## tinyclient — MCP x LiteLLM single-file chat

Tiny, battery-included REPL that connects LiteLLM to one or more MCP servers over stdio. It lists tools, namespaces them by server, streams assistant output, and executes tool calls with retries, timeouts, and bounded parallelism.

### Why you might want this
- **One file**: drop `tinyclient.py` anywhere, no framework.
- **Real tools**: speak MCP; run any stdio MCP server you point at.
- **Great UX**: streaming tokens, live tool call logs, `/reload` without restart.

### Requirements
- **Python**: 3.10+ (developed and tested on 3.13)
- **Pip packages**: `pip install -U litellm mcp pyyaml`
- **Provider key(s)**: set one or more of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `GROQ_API_KEY`
- **Your MCP servers**: whatever you reference in `mcp_config.json` (Node for filesystem server, Docker for others, etc.)

### Quick start
```bash
# 1) Install deps
pip install -U litellm mcp pyyaml

# 2) Put a config next to tinyclient.py
echo {"mcpServers":{}} > mcp_config.json  # fill this in; see below

# 3) Set an API key (example: Groq)
export GROQ_API_KEY=sk-...

# 4) Run
python tinyclient.py
```

Windows (PowerShell) quick start:
```powershell
pip install -U litellm mcp pyyaml
Set-Content -Path mcp_config.json -Value '{"mcpServers":{}}'
$env:GROQ_API_KEY = 'sk-...'
python .\tinyclient.py
```

### MCP servers (mcp_config.json)
Place a `mcp_config.json` next to `tinyclient.py` (or pass `--config`). Example with filesystem and sqlite:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
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
Notes:
- **filesystem** needs Node.js; `npx @modelcontextprotocol/server-filesystem <root>` exposes a directory.
- **sqlite** needs Docker; on Windows prefer absolute host paths for `-v`.
- Tool names are namespaced by server (e.g., `filesystem_list`) and deduped if needed.

### App config (tinyclient_config.yaml)
Optional file next to `tinyclient.py` for defaults/aliases and runtime settings. CLI flags override YAML.
```yaml
system_prompt: |
  You are a helpful assistant.

model_aliases:
  # Aliases you can pass via --model
  "4o-mini": "gpt-4o-mini"
  "sonnet": "claude-3-7-sonnet-20250219"
  "pro": "gemini/gemini-2.5-pro"
  "llamaL": "groq/llama-3.3-70b-versatile"

# Runtime settings (all optional)
log_level: INFO          # or DEBUG, WARNING, ERROR
log_json: false          # emit structured logs to stderr
max_tokens: 4096         # safely capped depending on provider
max_tool_hops: 25
tool_result_max_chars: 8000
tool_timeout_seconds: 45
max_parallel_tools: 4    # 0=serial, N>0 bounded
tool_preview_lines: 0    # print first N lines of each tool result
```
Precedence for the system prompt: `--system-prompt` > `--system-prompt-file` > `tinyclient_config.yaml` > legacy `prompt.txt` (if present) > built-in default.
Other settings precedence: CLI flag > `tinyclient_config.yaml` > built-in default.

### Run it
```bash
# default: reads ./mcp_config.json
python tinyclient.py

# choose provider defaults quickly
python tinyclient.py -o   # OpenAI
python tinyclient.py -a   # Anthropic
python tinyclient.py -g   # Gemini
python tinyclient.py -q   # Groq

# override model explicitly (alias or full name)
python tinyclient.py --model 4o-mini
```

REPL commands:
- `/new`: reset conversation
- `/history`: dump current messages
- `/tools`: list servers → tools
- `/model`: list model aliases from YAML; `/model <alias|full-name>` switches model without adding a message
- `/reload`: restart servers and re-list tools
- `/clean`: shutdown; may remove `./data` (helpful for Docker volume examples)
- `quit`/`exit`: leave

### CLI flags
```text
--config PATH                Path to mcp_config.json
--model NAME                 Model name or alias (e.g., sonnet, gpt-4o-mini)
-o | -a | -g | -q            Provider shortcuts (OpenAI | Anthropic | Gemini | Groq)
--max-tokens N               Max tokens per response (safely capped for some providers)
--max-tool-hops N            Max tool-use hops per user turn
--max-parallel-tools N       0 = serial, N > 0 = bounded concurrency (default 4)
--tool-preview-lines N       Print first N lines of each tool result (0 = off)
--tool-result-max-chars N    Truncate tool results to this many chars (default 8000)
--tool-timeout-seconds SEC   Per tool-call timeout (default 30)
--system-prompt TEXT         Inline system prompt
--system-prompt-file PATH    Load system prompt from file
--log-level LEVEL            DEBUG | INFO | WARNING | ERROR
--log-json                   Emit structured JSON logs to stderr
```

### What you’ll see
- Streaming assistant tokens on stdout.
- Tool calls logged as they happen: `→ [n] server_tool {args}`.
- Summaries when tools resolve: `← [n] server_tool: ok (1.23s) [42 lines]`.
- Tool results are fed back to the model (truncated to `--tool-result-max-chars`).

### Provider/auth notes
- The client infers provider from the model name and requires relevant env vars:
  - OpenAI → `OPENAI_API_KEY`
  - Anthropic → `ANTHROPIC_API_KEY`
  - Gemini → `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - Groq → `GROQ_API_KEY`
- Gemini ADC gotcha: if you hit Google ADC errors you’re on a Vertex path. Prefer AI Studio with `gemini/<model>` and `GEMINI_API_KEY`, or explicitly use Vertex via provider-prefixed models.

### Implementation details
- MCP servers are spawned via stdio. Tool input schemas must be JSON Schema objects.
- Tools are converted to OpenAI-style function tools and namespaced `<server>_<tool>`.
- Multi-hop tool use with retry/backoff on LLM calls; bounded tool concurrency (`--max-parallel-tools`).
- Safe token caps per provider to avoid over-asking.

### Troubleshooting
- Config missing: create or pass `--config`.
- Command not found: ensure `command` in `mcp_config.json` exists on PATH.
- No tools loaded: your server may have started without tools or failed to init.
- Timeouts: raise `--tool-timeout-seconds` or inspect server logs.
- Windows + Docker: use absolute host paths for `-v` mounts.

### License
Unlicensed. Add your own license file if you plan to distribute.

