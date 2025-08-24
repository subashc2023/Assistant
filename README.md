### LiteLLM MCP CLI Chat (tinyclient)

A minimal CLI that connects LiteLLM's OpenAI-compatible chat API to one or more Model Context Protocol (MCP) servers over stdio. It aggregates tools exposed by MCP servers, namespaces them, and routes tool calls during a streamed conversation.

### Features
- **MCP client**: Starts multiple MCP servers from `mcp_config.json`, lists tools, and routes calls.
- **Streaming**: Streams assistant tokens live; shows tool requests as they occur.
- **Multi-hop tool use**: Executes tool calls, returns results to the model, with retries and timeouts.
- **Simple UX**: Interactive REPL with `/new`, `/history`, and graceful cleanup.

### Requirements
- **Python**: >= 3.13
- **Provider API key(s)**: e.g., set `OPENAI_API_KEY` (LiteLLM reads common provider keys)
- Tools per your `mcp_config.json` (e.g., Node + npx for filesystem server, Docker for sqlite server)

### Configure MCP servers
Define MCP servers in `mcp_config.json`. By default, the script looks for this file next to `tinyclient.py` (override with `-c/--config`).
```1:25:mcp_config.json
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
              "-y",
              "@modelcontextprotocol/server-filesystem",
              "C:\\workspace\\environment\\Assistant"
            ]
        },
        "sqlite": {
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                "-v",
                "./data:/mcp",
                "mcp/sqlite",
                "--db-path",
                "/mcp/test.db"
            ]
        }  
    } 
}
```
Notes:
- **filesystem**: requires Node.js/npm. `npx @modelcontextprotocol/server-filesystem <root>` exposes a directory.
- **sqlite**: requires Docker. On Windows, bind mounts may work better with absolute paths (e.g., `${pwd}/data:/mcp`).
- Tool names are **namespaced** by server (e.g., `filesystem_list`), and deduplicated if needed.

### Usage
Run the CLI:
```powershell
# Default: looks for mcp_config.json next to the script
$env:OPENAI_API_KEY = "sk-..."
python tinyclient.py

# Or pass a custom file via -c/--config
python tinyclient.py -c path\\to\\mcp_config.json
```
During a session:
- Type your message and press Enter
- `/new`: reset conversation
- `/history`: print current conversation buffer
- `/clean`: gracefully stop servers and delete the `data` directory (with the sqlite DB)
- `quit` or `exit`: leave the program

The CLI prints:
- `→ Tool: <name> {args}` before execution
- `← Result:` with the tool's normalized output (truncated to a safe length)

### CLI flags
All defaults can be overridden at runtime via flags:

```powershell
python tinyclient.py \
  --model gemini-2.5-pro \
  --max-tokens 1500 \
  --max-tool-hops 10 \
  --tool-timeout-seconds 30 \
  --tool-result-max-chars 8000 \
  --system-prompt "You are helpful." \
  --system-prompt-file C:\\path\\to\\prompt.txt \
  --no-prompt-txt \
  --log-level INFO
```
Provider shortcuts (default provider is Google/Gemini):

```powershell
# Gemini (default if none specified)
python tinyclient.py -g

# OpenAI
python tinyclient.py -o

# Anthropic
python tinyclient.py -a

# Groq
python tinyclient.py -q

# You can still override with an explicit --model at any time
python tinyclient.py -o --model gpt-4.1-mini
```


Notes:
- Flags are optional; unspecified ones use compiled defaults.
- `--no-prompt-txt` disables auto-loading `prompt.txt` next to the script.

### System prompt
- You can provide a system prompt inline with `--system-prompt` or via `--system-prompt-file`.
- If no flag is provided and a `prompt.txt` file exists next to `tinyclient.py`, it will be used automatically.
- If none of the above are provided, a compiled-in default will be used.

Precedence (highest to lowest): `--system-prompt` > `--system-prompt-file` > `prompt.txt` (same folder) > default.

### How it works (high-level)
- `MCPToolRouter` loads servers, aggregates tool schemas into OpenAI-style `tools=[{type:"function", function:{name, description, parameters}}]`.
- `LLMOrchestrator` streams messages via LiteLLM, detects `tool_calls`, executes them via the router, and appends `role:"tool"` messages with `tool_call_id` until completion or hop limit.
- Stdout shows streaming text and tool activity for transparency.

### Provider quickstart

- OpenAI
  - Set `OPENAI_API_KEY`
  - Example models: `gpt-4o-mini`, `gpt-4o`, `o3-mini`
  - Default here: `gpt-4o-mini`

- Anthropic
  - Set `ANTHROPIC_API_KEY`
  - Example models: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-latest`
  - Default here: `claude-3-5-sonnet-20241022`

- Gemini (Google)
  - Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
  - Example models: `gemini-1.5-pro`, `gemini-1.5-flash`
  - Default here: `gemini-1.5-pro`

The script will emit a warning if it detects a provider by model name and the likely env var isn’t set.

### Troubleshooting
- "Config file not found": ensure `mcp_config.json` exists or pass `-c <path>`.
- "Requested command ... not found on PATH": install the tool and ensure PATH contains it (e.g., Node for `npx`, Docker for `docker`).
- Tool timeouts: increase `TOOL_TIMEOUT_SEC` or check server responsiveness.
- No tools loaded: confirm your servers actually expose tools and initialize without errors.
- Auth: set the appropriate provider key(s), e.g., `OPENAI_API_KEY`, and verify network egress.
- Windows + Docker volumes: prefer absolute host paths for `-v`.

Gemini ADC error:
- If you see a `DefaultCredentialsError` referencing Google ADC, you're hitting the Vertex AI flow. This client defaults Gemini models to the AI Studio provider (`gemini/<model>`) when a `GEMINI_API_KEY`/`GOOGLE_API_KEY` is present. If you want Vertex, set `GOOGLE_APPLICATION_CREDENTIALS` and `VERTEXAI_PROJECT` (and optionally region) or pass a provider-prefixed model like `vertex_ai/gemini-2.5-pro`. If you want AI Studio, ensure your model is unprefixed and `GEMINI_API_KEY` is set; the code will map it to `gemini/<model>` for you.

### License
Not specified. Add a license if you plan to distribute.


