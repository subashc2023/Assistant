### Anthropic MCP CLI Chat (tinyclient)

A minimal CLI that connects Anthropic's Messages API to one or more Model Context Protocol (MCP) servers over stdio. It aggregates tools exposed by MCP servers, namespaces them, and routes tool_use calls during a streamed Claude conversation.

### Features
- **MCP client**: Starts multiple MCP servers from `mcp_config.json`, lists tools, and routes calls.
- **Claude streaming**: Streams assistant tokens live; shows tool requests as they occur.
- **Multi-hop tool use**: Executes tool calls, returns results to the model, with retries and timeouts.
- **History control**: Bounded message count and optional character cap.
- **Simple UX**: Interactive REPL with `/new`, `/history`, and graceful cleanup.

### Requirements
- **Python**: >= 3.13
- **Anthropic API key**: set `ANTHROPIC_API_KEY` in your environment
- Tools per your `mcp_config.json` (e.g., Node + npx for filesystem server, Docker for sqlite server)

### Install
Option A — uv (quick start):
```powershell
uv run tinyclient.py
```

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
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python tinyclient.py

# Or pass a custom file via -c/--config
python tinyclient.py -c path\\to\\mcp_config.json
```
During a session:
- Type your message and press Enter
- `/new`: reset conversation
- `/history`: print current conversation buffer
- `/clean`: gracefully stop servers and delete the `data` directory (and the sqlite DB) next to the script
- `quit` or `exit`: leave the program

The CLI prints:
- `→ Tool requested: <name>` when Claude emits a tool_use
- `→ Tool: <name> {args}` before execution
- `← Result:` with the tool's normalized output (truncated to a safe length)

### Tuning
Edit constants in `tinyclient.py` if you want to change defaults:
- `DEFAULT_MODEL`
- `DEFAULT_MAX_TOKENS`
- `DEFAULT_MAX_TOOL_HOPS`
- `DEFAULT_TOOL_TIMEOUT_SEC`
- `DEFAULT_LLM_RETRIES`
- `DEFAULT_LLM_RETRY_BACKOFF_SEC`
- `DEFAULT_TOOL_RESULT_MAX_CHARS`

History limits (in code defaults):
- `DEFAULT_HISTORY_MAX_MESSAGES = 40`
- `DEFAULT_HISTORY_MAX_CHARS = 0` (0 disables char cap)

System prompt:
- The default `DEFAULT_SYSTEM_PROMPT` is a playful pirate persona. Edit `tinyclient.py` to change or clear it. For a neutral assistant, set it to an empty string.

### How it works (high-level)
- `MCPToolRouter` loads servers, aggregates tool schemas for Anthropic.
- `AnthropicOrchestrator` streams messages, detects tool_use blocks, executes tools via the router, and feeds `tool_result` back to the model until completion or hop limit.
- Stdout shows streaming text and tool activity for transparency.

### Troubleshooting
- "Config file not found": ensure `mcp_config.json` exists or pass `-c <path>`.
- "Requested command ... not found on PATH": install the tool and ensure PATH contains it (e.g., Node for `npx`, Docker for `docker`).
- Tool timeouts: increase `TOOL_TIMEOUT_SEC` or check server responsiveness.
- No tools loaded: confirm your servers actually expose tools and initialize without errors.
- Anthropic auth: set `ANTHROPIC_API_KEY` and verify network egress.
- Windows + Docker volumes: prefer absolute host paths for `-v`.

### License
Not specified. Add a license if you plan to distribute.
