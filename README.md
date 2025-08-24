### Anthropic MCP CLI Chat (tinyclient)

A minimal CLI that connects Anthropic's Messages API to one or more Model Context Protocol (MCP) servers over stdio. It aggregates tools exposed by MCP servers, namespaces them, and routes tool_use calls during a streamed Claude conversation.

### Features
- **MCP client**: Starts multiple MCP servers from `mcp_config.json`, lists tools, and routes calls.
- **Claude streaming**: Streams assistant tokens live; shows tool requests as they occur.
- **Multi-hop tool use**: Executes tool calls, returns results to the model, with retries and timeouts.
- **Simple UX**: Interactive REPL with `/new`, `/history`, and graceful cleanup.

### Requirements
- **Python**: >= 3.13
- **Anthropic API key**: set `ANTHROPIC_API_KEY` in your environment
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
$env:ANTHROPIC_API_KEY = "sk-ant-..."
uv run tinyclient.py

# Or pass a custom file via -c/--config
uv run tinyclient.py -c path\\to\\mcp_config.json
```
During a session:
- Type your message and press Enter
- `/new`: reset conversation
- `/history`: print current conversation buffer
- `/clean`: gracefully stop servers and delete the `data` directory (with the sqlite DB)
- `quit` or `exit`: leave the program

The CLI prints:
- `→ Tool requested: <name>` when Claude emits a tool_use
- `→ Tool: <name> {args}` before execution
- `← Result:` with the tool's normalized output (truncated to a safe length)

### CLI flags
All defaults can be overridden at runtime via flags:

```powershell
python tinyclient.py \
  --model claude-3-5-sonnet-20241022 \
  --max-tokens 1500 \
  --max-tool-hops 10 \
  --tool-result-max-chars 8000 \
  --system-prompt "You are helpful." \
  --system-prompt-file C:\\path\\to\\prompt.txt \
  --no-prompt-txt \
  --log-level INFO
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


