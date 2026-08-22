use crate::mcp::daemon_rpc::{DaemonRpcRequest, DaemonRpcResponse};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Stdout};
use tokio::net::UnixStream;

/// Fallback only. Used when the client's `initialize` omits a version.
const DEFAULT_MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Must match `[mcp_servers.<name>]` in `data/config/codex-config.toml`, and the
/// name the instructions tell the agent to call tools on.
pub const MCP_SERVER_NAME: &str = "tera";

/// Tool definitions, from `data/config/mcp-tools.json`.
///
/// Parsed rather than embedded as a `json!` literal so descriptions, which are
/// prompt text the model reads. Live with every other prompt in `data/`. That
/// also means they carry `{{OWNER}}` and have to be rendered before the model sees
/// them: a tool whose description says "send a message to {{OWNER}}" is a worse
/// prompt than one that names the actual person.
///
/// The parse is validated by a test, so a failure here means the binary was built
/// from a broken file.
fn tool_definitions(owner: &str) -> Result<Value> {
    let rendered = crate::data::render(crate::data::MCP_TOOLS_JSON, &[("OWNER", owner)]);
    serde_json::from_str(&rendered).context("embedded data/config/mcp-tools.json is not valid JSON")
}

/// Writes a JSON-RPC message as a line and flushes. The flush is required: a
/// missing flush hangs the client waiting for a response it can't see yet.
async fn respond(stdout: &mut Stdout, value: &Value) -> Result<()> {
    let line = serde_json::to_string(value)? + "\n";
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

pub struct StdioMcpProxy {
    socket_path: PathBuf,
    /// Only used to render the tool descriptions. The proxy is a child process
    /// addressed by socket and has no workspace of its own, so it is passed in
    /// rather than read from a `Config`.
    owner_name: String,
}

impl StdioMcpProxy {
    pub fn new(socket_path: PathBuf, owner_name: String) -> Self {
        Self {
            socket_path,
            owner_name,
        }
    }

    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin).lines();

        while let Ok(Some(line)) = reader.next_line().await {
            if line.trim().is_empty() {
                continue;
            }

            let req_val: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let id = req_val.get("id").cloned();
            let method = req_val["method"].as_str().unwrap_or_default();

            if method == "initialize" {
                // Echo the client's protocol version instead of pinning one:
                // Codex refuses to complete the handshake on a version it did
                // not ask for, and a failed required MCP server is fatal to the
                // whole thread.
                let protocol_version = req_val["params"]["protocolVersion"]
                    .as_str()
                    .unwrap_or(DEFAULT_MCP_PROTOCOL_VERSION);
                let init_resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": protocol_version,
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": MCP_SERVER_NAME,
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }
                });
                respond(&mut stdout, &init_resp).await?;
            } else if method == "tools/list" {
                let tools_resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "tools": tool_definitions(&self.owner_name)? }
                });
                respond(&mut stdout, &tools_resp).await?;
            } else if method == "tools/call" {
                let tool_name = req_val["params"]["name"].as_str().unwrap_or_default();
                let tool_args = req_val["params"]["arguments"].clone();

                match self.forward_to_daemon(1, tool_name, tool_args).await {
                    Ok(val) => {
                        let tool_resp = json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": serde_json::to_string(&val)?
                                    }
                                ]
                            }
                        });
                        respond(&mut stdout, &tool_resp).await?;
                    }
                    Err(e) => {
                        let err_resp = json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32603,
                                "message": e.to_string()
                            }
                        });
                        respond(&mut stdout, &err_resp).await?;
                    }
                }
            } else if id.is_some() {
                let ack_resp = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {}
                });
                respond(&mut stdout, &ack_resp).await?;
            }
        }

        Ok(())
    }

    async fn forward_to_daemon(&self, id: u64, name: &str, args: Value) -> Result<Value> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to daemon socket at {:?}",
                    self.socket_path
                )
            })?;

        let (reader, mut writer) = stream.into_split();

        let req = DaemonRpcRequest {
            id,
            tool_name: name.to_string(),
            arguments: args,
        };

        let req_line = serde_json::to_string(&req)? + "\n";
        writer.write_all(req_line.as_bytes()).await?;
        writer.flush().await?;

        let mut buf_reader = BufReader::new(reader);
        let mut resp_line = String::new();
        buf_reader.read_line(&mut resp_line).await?;

        let daemon_resp: DaemonRpcResponse = serde_json::from_str(&resp_line)?;
        if let Some(err) = daemon_resp.error {
            return Err(anyhow!("Daemon tool error: {}", err));
        }

        Ok(daemon_resp.result.unwrap_or(Value::Null))
    }
}
