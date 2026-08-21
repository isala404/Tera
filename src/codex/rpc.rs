use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: &str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        }
    }
}

/// `codex app-server` omits the `jsonrpc` member on responses and notifications
/// (verified against codex-cli 0.147.0), so it must not be required to parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jsonrpc: Option<String>,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

// Notifications are deliberately not modelled as a struct: they are read as raw
// JSON, because their `params` shape is Codex's and changes with it, and a strict
// type here would silently drop events we could still route by name.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_jsonrpc_request_serde() {
        let req = JsonRpcRequest::new(1, "initialize", Some(json!({"param": "value"})));
        let json_str = serde_json::to_string(&req).unwrap();
        assert!(json_str.contains("\"jsonrpc\":\"2.0\""));
        assert!(json_str.contains("\"method\":\"initialize\""));

        let deserialized: JsonRpcRequest = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.id, 1);
        assert_eq!(deserialized.method, "initialize");
    }

    #[test]
    fn test_parses_response_without_jsonrpc_member() {
        // Verbatim shape emitted by codex-cli 0.147.0 `app-server`.
        let line = r#"{"id":1,"result":{"codexHome":"/home/ada/.codex"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(line).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.jsonrpc.is_none());
        assert_eq!(resp.result.unwrap()["codexHome"], "/home/ada/.codex");
    }

    /// A server-initiated request parses perfectly well as a response: same
    /// shape, minus `result`. That is why the reader dispatches on `method`
    /// first, the app-server numbers its requests from its own sequence, so
    /// parsing one as a response resolved whichever of our pending requests
    /// happened to share that id, with an empty result, and left the approval
    /// unanswered until the turn timed out.
    #[test]
    fn test_a_server_request_would_pass_for_a_response() {
        let line = r#"{"id":3,"method":"item/commandExecution/requestApproval","params":{}}"#;
        let parsed: JsonRpcResponse = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.id, 3);
        assert!(parsed.result.is_none() && parsed.error.is_none());
    }

    /// The reader tries a response first and falls back to raw JSON, so a
    /// notification must not parse as a response. Otherwise every event would be
    /// swallowed as a reply to a request nobody made.
    #[test]
    fn test_notification_is_not_parsed_as_response() {
        let line = r#"{"method":"turn/completed","params":{"threadId":"t1"},"emittedAtMs":1}"#;
        assert!(serde_json::from_str::<JsonRpcResponse>(line).is_err());
        assert_eq!(
            serde_json::from_str::<Value>(line).unwrap()["method"],
            "turn/completed"
        );
    }
}
