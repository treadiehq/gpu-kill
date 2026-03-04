//! MCP Protocol Types for GPU Kill

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// JSON-RPC 2.0 Request ID type
///
/// Per JSON-RPC 2.0 specification, the id field can be a String, Number, or Null.
/// Per MCP specification, the id MUST be a string or integer (null is not allowed).
///
/// This enum supports all valid JSON-RPC 2.0 id types for maximum compatibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// String identifier
    String(String),
    /// Integer identifier (JSON-RPC uses Number, we use i64 for integers)
    Number(i64),
    /// Null identifier (valid in JSON-RPC 2.0, but not in MCP)
    Null,
}

impl Default for RequestId {
    fn default() -> Self {
        RequestId::Null
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestId::String(s) => write!(f, "{}", s),
            RequestId::Number(n) => write!(f, "{}", n),
            RequestId::Null => write!(f, "null"),
        }
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        RequestId::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        RequestId::String(s.to_string())
    }
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        RequestId::Number(n)
    }
}

impl From<i32> for RequestId {
    fn from(n: i32) -> Self {
        RequestId::Number(n as i64)
    }
}

fn validate_jsonrpc_version<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = String::deserialize(deserializer)?;
    if version != "2.0" {
        return Err(serde::de::Error::custom(format!(
            "jsonrpc must be '2.0', got '{}'",
            version
        )));
    }
    Ok(version)
}

fn deserialize_request_id<'de, D>(deserializer: D) -> Result<Option<RequestId>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Deserialize as RequestId directly so an explicit JSON `null` is captured
    // as RequestId::Null rather than being swallowed as Option::None.
    let id = RequestId::deserialize(deserializer)?;
    if matches!(id, RequestId::Null) {
        return Err(serde::de::Error::custom("jsonrpc id must not be null"));
    }
    Ok(Some(id))
}

/// MCP Request/Response types
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(deserialize_with = "validate_jsonrpc_version")]
    pub jsonrpc: String,
    /// Request identifier - optional for JSON-RPC notifications
    #[serde(
        default,
        deserialize_with = "deserialize_request_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub id: Option<RequestId>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    #[serde(deserialize_with = "validate_jsonrpc_version")]
    pub jsonrpc: String,
    /// Response identifier - must match the request id per JSON-RPC 2.0
    pub id: RequestId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// MCP Protocol Messages
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeRequest {
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    pub client_info: ClientInfo,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResponse {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootsCapability {
    pub list_changed: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingCapability {}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    pub subscribe: Option<bool>,
    pub list_changed: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    pub list_changed: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoggingCapability {}

/// Resource Types
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Resource {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContents {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>, // Base64 encoded
}

/// Tool Types
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub name: String,
    pub arguments: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    pub is_error: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// GPU Kill specific types
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub id: u32,
    pub name: String,
    pub vendor: String,
    pub memory_used: f64,
    pub memory_total: f64,
    pub utilization: f64,
    pub temperature: Option<f64>,
    pub power_usage: Option<f64>,
    pub processes: Vec<GpuProcess>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    pub memory_usage: f64,
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThreatInfo {
    pub id: String,
    pub threat_type: String,
    pub severity: String,
    pub confidence: f64,
    pub description: String,
    pub process_info: Option<GpuProcess>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolicyInfo {
    pub policy_type: String,
    pub name: String,
    pub enabled: bool,
    pub limits: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{from_value, json};

    #[test]
    fn test_request_id_string() {
        let id: RequestId = from_value(json!("test-id-123")).unwrap();
        assert_eq!(id, RequestId::String("test-id-123".to_string()));
        assert_eq!(id.to_string(), "test-id-123");
    }

    #[test]
    fn test_request_id_number() {
        let id: RequestId = from_value(json!(123)).unwrap();
        assert_eq!(id, RequestId::Number(123));
        assert_eq!(id.to_string(), "123");
    }

    #[test]
    fn test_request_id_null() {
        let id: RequestId = from_value(json!(null)).unwrap();
        assert_eq!(id, RequestId::Null);
        assert_eq!(id.to_string(), "null");
    }

    #[test]
    fn test_jsonrpc_request_with_string_id() {
        let request = json!({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": "request-1"
        });

        let parsed: JsonRpcRequest = from_value(request).unwrap();
        assert_eq!(parsed.id, Some(RequestId::String("request-1".to_string())));
        assert_eq!(parsed.method, "tools/list");
    }

    #[test]
    fn test_jsonrpc_request_with_numeric_id() {
        // This was the bug - numeric IDs should be accepted per JSON-RPC 2.0
        let request = json!({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
            "id": 123
        });

        let parsed: JsonRpcRequest = from_value(request).unwrap();
        assert_eq!(parsed.id, Some(RequestId::Number(123)));
        assert_eq!(parsed.method, "initialize");
    }

    #[test]
    fn test_jsonrpc_request_rejects_wrong_version() {
        let request = json!({
            "jsonrpc": "1.0",
            "method": "initialize",
            "params": {},
            "id": 1
        });

        let parsed: Result<JsonRpcRequest, _> = from_value(request);
        assert!(parsed.is_err(), "expected jsonrpc version to be rejected");
    }

    #[test]
    fn test_jsonrpc_request_rejects_missing_version() {
        let request = json!({
            "method": "initialize",
            "params": {},
            "id": 1
        });

        let parsed: Result<JsonRpcRequest, _> = from_value(request);
        assert!(
            parsed.is_err(),
            "expected missing jsonrpc field to be rejected"
        );
    }

    #[test]
    fn test_jsonrpc_request_rejects_null_id() {
        let request = json!({
            "jsonrpc": "2.0",
            "method": "notify",
            "params": {},
            "id": null
        });

        let parsed: Result<JsonRpcRequest, _> = from_value(request);
        assert!(parsed.is_err(), "expected null id to be rejected");
    }

    #[test]
    fn test_jsonrpc_notification_without_id() {
        let request = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        });

        let parsed: JsonRpcRequest = from_value(request).unwrap();
        assert_eq!(parsed.id, None);
    }

    #[test]
    fn test_request_id_serialization_roundtrip() {
        // Test that IDs serialize back to the same type
        let string_id = RequestId::String("test".to_string());
        let number_id = RequestId::Number(42);
        let null_id = RequestId::Null;

        assert_eq!(serde_json::to_value(&string_id).unwrap(), json!("test"));
        assert_eq!(serde_json::to_value(&number_id).unwrap(), json!(42));
        assert_eq!(serde_json::to_value(&null_id).unwrap(), json!(null));
    }

    #[test]
    fn test_jsonrpc_response_preserves_id_type() {
        // Test that responses can use all ID types
        let response_with_number = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: RequestId::Number(123),
            result: Some(json!({"status": "ok"})),
            error: None,
        };

        let serialized = serde_json::to_value(&response_with_number).unwrap();
        assert_eq!(serialized["id"], json!(123));
    }

    #[test]
    fn test_initialize_response_uses_camel_case() {
        let response = InitializeResponse {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ServerCapabilities {
                resources: None,
                tools: None,
                logging: None,
            },
            server_info: ServerInfo {
                name: "test".to_string(),
                version: "1.0.0".to_string(),
            },
        };

        let serialized = serde_json::to_value(&response).unwrap();
        let object = serialized.as_object().unwrap();
        assert!(object.contains_key("protocolVersion"));
        assert!(object.contains_key("serverInfo"));
        assert!(!object.contains_key("protocol_version"));
        assert!(!object.contains_key("server_info"));
    }
}
