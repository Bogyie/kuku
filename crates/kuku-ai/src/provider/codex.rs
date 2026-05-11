use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt;
use http::StatusCode;
use serde::Deserialize;
use serde_json::{Value, json};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use crate::{
    AiError,
    provider::{CompletionBackend, CompletionEvent, CompletionTurnRequest, CompletionTurnStream},
    tools::ToolDescriptor,
    types::{ChatMessage, FinishReason, ModelToolCall, TokenUsage},
};

const OPENAI_RESPONSES_URL: &str = "https://api.openai.com/v1/responses";
const OPENAI_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_AUTH_MISSING: &str =
    "Codex OAuth token not found. Run `codex login` in a terminal, then try again.";
static CODEX_AUTH_WRITE_ID: AtomicU64 = AtomicU64::new(0);
static CODEX_REFRESH_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

pub struct CodexBackend {
    client: reqwest::Client,
    model: String,
}

impl CodexBackend {
    pub fn new(model: &str) -> Result<Self, AiError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|error| AiError::ProviderInit(error.to_string()))?;
        Ok(Self {
            client,
            model: model.trim().to_string(),
        })
    }
}

#[async_trait]
impl CompletionBackend for CodexBackend {
    async fn stream_turn(
        &self,
        request: CompletionTurnRequest,
    ) -> Result<CompletionTurnStream, AiError> {
        let request_model = request.model.trim();
        let model = if request_model.is_empty() {
            self.model.as_str()
        } else {
            request_model
        };
        if model.is_empty() {
            return Err(AiError::InvalidArguments(
                "Codex model is required".to_string(),
            ));
        }
        let body = responses_request_body(model, &request)?;

        let token = read_codex_access_token()?;
        let mut response = self.send_responses_request(&token.value, &body).await?;
        if is_auth_error(response.status())
            && let CodexTokenSource::OAuth { auth_path } = token.source
            && let Some(refreshed) =
                refresh_codex_access_token(&self.client, &auth_path, &token.value).await?
        {
            response = self.send_responses_request(&refreshed, &body).await?;
        }

        let status = response.status();
        if is_auth_error(status) {
            return Err(AiError::Unauthorized);
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AiError::ProviderError(format!(
                "OpenAI Responses request failed ({status}): {}",
                error_message_from_body(&body)
            )));
        }

        let stream = try_stream! {
            let mut bytes = response.bytes_stream();
            let mut buffer = Vec::new();
            let mut saw_tool_calls = false;
            let mut saw_finished = false;

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|error| AiError::ProviderError(error.to_string()))?;
                buffer.extend_from_slice(&chunk);

                while let Some((boundary, delimiter_len)) = sse_event_boundary(&buffer) {
                    let raw_event = String::from_utf8(buffer[..boundary].to_vec())
                        .map_err(|error| AiError::ProviderError(format!("Invalid OpenAI stream event: {error}")))?;
                    buffer.drain(..boundary + delimiter_len);
                    if raw_event.trim().is_empty() {
                        continue;
                    }

                    match completion_event_from_sse(&raw_event)? {
                        ParsedCodexEvent::None => {}
                        ParsedCodexEvent::TextDelta(delta) => {
                            yield CompletionEvent::TextDelta(delta);
                        }
                        ParsedCodexEvent::ToolCall(call) => {
                            saw_tool_calls = true;
                            yield CompletionEvent::ToolCalls(vec![call]);
                        }
                        ParsedCodexEvent::Finished { usage } => {
                            saw_finished = true;
                            yield CompletionEvent::Finished {
                                finish_reason: if saw_tool_calls {
                                    FinishReason::ToolCalls
                                } else {
                                    FinishReason::Stop
                                },
                                usage,
                            };
                        }
                    }
                }
            }

            if !buffer.is_empty() {
                Err(AiError::ProviderError("OpenAI stream ended with a partial event".to_string()))?;
            }

            ensure_completed_event(saw_finished)?;
        };

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<String>, AiError> {
        Ok(vec![self.model.clone()])
    }
}

impl CodexBackend {
    async fn send_responses_request(
        &self,
        token: &str,
        body: &Value,
    ) -> Result<reqwest::Response, AiError> {
        self.client
            .post(OPENAI_RESPONSES_URL)
            .bearer_auth(token)
            .header("accept", "text/event-stream")
            .json(body)
            .send()
            .await
            .map_err(|error| AiError::ProviderError(error.to_string()))
    }
}

fn responses_request_body(model: &str, request: &CompletionTurnRequest) -> Result<Value, AiError> {
    let mut body = json!({
        "model": model,
        "input": response_input_items(&request.messages)?,
        "stream": true,
        "store": false,
    });
    if let Some(system_prompt) = request
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
    {
        body["instructions"] = Value::String(system_prompt.to_string());
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(response_tool_from).collect());
    }
    Ok(body)
}

fn response_tool_from(tool: &ToolDescriptor) -> Value {
    // Keep the app's existing permissive tool schemas intact. Rig's OpenAI
    // adapter forces strict schemas, which would turn our optional tool
    // arguments into required ones.
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": false,
    })
}

fn response_input_items(messages: &[ChatMessage]) -> Result<Vec<Value>, AiError> {
    let mut items = Vec::new();
    for message in messages {
        match message {
            ChatMessage::System { .. } => {
                return Err(AiError::InvalidArguments(
                    "System messages should be passed via the system prompt".to_string(),
                ));
            }
            ChatMessage::User { content, .. } => {
                items.push(message_item("user", "input_text", content));
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
            } => {
                if !content.is_empty() {
                    items.push(message_item("assistant", "output_text", content));
                }
                for call in tool_calls {
                    items.push(json!({
                        "type": "function_call",
                        "id": call.tool_call_id.clone().unwrap_or_else(|| call.call_id.clone()),
                        "call_id": provider_call_id(call),
                        "name": call.tool_name,
                        "arguments": call.arguments.to_string(),
                    }));
                }
            }
            ChatMessage::ToolResult {
                call_id,
                output,
                tool_call_id,
                provider_call_id,
                ..
            } => {
                items.push(json!({
                    "type": "function_call_output",
                    "call_id": provider_call_id
                        .clone()
                        .or_else(|| tool_call_id.clone())
                        .unwrap_or_else(|| call_id.clone()),
                    "output": output,
                }));
            }
        }
    }

    if items.is_empty() {
        return Err(AiError::InvalidArguments(
            "Completion request requires at least one message".to_string(),
        ));
    }
    Ok(items)
}

fn message_item(role: &str, content_type: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": [
            {
                "type": content_type,
                "text": text,
            }
        ],
    })
}

fn provider_call_id(call: &ModelToolCall) -> String {
    call.provider_call_id
        .clone()
        .or_else(|| call.tool_call_id.clone())
        .unwrap_or_else(|| call.call_id.clone())
}

enum ParsedCodexEvent {
    None,
    TextDelta(String),
    ToolCall(ModelToolCall),
    Finished { usage: Option<TokenUsage> },
}

fn completion_event_from_sse(raw_event: &str) -> Result<ParsedCodexEvent, AiError> {
    let Some(data) = sse_data(raw_event) else {
        return Ok(ParsedCodexEvent::None);
    };
    if data == "[DONE]" {
        return Ok(ParsedCodexEvent::None);
    }

    let value: Value = serde_json::from_str(&data)
        .map_err(|error| AiError::ProviderError(format!("Invalid OpenAI stream event: {error}")))?;
    match value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "response.output_text.delta" => Ok(value
            .get("delta")
            .and_then(Value::as_str)
            .filter(|delta| !delta.is_empty())
            .map(|delta| ParsedCodexEvent::TextDelta(delta.to_string()))
            .unwrap_or(ParsedCodexEvent::None)),
        "response.output_item.done" => {
            let Some(item) = value.get("item") else {
                return Ok(ParsedCodexEvent::None);
            };
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                return Ok(ParsedCodexEvent::None);
            }
            Ok(ParsedCodexEvent::ToolCall(model_tool_call_from_item(item)?))
        }
        "response.completed" => Ok(ParsedCodexEvent::Finished {
            usage: value
                .get("response")
                .and_then(|response| response.get("usage"))
                .map(token_usage_from)
                .transpose()?,
        }),
        "response.failed" | "response.incomplete" => Err(AiError::ProviderError(
            response_error_message(&value).unwrap_or_else(|| "OpenAI response failed".to_string()),
        )),
        "error" => Err(AiError::ProviderError(
            response_error_message(&value).unwrap_or_else(|| "OpenAI stream error".to_string()),
        )),
        _ => Ok(ParsedCodexEvent::None),
    }
}

fn ensure_completed_event(saw_finished: bool) -> Result<(), AiError> {
    if saw_finished {
        Ok(())
    } else {
        Err(AiError::ProviderError(
            "OpenAI stream ended without response.completed".to_string(),
        ))
    }
}

fn model_tool_call_from_item(item: &Value) -> Result<ModelToolCall, AiError> {
    let tool_call_id = item
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| AiError::ProviderError("OpenAI function call is missing an id".to_string()))?
        .to_string();
    let provider_call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|call_id| !call_id.trim().is_empty())
        .ok_or_else(|| {
            AiError::ProviderError("OpenAI function call is missing a call_id".to_string())
        })?
        .to_string();
    let tool_name = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| {
            AiError::ProviderError("OpenAI function call is missing a name".to_string())
        })?
        .to_string();
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            serde_json::from_str(value).map_err(|error| {
                AiError::ProviderError(format!("Invalid OpenAI function call arguments: {error}"))
            })
        })
        .transpose()?
        .unwrap_or(Value::Null);

    Ok(ModelToolCall {
        call_id: tool_call_id.clone(),
        tool_name,
        arguments,
        signature: None,
        tool_call_id: Some(tool_call_id),
        provider_call_id: Some(provider_call_id),
    })
}

fn token_usage_from(value: &Value) -> Result<TokenUsage, AiError> {
    Ok(TokenUsage {
        input_tokens: u64_field(value, "input_tokens"),
        output_tokens: u64_field(value, "output_tokens"),
        total_tokens: u64_field(value, "total_tokens"),
        cached_input_tokens: value
            .get("input_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .or_else(|| value.get("cached_input_tokens").and_then(Value::as_u64))
            .unwrap_or_default(),
    })
}

fn u64_field(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or_default()
}

fn response_error_message(value: &Value) -> Option<String> {
    value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
        })
        .map(str::to_string)
}

fn is_auth_error(status: StatusCode) -> bool {
    status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN
}

fn error_message_from_body(body: &str) -> String {
    let body = body.trim();
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| response_error_message(&value))
        .unwrap_or_else(|| {
            if body.is_empty() {
                "empty response body".to_string()
            } else {
                body.to_string()
            }
        })
}

fn sse_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = find_bytes(buffer, b"\n\n").map(|index| (index, 2));
    let crlf = find_bytes(buffer, b"\r\n\r\n").map(|index| (index, 4));
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn sse_data(raw_event: &str) -> Option<String> {
    let lines = raw_event
        .lines()
        .filter_map(|line| {
            line.strip_prefix("data:")
                .map(str::trim_start)
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

#[derive(Deserialize)]
struct CodexAuthFile {
    #[serde(rename = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,
    tokens: Option<CodexTokens>,
}

#[derive(Deserialize)]
struct CodexTokens {
    access_token: Option<String>,
    refresh_token: Option<String>,
}

struct CodexAccessToken {
    value: String,
    source: CodexTokenSource,
}

enum CodexTokenSource {
    OAuth { auth_path: PathBuf },
    ApiKey,
}

fn read_codex_access_token() -> Result<CodexAccessToken, AiError> {
    let path = codex_auth_path()?;
    let auth = read_codex_auth_file(&path)?;
    if let Some(token) = auth
        .tokens
        .and_then(|tokens| tokens.access_token)
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
    {
        return Ok(CodexAccessToken {
            value: token,
            source: CodexTokenSource::OAuth { auth_path: path },
        });
    }

    auth.openai_api_key
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .map(|token| CodexAccessToken {
            value: token,
            source: CodexTokenSource::ApiKey,
        })
        .ok_or_else(|| AiError::ProviderInit(CODEX_AUTH_MISSING.to_string()))
}

fn read_codex_auth_file(path: &Path) -> Result<CodexAuthFile, AiError> {
    let content = read_codex_auth_content(path)?;
    serde_json::from_str(&content)
        .map_err(|error| AiError::ProviderInit(format!("Invalid Codex auth JSON: {error}")))
}

fn read_codex_auth_content(path: &Path) -> Result<String, AiError> {
    std::fs::read_to_string(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => AiError::ProviderInit(CODEX_AUTH_MISSING.to_string()),
        _ => AiError::Io(format!("Failed to read {}: {error}", path.display())),
    })
}

fn codex_auth_path() -> Result<PathBuf, AiError> {
    let home = dirs::home_dir()
        .ok_or_else(|| AiError::State("Cannot resolve the user home directory".to_string()))?;
    Ok(home.join(".codex").join("auth.json"))
}

#[derive(Deserialize)]
struct CodexRefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
}

async fn refresh_codex_access_token(
    client: &reqwest::Client,
    auth_path: &Path,
    stale_token: &str,
) -> Result<Option<String>, AiError> {
    let _refresh_guard = CODEX_REFRESH_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    let auth = read_codex_auth_file(auth_path)?;
    let current_token = auth
        .tokens
        .as_ref()
        .and_then(|tokens| tokens.access_token.as_deref())
        .map(str::trim)
        .filter(|token| !token.is_empty());
    if let Some(current_token) = current_token.filter(|token| *token != stale_token) {
        return Ok(Some(current_token.to_string()));
    }

    let Some(refresh_token) = auth
        .tokens
        .and_then(|tokens| tokens.refresh_token)
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
    else {
        return Ok(None);
    };

    let response = client
        .post(OPENAI_OAUTH_TOKEN_URL)
        .header("content-type", "application/x-www-form-urlencoded")
        .body(refresh_token_form_body(&refresh_token))
        .send()
        .await
        .map_err(|error| AiError::ProviderError(error.to_string()))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(AiError::Unauthorized);
    }

    let refreshed: CodexRefreshResponse = serde_json::from_str(&body).map_err(|error| {
        AiError::ProviderError(format!("Invalid Codex refresh response: {error}"))
    })?;
    let access_token = refreshed.access_token.trim().to_string();
    if access_token.is_empty() {
        return Err(AiError::ProviderError(
            "Codex token refresh returned an empty access token".to_string(),
        ));
    }

    persist_refreshed_codex_tokens(
        auth_path,
        &access_token,
        refreshed.refresh_token,
        refreshed.id_token,
    )?;
    Ok(Some(access_token))
}

fn persist_refreshed_codex_tokens(
    auth_path: &Path,
    access_token: &str,
    refresh_token: Option<String>,
    id_token: Option<String>,
) -> Result<(), AiError> {
    let content = read_codex_auth_content(auth_path)?;
    let mut value: Value = serde_json::from_str(&content)
        .map_err(|error| AiError::ProviderInit(format!("Invalid Codex auth JSON: {error}")))?;
    let Some(tokens) = value.get_mut("tokens").and_then(Value::as_object_mut) else {
        return Err(AiError::ProviderInit(CODEX_AUTH_MISSING.to_string()));
    };

    tokens.insert(
        "access_token".to_string(),
        Value::String(access_token.to_string()),
    );
    if let Some(refresh_token) = refresh_token
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
    {
        tokens.insert("refresh_token".to_string(), Value::String(refresh_token));
    }
    if let Some(id_token) = id_token
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
    {
        tokens.insert("id_token".to_string(), Value::String(id_token));
    }

    let next = serde_json::to_string_pretty(&value)
        .map_err(|error| AiError::ProviderError(error.to_string()))?;
    write_codex_auth_file(auth_path, &format!("{next}\n"))
}

fn write_codex_auth_file(auth_path: &Path, content: &str) -> Result<(), AiError> {
    let temp_path = auth_path.with_extension(format!(
        "json.{}.{}.tmp",
        std::process::id(),
        CODEX_AUTH_WRITE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    write_codex_auth_temp_file(&temp_path, content)?;
    fs::rename(&temp_path, auth_path).map_err(|error| {
        let _ = fs::remove_file(&temp_path);
        AiError::Io(format!(
            "Failed to replace {}: {error}",
            auth_path.display()
        ))
    })
}

fn write_codex_auth_temp_file(temp_path: &Path, content: &str) -> Result<(), AiError> {
    let result = write_codex_auth_temp_file_inner(temp_path, content);
    if let Err(error) = result {
        let _ = fs::remove_file(temp_path);
        return Err(AiError::Io(format!(
            "Failed to write {}: {error}",
            temp_path.display()
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn write_codex_auth_temp_file_inner(temp_path: &Path, content: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(temp_path)?;
    file.write_all(content.as_bytes())?;
    file.sync_all()
}

#[cfg(not(unix))]
fn write_codex_auth_temp_file_inner(temp_path: &Path, content: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp_path)?;
    file.write_all(content.as_bytes())?;
    file.sync_all()
}

fn refresh_token_form_body(refresh_token: &str) -> String {
    format!(
        "grant_type=refresh_token&client_id={}&refresh_token={}",
        form_encode(CODEX_OAUTH_CLIENT_ID),
        form_encode(refresh_token)
    )
}

fn form_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push('+'),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn response_input_preserves_function_call_ids_for_tool_results() {
        let messages = vec![
            ChatMessage::Assistant {
                content: String::new(),
                tool_calls: vec![ModelToolCall {
                    call_id: "internal-call".into(),
                    tool_name: "read_file".into(),
                    arguments: json!({ "path": "note.md" }),
                    signature: None,
                    tool_call_id: Some("fc_item".into()),
                    provider_call_id: Some("call_provider".into()),
                }],
            },
            ChatMessage::ToolResult {
                call_id: "internal-call".into(),
                tool_name: "read_file".into(),
                output: "contents".into(),
                is_error: false,
                tool_call_id: Some("fc_item".into()),
                provider_call_id: Some("call_provider".into()),
            },
        ];

        let items = response_input_items(&messages).expect("messages should convert");

        assert_eq!(items[0]["type"], "function_call");
        assert_eq!(items[0]["call_id"], "call_provider");
        assert_eq!(items[1]["type"], "function_call_output");
        assert_eq!(items[1]["call_id"], "call_provider");
    }

    #[test]
    fn response_input_rejects_system_messages() {
        let messages = vec![ChatMessage::System {
            content: "Use the system prompt instead".into(),
        }];

        let result = response_input_items(&messages);

        assert!(matches!(result, Err(AiError::InvalidArguments(_))));
    }

    #[test]
    fn response_tool_keeps_optional_schema_permissive() {
        let tool = ToolDescriptor {
            tool_id: "builtin.search_vault".into(),
            name: "search_vault".into(),
            description: "Search indexed markdown content".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_results": { "type": "integer" }
                },
                "required": ["query"]
            }),
            category: "search".into(),
            access: crate::tools::ToolAccess::ReadOnly,
            source: crate::tools::ToolSource::Native,
        };

        let converted = response_tool_from(&tool);

        assert_eq!(converted["strict"], false);
        assert_eq!(converted["parameters"]["required"], json!(["query"]));
    }

    #[test]
    fn parses_output_text_delta_sse_event() {
        let event = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n";

        match completion_event_from_sse(event).expect("event should parse") {
            ParsedCodexEvent::TextDelta(delta) => assert_eq!(delta, "hello"),
            _ => panic!("expected text delta"),
        }
    }

    #[test]
    fn ignores_done_marker_and_waits_for_completed_event() {
        match completion_event_from_sse("data: [DONE]\n").expect("event should parse") {
            ParsedCodexEvent::None => {}
            _ => panic!("expected done marker to be ignored"),
        }
    }

    #[test]
    fn parses_function_call_done_sse_event() {
        let event = concat!(
            "data: {",
            "\"type\":\"response.output_item.done\",",
            "\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",",
            "\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"note.md\\\"}\"}}",
            "\n"
        );

        match completion_event_from_sse(event).expect("event should parse") {
            ParsedCodexEvent::ToolCall(call) => {
                assert_eq!(call.call_id, "fc_1");
                assert_eq!(call.tool_call_id.as_deref(), Some("fc_1"));
                assert_eq!(call.provider_call_id.as_deref(), Some("call_1"));
                assert_eq!(call.tool_name, "read_file");
                assert_eq!(call.arguments, json!({ "path": "note.md" }));
            }
            _ => panic!("expected tool call"),
        }
    }

    #[test]
    fn rejects_function_call_without_provider_id() {
        let event = concat!(
            "data: {",
            "\"type\":\"response.output_item.done\",",
            "\"item\":{\"type\":\"function_call\",\"name\":\"read_file\",",
            "\"arguments\":\"{\\\"path\\\":\\\"note.md\\\"}\"}}",
            "\n"
        );

        let error = match completion_event_from_sse(event) {
            Ok(_) => panic!("missing call id should fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("missing an id"));
    }

    #[test]
    fn rejects_function_call_without_provider_call_id() {
        let event = concat!(
            "data: {",
            "\"type\":\"response.output_item.done\",",
            "\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"name\":\"read_file\",",
            "\"arguments\":\"{\\\"path\\\":\\\"note.md\\\"}\"}}",
            "\n"
        );

        let error = match completion_event_from_sse(event) {
            Ok(_) => panic!("missing provider call id should fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("missing a call_id"));
    }

    #[test]
    fn parses_completed_usage() {
        let event = concat!(
            "data: {",
            "\"type\":\"response.completed\",",
            "\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":4,",
            "\"total_tokens\":14,\"input_tokens_details\":{\"cached_tokens\":3}}}}",
            "\n"
        );

        match completion_event_from_sse(event).expect("event should parse") {
            ParsedCodexEvent::Finished { usage } => {
                let usage = usage.expect("usage should exist");
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 4);
                assert_eq!(usage.total_tokens, 14);
                assert_eq!(usage.cached_input_tokens, 3);
            }
            _ => panic!("expected finished"),
        }
    }

    #[test]
    fn missing_completed_event_is_rejected() {
        let error = ensure_completed_event(false).expect_err("missing terminal event should fail");

        assert!(error.to_string().contains("response.completed"));
    }

    #[test]
    fn empty_error_body_has_readable_message() {
        assert_eq!(error_message_from_body(" \n"), "empty response body");
    }

    #[test]
    fn finds_sse_boundaries_in_byte_buffer() {
        let mut buffer = "data: {\"delta\":\"안녕\"}".as_bytes().to_vec();
        assert_eq!(sse_event_boundary(&buffer), None);

        buffer.extend_from_slice(b"\n\n");
        assert_eq!(
            sse_event_boundary(&buffer),
            Some(("data: {\"delta\":\"안녕\"}".len(), 2))
        );
    }

    #[cfg(unix)]
    #[test]
    fn writes_codex_auth_file_with_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "kuku-codex-auth-test-{}-{}",
            std::process::id(),
            CODEX_AUTH_WRITE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).expect("test temp dir should be created");
        let auth_path = dir.join("auth.json");

        write_codex_auth_file(&auth_path, "{\"tokens\":{}}\n").expect("auth file should write");

        let mode = fs::metadata(&auth_path)
            .expect("auth file should exist")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);

        fs::remove_file(&auth_path).expect("test auth file should be removed");
        fs::remove_dir(&dir).expect("test temp dir should be removed");
    }

    #[test]
    fn refresh_token_form_body_is_url_encoded() {
        let body = refresh_token_form_body("rt value+/=");

        assert!(body.contains("grant_type=refresh_token"));
        assert!(body.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
        assert!(body.contains("refresh_token=rt+value%2B%2F%3D"));
    }
}
