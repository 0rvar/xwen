use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
pub const APP_ID: &str = "com.xwen.image-studio";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server_url: String,
    pub api_key: String,
    pub workspaces: Vec<String>,
    pub last_workspace: Option<String>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            server_url: String::new(),
            api_key: String::new(),
            workspaces: Vec::new(),
            last_workspace: None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Workspace {
    pub path: String,
    pub session_id: String,
    pub session_path: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct Bootstrap {
    pub config: Config,
    pub config_path: String,
    pub workspace: Option<Workspace>,
    pub warning: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct InputImage {
    pub name: String,
    pub data_url: String,
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraSpec {
    pub name: String,
    pub weight: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Control {
    pub image: String,
    pub preprocess: Option<String>,
    pub scale: Option<f64>,
    pub start: Option<f64>,
    pub end: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: u64,
    pub n: u32,
    #[serde(default)]
    pub loras: Vec<LoraSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init_image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_blur: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control: Option<Control>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoraCandidate {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct SavedImage {
    pub id: String,
    pub path: String,
    pub metadata_path: String,
    pub data_url: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub session_id: String,
    pub prompt: String,
    pub metadata: Value,
}
#[derive(Clone, Debug, Serialize)]
pub struct Preprocessed {
    pub data_url: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub image_count: usize,
}
