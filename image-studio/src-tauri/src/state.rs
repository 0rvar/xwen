use crate::{
    models::*,
    storage::{self, Result},
};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
#[derive(Default)]
struct Inner {
    config: Option<Config>,
    workspace: Option<Workspace>,
    warning: Option<String>,
    busy: bool,
}
pub struct Studio {
    pub client: reqwest::Client,
    pub(crate) batch_lock: Mutex<()>,
    inner: Arc<Mutex<Inner>>,
    config_path: PathBuf,
    args: Vec<String>,
}
struct Lease(Arc<Mutex<Inner>>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.lock().unwrap().busy = false
    }
}
pub fn normalize_url(value: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(value.trim()).map_err(|e| e.to_string())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("Server URL must be HTTP(S), without credentials, query or fragment".into());
    }
    let path = url.path().trim_end_matches('/');
    let path = path.strip_suffix("/v1").unwrap_or(path).to_owned();
    url.set_path(&path);
    Ok(url.to_string().trim_end_matches('/').into())
}
pub fn parse_args(args: &[String]) -> Result<Option<PathBuf>> {
    match args {
        [] => Ok(None),
        [path] if !path.starts_with('-') => Ok(Some(path.into())),
        [flag, path] if flag == "--workspace" && !path.is_empty() => Ok(Some(path.into())),
        _ => Err("Usage: xwen-image-studio [WORKSPACE | --workspace PATH]".into()),
    }
}
impl Studio {
    pub fn new(config_path: PathBuf, args: Vec<String>) -> Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(1800))
                .build()
                .map_err(|e| e.to_string())?,
            inner: Default::default(),
            batch_lock: Mutex::new(()),
            config_path,
            args,
        })
    }
    fn load_config(&self) -> Result<Config> {
        crate::logging::register_config_file(&self.config_path);
        match std::fs::read(&self.config_path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
                format!(
                    "Invalid config {}: {e}; repair this file before saving",
                    self.config_path.display()
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(e.to_string()),
        }
    }
    fn persist(&self, config: &Config) -> Result<()> {
        self.load_config()?;
        storage::atomic(
            &self.config_path,
            &serde_json::to_vec_pretty(config).map_err(|e| e.to_string())?,
            true,
        )
    }
    pub fn bootstrap(&self) -> Result<Bootstrap> {
        let mut inner = self.inner.lock().unwrap();
        if inner.config.is_none() {
            let mut config = self.load_config()?;
            if !config.server_url.is_empty() {
                config.server_url = normalize_url(&config.server_url)?;
            }
            let launch = parse_args(&self.args)?;
            let selected = launch
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .or_else(|| config.last_workspace.clone());
            if let Some(path) = selected {
                let result = if launch.is_some() {
                    make_workspace(Path::new(&path), true)
                } else {
                    make_workspace(Path::new(&path), false)
                };
                match result {
                    Ok(workspace) => {
                        remember(&mut config, &workspace.path);
                        if launch.is_some() {
                            self.persist(&config)?
                        }
                        inner.workspace = Some(workspace)
                    }
                    Err(e) if launch.is_none() => {
                        inner.warning = Some(format!("Last workspace unavailable: {e}"))
                    }
                    Err(e) => return Err(e),
                }
            }
            inner.config = Some(config)
        }
        Ok(Bootstrap {
            config: inner.config.clone().unwrap(),
            config_path: self.config_path.to_string_lossy().into(),
            workspace: inner.workspace.clone(),
            warning: inner.warning.clone(),
        })
    }
    pub fn save_config(&self, mut config: Config) -> Result<Config> {
        crate::logging::register_key(&config.api_key);
        let mut inner = self.inner.lock().unwrap();
        if inner.busy {
            return Err("Wait for the active render before changing settings".into());
        }
        config.server_url = normalize_url(&config.server_url)?;
        if config.api_key.contains(['\r', '\n']) {
            return Err("API key cannot contain line breaks".into());
        }
        if let Some(current) = &inner.config {
            for path in &current.workspaces {
                if !config.workspaces.contains(path) {
                    config.workspaces.push(path.clone())
                }
            }
        }
        if let Some(workspace) = &inner.workspace {
            remember(&mut config, &workspace.path)
        }
        self.persist(&config)?;
        inner.config = Some(config.clone());
        Ok(config)
    }
    pub fn select_workspace(&self, path: &str) -> Result<Workspace> {
        let mut inner = self.inner.lock().unwrap();
        if inner.busy {
            return Err("Wait for the active render before switching workspace".into());
        }
        let workspace = make_workspace(Path::new(path), true)?;
        let mut config = inner.config.clone().ok_or("Bootstrap first")?;
        remember(&mut config, &workspace.path);
        self.persist(&config)?;
        inner.config = Some(config);
        inner.workspace = Some(workspace.clone());
        inner.warning = None;
        Ok(workspace)
    }
    fn config(&self) -> Result<Config> {
        self.inner
            .lock()
            .unwrap()
            .config
            .clone()
            .ok_or("Bootstrap first".into())
    }
    pub fn list_images(&self) -> Result<Vec<SavedImage>> {
        let workspace = self.inner.lock().unwrap().workspace.clone();
        match workspace {
            Some(w) => storage::gallery(&w),
            None => Ok(vec![]),
        }
    }
    pub(crate) fn batch_scope<T>(
        &self,
        session_id: &str,
        f: impl FnOnce(&Workspace, &Config) -> Result<T>,
    ) -> Result<T> {
        let inner = self.inner.lock().unwrap();
        let workspace = inner.workspace.as_ref().ok_or("Select a workspace first")?;
        if workspace.session_id != session_id {
            return Err("Batch session no longer matches the selected workspace".into());
        }
        f(workspace, inner.config.as_ref().ok_or("Bootstrap first")?)
    }
    pub fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let inner = self.inner.lock().unwrap();
        match &inner.workspace {
            Some(w) => storage::sessions(w),
            None => Ok(vec![]),
        }
    }
    pub fn delete_image(
        &self,
        workspace_path: String,
        session_id: String,
        image_id: String,
    ) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let workspace = inner.workspace.as_ref().ok_or("Select a workspace first")?;
        if workspace.path != workspace_path {
            return Err("Selected workspace changed".into());
        }
        let session = storage::session_path(workspace, &session_id)?;
        let file = image_id
            .strip_prefix(&format!("{session_id}/"))
            .ok_or("Image does not belong to requested session")?;
        let (png, yaml) = storage::owned_output(&session, file)?;
        std::fs::remove_file(&png).map_err(|e| e.to_string())?;
        std::fs::remove_file(&yaml)
            .map_err(|e| format!("Image deleted but metadata removal failed: {e}"))?;
        Ok(())
    }
    pub fn delete_session(&self, workspace_path: String, session_id: String) -> Result<Workspace> {
        let mut inner = self.inner.lock().unwrap();
        if inner.busy {
            return Err("Wait for the active render before deleting sessions".into());
        }
        let workspace = inner.workspace.clone().ok_or("Select a workspace first")?;
        if workspace.path != workspace_path {
            return Err("Selected workspace changed".into());
        }
        let session = storage::session_path(&workspace, &session_id)?;
        let next = if workspace.session_id == session_id {
            make_workspace(Path::new(&workspace.path), false)?
        } else {
            workspace
        };
        if session.exists() {
            storage::session_image_count(&session)?;
            std::fs::remove_dir_all(&session).map_err(|e| e.to_string())?;
        }
        inner.workspace = Some(next.clone());
        Ok(next)
    }
    pub async fn generate_prompt(&self, idea: String) -> Result<String> {
        let value = self
            .request(
                &self.config()?,
                "/v1/chat/completions",
                Some(json!({
                    "model": "Qwen3.8-Flash-Next",
                    "stream": false,
                    "max_tokens": 512,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "messages": [
                        {
                            "role": "system",
                            "content": "Write one useful, vivid image-generation prompt based on the user's idea. Return only the prompt, with no preamble, quotes, explanation, or reasoning. If the idea is blank, invent a distinctive visual scene. Describe subject, composition, setting, lighting, and visual style in one concise paragraph."
                        },
                        {"role": "user", "content": idea}
                    ]
                })),
            )
            .await?;
        let choice = &value["choices"][0];
        if choice["finish_reason"] == "length" {
            return Err(
                "Prompt generation reached its token limit; try again with a simpler idea".into(),
            );
        }
        let content = choice["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .trim();
        if content.is_empty() {
            return Err("Server returned no prompt text".into());
        }
        Ok(content.to_owned())
    }
    async fn request(&self, config: &Config, endpoint: &str, body: Option<Value>) -> Result<Value> {
        let url = format!("{}{endpoint}", normalize_url(&config.server_url)?);
        let mut request = if let Some(body) = body {
            self.client.post(url).json(&body)
        } else {
            self.client.get(url).timeout(Duration::from_secs(15))
        };
        if !config.api_key.is_empty() {
            request = request.bearer_auth(&config.api_key)
        }
        let mut response = request
            .send()
            .await
            .map_err(|e| format!("Server request failed: {e}"))?;
        let status = response.status();
        let limit = if endpoint == "/v1/images/render" {
            256_000_000
        } else {
            32_000_000
        };
        if response.content_length().is_some_and(|n| n > limit) {
            return Err("Server response exceeds size limit".into());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
            if bytes.len() + chunk.len() > limit as usize {
                return Err("Server response exceeds size limit".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| format!("Server returned HTTP {status} without valid JSON"))?;
        if !status.is_success() {
            let message = value["error"]["message"]
                .as_str()
                .unwrap_or("Request rejected");
            let message = if config.api_key.is_empty() {
                message.into()
            } else {
                message.replace(&config.api_key, "[redacted]")
            };
            return Err(format!("HTTP {status}: {message}"));
        }
        Ok(value)
    }
    pub async fn check_server(&self, candidate: Option<Config>) -> Result<Value> {
        if let Some(config) = &candidate {
            crate::logging::register_key(&config.api_key);
        }
        let mut config = match candidate {
            Some(config) => config,
            None => self.config()?,
        };
        config.server_url = normalize_url(&config.server_url)?;
        if config.api_key.contains(['\r', '\n']) {
            return Err("API key cannot contain line breaks".into());
        }
        self.request(&config, "/health", None).await
    }
    pub async fn list_loras(&self) -> Result<Vec<LoraCandidate>> {
        let value = self
            .request(&self.config()?, "/v1/images/loras", None)
            .await?;
        serde_json::from_value(value["data"].clone()).map_err(|e| e.to_string())
    }
    pub async fn preprocess(&self, image: String, kind: String) -> Result<Preprocessed> {
        if !matches!(kind.as_str(), "canny" | "pose" | "depth") {
            return Err("Unknown preprocessor".into());
        }
        storage::decode_data(&image)?;
        let value = self
            .request(
                &self.config()?,
                "/v1/images/preprocess",
                Some(json!({"image":image,"type":kind})),
            )
            .await?;
        let bytes = storage::decode_data(
            value["b64_json"]
                .as_str()
                .ok_or("Missing preprocess image")?,
        )?;
        let (img, mime) = storage::decode(&bytes)?;
        Ok(Preprocessed {
            data_url: storage::data(&bytes, mime),
            width: img.width(),
            height: img.height(),
        })
    }
    pub async fn render(
        &self,
        session_id: String,
        request: RenderRequest,
        context: Value,
    ) -> Result<Vec<SavedImage>> {
        self.render_for_server(session_id, request, context, None)
            .await
    }
    pub(crate) async fn render_for_server(
        &self,
        session_id: String,
        request: RenderRequest,
        context: Value,
        expected_server_url: Option<String>,
    ) -> Result<Vec<SavedImage>> {
        validate(&request)?;
        if !context.is_object() {
            return Err("Render context must be an object".into());
        }
        let (workspace, config, _lease) = {
            let mut inner = self.inner.lock().unwrap();
            let workspace = inner.workspace.clone().ok_or("Select a workspace first")?;
            if workspace.session_id != session_id {
                return Err("Render session no longer matches the selected workspace".into());
            }
            if inner.busy {
                return Err("A render is already active".into());
            }
            let config = inner.config.clone().ok_or("Bootstrap first")?;
            if expected_server_url
                .as_deref()
                .is_some_and(|expected| expected != config.server_url)
            {
                return Err("Batch server differs from the configured server".into());
            }
            inner.busy = true;
            (workspace, config, Lease(self.inner.clone()))
        };
        let session = Path::new(&workspace.session_path);
        std::fs::create_dir_all(session).map_err(|e| e.to_string())?;
        if std::fs::symlink_metadata(session)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("Session must not be a symbolic link".into());
        }
        if session.canonicalize().map_err(|e| e.to_string())?.parent()
            != Some(Path::new(&workspace.path))
        {
            return Err("Session path escapes workspace".into());
        }
        let mut persisted = serde_json::to_value(&request).map_err(|e| e.to_string())?;
        let mut assets = json!({});
        for field in ["init_image", "mask"] {
            if let Some(source) = persisted[field].as_str() {
                let asset = storage::snapshot(session, source)?;
                persisted[field] = asset["path"].clone();
                assets[field] = asset;
            }
        }
        if let Some(source) = persisted["control"]["image"].as_str() {
            let asset = storage::snapshot(session, source)?;
            persisted["control"]["image"] = asset["path"].clone();
            assets["control.image"] = asset;
        }
        let probe = session.join(format!(".write-check-{}", storage::unique()));
        storage::atomic(&probe, b"", false)?;
        std::fs::remove_file(probe).map_err(|e| e.to_string())?;
        let started = Instant::now();
        let value = self
            .request(
                &config,
                "/v1/images/render",
                Some(serde_json::to_value(&request).map_err(|e| e.to_string())?),
            )
            .await?;
        let data = value["data"].as_array().ok_or("Missing render images")?;
        if data.len() != request.n as usize {
            return Err("Server image count differs from request".into());
        }
        let mut decoded = Vec::new();
        for (index, item) in data.iter().enumerate() {
            let bytes =
                storage::decode_data(item["b64_json"].as_str().ok_or("Missing render PNG")?)?;
            let (img, mime) = storage::decode(&bytes)?;
            if mime != "image/png" || img.width() != request.width || img.height() != request.height
            {
                return Err("Server output format or dimensions differ from request".into());
            }
            if item["seed"].as_u64() != Some(request.seed + index as u64)
                || item["start_step"]
                    .as_u64()
                    .filter(|n| *n <= request.steps as u64)
                    .is_none()
            {
                return Err("Server returned invalid seed or start_step".into());
            }
            decoded.push(bytes)
        }
        let mut saved = Vec::new();
        for (index, bytes) in decoded.iter().enumerate() {
            let item = &data[index];
            let id = format!(
                "{}-{}-{}",
                chrono::Utc::now().format("%H%M%S%3f"),
                request.seed + index as u64,
                storage::unique()
            );
            let filename = format!("{id}.png");
            let path = session.join(&filename);
            let metadata_path = session.join(format!("{id}.yaml"));
            let mut image_assets = assets.clone();
            if let Some(map) = item["control_map"].as_str() {
                image_assets["control_map"] = storage::snapshot(session, map)?;
            }
            let mut response = value.clone();
            response.as_object_mut().unwrap().remove("data");
            let mut image_response = item.clone();
            image_response
                .as_object_mut()
                .ok_or("Invalid image response")?
                .remove("b64_json");
            image_response
                .as_object_mut()
                .unwrap()
                .remove("control_map");
            response["image"] = image_response;
            let metadata = json!({"schema_version":1,"app_id":APP_ID,"app_version":env!("CARGO_PKG_VERSION"),"server":{"url":config.server_url,"endpoint":"/v1/images/render"},"created_at":chrono::Utc::now().to_rfc3339(),"duration_ms":started.elapsed().as_millis() as u64,"session_id":session_id,"request":persisted,"assets":image_assets,"context":redact(context.clone()),"output":{"file":filename,"sha256":storage::hash(bytes),"width":request.width,"height":request.height,"seed":item["seed"],"start_step":item["start_step"]},"response":redact(response)});
            storage::atomic(&path, bytes, false)?;
            if let Err(e) = storage::atomic(
                &metadata_path,
                serde_yaml_ng::to_string(&metadata)
                    .map_err(|e| e.to_string())?
                    .as_bytes(),
                false,
            ) {
                let cleanup = std::fs::remove_file(&path);
                return Err(format!(
                    "Metadata save failed: {e}; PNG cleanup: {cleanup:?}; {} previous outputs saved",
                    saved.len()
                ));
            }
            saved.push(SavedImage {
                id: format!("{session_id}/{filename}"),
                path: path.to_string_lossy().into(),
                metadata_path: metadata_path.to_string_lossy().into(),
                data_url: storage::thumbnail(bytes)?,
                seed: request.seed + index as u64,
                width: request.width,
                height: request.height,
                session_id: session_id.clone(),
                prompt: request.prompt.clone(),
                metadata,
            });
        }
        Ok(saved)
    }
}
fn remember(config: &mut Config, path: &str) {
    config.workspaces.retain(|p| p != path);
    config.workspaces.insert(0, path.into());
    config.last_workspace = Some(path.into())
}
fn make_workspace(path: &Path, create: bool) -> Result<Workspace> {
    if path.as_os_str().is_empty() {
        return Err("Workspace path is empty".into());
    }
    if create {
        std::fs::create_dir_all(path).map_err(|e| e.to_string())?
    }
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    if !path.is_dir() {
        return Err("Workspace is not a directory".into());
    }
    let session_id = format!(
        "{}-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        storage::unique()
    );
    Ok(Workspace {
        session_path: path.join(&session_id).to_string_lossy().into(),
        path: path.to_string_lossy().into(),
        session_id,
    })
}
pub(crate) fn redact(mut value: Value) -> Value {
    match &mut value {
        Value::Object(map) => {
            map.retain(|key, _| {
                !matches!(
                    key.to_ascii_lowercase().as_str(),
                    "api_key" | "apikey" | "authorization" | "password" | "token"
                )
            });
            for v in map.values_mut() {
                *v = redact(v.take())
            }
        }
        Value::Array(values) => {
            for v in values {
                *v = redact(v.take())
            }
        }
        _ => {}
    }
    value
}
pub fn validate(r: &RenderRequest) -> Result<()> {
    validate_fields(r)?;
    for image in [&r.init_image, &r.mask].into_iter().flatten() {
        storage::decode_data(image)?;
    }
    if let Some(control) = &r.control {
        storage::decode_data(&control.image)?;
    }
    Ok(())
}
pub(crate) fn validate_fields(r: &RenderRequest) -> Result<()> {
    if r.prompt.trim().is_empty() {
        return Err("Prompt is required".into());
    }
    if r.width == 0
        || r.height == 0
        || r.width > 8192
        || r.height > 8192
        || r.width % 16 != 0
        || r.height % 16 != 0
    {
        return Err("Dimensions must be positive multiples of 16".into());
    }
    let tokens = (r.width as u64 / 16) * (r.height as u64 / 16);
    if tokens % 32 != 0 {
        return Err("Image token count must be a multiple of 32".into());
    }
    if !(1..=50).contains(&r.steps)
        || !(1..=4).contains(&r.n)
        || r.seed > MAX_SAFE_INTEGER - (r.n.saturating_sub(1) as u64)
    {
        return Err("Invalid steps, image count or safe integer seed".into());
    }
    if r.init_image.is_none() && (r.mask.is_some() || r.strength.is_some()) {
        return Err("Mask and strength require a source image".into());
    }
    if r.mask.is_none() && r.mask_blur.is_some() {
        return Err("Mask blur requires a mask".into());
    }
    if r.strength
        .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        || r.mask_blur.is_some_and(|v| !v.is_finite() || v < 0.0)
    {
        return Err("Invalid strength or mask blur".into());
    }
    for lora in &r.loras {
        if lora.name.is_empty() || !lora.weight.is_finite() {
            return Err("Invalid LoRA".into());
        }
    }
    if let Some(c) = &r.control {
        if c.preprocess
            .as_deref()
            .is_some_and(|k| !matches!(k, "none" | "canny" | "pose" | "depth"))
        {
            return Err("Unknown control preprocessor".into());
        }
        let (scale, start, end) = (
            c.scale.unwrap_or(0.75),
            c.start.unwrap_or(0.0),
            c.end.unwrap_or(0.8),
        );
        if [scale, start, end]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || start > end
        {
            return Err("Invalid control scale or window".into());
        }
    }
    Ok(())
}
