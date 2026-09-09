use crate::{
    logging,
    models::{APP_ID, RenderRequest, SavedImage},
    state::{self, Studio},
    storage::{self, Result},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
const MAX_BYTES: usize = 16 * 1024 * 1024;
// An 8192-byte error needs at most sixfold YAML escaping; four output references
// and completion timestamps remain well below the remaining 16 KiB.
const TERMINAL_RESERVE: usize = 64 * 1024;
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Single,
    Matrix,
    Paired,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Axis {
    parameter: String,
    values: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    mode: Mode,
    axes: Vec<Axis>,
    count: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    request: RenderRequest,
    context: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Draft {
    definition: Definition,
    inputs: BTreeMap<String, String>,
    jobs: Vec<Job>,
}
#[derive(Serialize)]
pub struct BatchRef {
    pub id: String,
    pub manifest_path: String,
    pub job_ids: Vec<String>,
}
fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}
pub fn valid_id(id: &str) -> bool {
    id.strip_prefix("batch-").is_some_and(|s| {
        s.len() == 24
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    })
}
fn refs(request: &mut RenderRequest, mut map: impl FnMut(&str) -> Result<String>) -> Result<()> {
    for image in [&mut request.init_image, &mut request.mask]
        .into_iter()
        .flatten()
    {
        *image = map(image)?;
    }
    if let Some(c) = &mut request.control {
        c.image = map(&c.image)?;
    }
    Ok(())
}
fn write(path: &Path, value: &Value, replace: bool) -> Result<()> {
    let bytes = serde_yaml_ng::to_string(value).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_BYTES {
        return Err("Batch manifest exceeds 16 MiB".into());
    }
    let running = value["state"]["jobs"]
        .as_object()
        .map(|jobs| {
            jobs.values()
                .filter(|job| job["status"] == "running")
                .count()
        })
        .unwrap_or(0);
    if bytes
        .len()
        .saturating_add(running.saturating_mul(TERMINAL_RESERVE))
        > MAX_BYTES
    {
        return Err("Batch manifest has insufficient space to record running job results; no new render was started".into());
    }
    storage::atomic(path, bytes.as_bytes(), replace)
}
fn directory(session: &Path, create: bool) -> Result<PathBuf> {
    let path = session.join("batches");
    if create {
        fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    }
    let info = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !info.is_dir()
        || info.file_type().is_symlink()
        || path.canonicalize().map_err(|e| e.to_string())?.parent()
            != Some(session.canonicalize().map_err(|e| e.to_string())?.as_path())
    {
        return Err("Invalid batch directory".into());
    }
    Ok(path)
}
fn manifest_path(session: &Path, id: &str) -> Result<PathBuf> {
    if !valid_id(id) {
        return Err("Invalid batch id".into());
    }
    Ok(directory(session, false)?.join(format!("{id}.yaml")))
}
pub fn load_owned(path: &Path, session_id: &str, id: &str) -> Result<Value> {
    if !valid_id(id)
        || path.file_name().and_then(|s| s.to_str()) != Some(format!("{id}.yaml").as_str())
    {
        return Err("Invalid batch filename".into());
    }
    let info = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !info.is_file() || info.file_type().is_symlink() || info.len() > MAX_BYTES as u64 {
        return Err("Invalid batch manifest file".into());
    }
    let m: Value = serde_yaml_ng::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if m["schema_version"] != 1
        || m["app_id"] != APP_ID
        || m["kind"] != "batch"
        || m["id"] != id
        || m["session_id"] != session_id
        || m["plan"]["jobs"]
            .as_array()
            .is_none_or(|j| j.is_empty() || j.len() > 1000)
    {
        return Err("Manifest is not owned by this batch session".into());
    }
    if !m["plan"]["assets"].is_object() || !m["state"]["jobs"].is_object() {
        return Err("Invalid batch plan or state".into());
    }
    for (index, job) in m["plan"]["jobs"].as_array().unwrap().iter().enumerate() {
        let expected = format!("job-{:06}", index + 1);
        if job["id"] != expected || !job["context"].is_object() {
            return Err("Invalid batch job".into());
        }
        let request: RenderRequest =
            serde_json::from_value(job["request"].clone()).map_err(|e| e.to_string())?;
        state::validate_fields(&request)?;
        let status = &m["state"]["jobs"][&expected];
        if !status.is_object()
            || !matches!(
                status["status"].as_str(),
                Some("pending" | "running" | "succeeded" | "failed" | "discarded")
            )
            || status["attempts"]
                .as_array()
                .is_none_or(|a| a.iter().any(|v| !v.is_object()))
            || !status["outputs"].is_array()
        {
            return Err("Invalid batch job state".into());
        }
    }
    Ok(m)
}
fn job_index(m: &Value, id: &str) -> Result<usize> {
    m["plan"]["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .position(|j| j["id"] == id)
        .ok_or("Unknown batch job id".into())
}
impl Studio {
    pub fn create_batch(&self, session_id: String, mut draft: Draft) -> Result<BatchRef> {
        let _guard = self.batch_lock.lock().unwrap();
        if draft.jobs.is_empty()
            || draft.jobs.len() > 1000
            || draft.definition.count == 0
            || draft.definition.count > 1000
            || draft.inputs.len() > 3000
        {
            return Err("Batch requires 1–1000 jobs and a valid repeat count".into());
        }
        if matches!(draft.definition.mode, Mode::Single) && !draft.definition.axes.is_empty() {
            return Err("Single batch cannot define axes".into());
        }
        for key in draft.inputs.keys() {
            if !key
                .strip_prefix("input_")
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            {
                return Err("Invalid batch input key".into());
            }
        }
        for job in &mut draft.jobs {
            state::validate_fields(&job.request)?;
            if !job.context.is_object() {
                return Err("Batch job context must be an object".into());
            }
            refs(&mut job.request, |key| {
                if draft.inputs.contains_key(key) {
                    Ok(key.into())
                } else {
                    Err("Batch job references an unknown shared input".into())
                }
            })?;
        }
        self.batch_scope(&session_id, |workspace, config| {
            let session = storage::session_path(workspace, &session_id)?;
            fs::create_dir_all(&session).map_err(|error| error.to_string())?;
            let dir = directory(&session, true)?;
            let id = format!("batch-{}", storage::unique());
            let path = dir.join(format!("{id}.yaml"));
            let mut assets = BTreeMap::new();
            for (key, data) in &draft.inputs {
                assets.insert(key.clone(), storage::snapshot(&session, data)?);
            }
            let mut jobs = Vec::new();
            let mut states = serde_json::Map::new();
            let mut job_ids = Vec::new();
            for (index, mut job) in draft.jobs.into_iter().enumerate() {
                let job_id = format!("job-{:06}", index + 1);
                refs(&mut job.request, |key| {
                    Ok(assets[key]["path"].as_str().unwrap().into())
                })?;
                jobs.push(json!({
                    "id": job_id,
                    "request": job.request,
                    "context": state::redact(job.context)
                }));
                states.insert(
                    job_id.clone(),
                    json!({
                        "status": "pending", "attempts": [], "outputs": []
                    }),
                );
                job_ids.push(job_id);
            }
            let time = now();
            let manifest = json!({
                "schema_version": 1, "app_id": APP_ID, "kind": "batch", "id": id,
                "session_id": session_id, "created_at": time,
                "server": {"url": config.server_url},
                "plan": {"definition": draft.definition, "jobs": jobs, "assets": assets},
                "state": {"updated_at": time, "jobs": states}
            });
            write(&path, &manifest, false)?;
            Ok(BatchRef {
                id,
                manifest_path: path.to_string_lossy().into(),
                job_ids,
            })
        })
    }
    pub fn discard_batch_jobs(
        &self,
        session_id: String,
        batch_id: String,
        job_ids: Vec<String>,
    ) -> Result<()> {
        let _guard = self.batch_lock.lock().unwrap();
        if job_ids.len() > 1000 {
            return Err("Too many discard jobs".into());
        }
        self.batch_scope(&session_id, |workspace, _| {
            let session = storage::session_path(workspace, &session_id)?;
            let path = manifest_path(&session, &batch_id)?;
            let mut m = load_owned(&path, &session_id, &batch_id)?;
            for id in &job_ids {
                job_index(&m, id)?;
                if !matches!(
                    m["state"]["jobs"][id]["status"].as_str(),
                    Some("pending" | "failed")
                ) {
                    return Err("Only pending or failed batch jobs may be discarded".into());
                }
            }
            for id in &job_ids {
                m["state"]["jobs"][id]["status"] = json!("discarded");
            }
            m["state"]["updated_at"] = json!(now());
            write(&path, &m, true)
        })
    }
    pub async fn render_batch_job(
        &self,
        session_id: String,
        batch_id: String,
        job_id: String,
    ) -> Result<Vec<SavedImage>> {
        let (workspace, path, request, context, attempt, server_url) = {
            let _guard = self.batch_lock.lock().unwrap();
            self.batch_scope(&session_id, |workspace, config| {
                let session = storage::session_path(workspace, &session_id)?;
                let path = manifest_path(&session, &batch_id)?;
                let mut m = load_owned(&path, &session_id, &batch_id)?;
                let index = job_index(&m, &job_id)?;
                if m["server"]["url"] != config.server_url {
                    return Err("Batch server differs from the configured server".into());
                }
                if !matches!(
                    m["state"]["jobs"][&job_id]["status"].as_str(),
                    Some("pending" | "failed")
                ) {
                    return Err("Batch job is already running, succeeded or discarded".into());
                }
                let mut request: RenderRequest =
                    serde_json::from_value(m["plan"]["jobs"][index]["request"].clone())
                        .map_err(|e| e.to_string())?;
                let assets = m["plan"]["assets"]
                    .as_object()
                    .ok_or("Missing batch assets")?;
                let mut hydrated: BTreeMap<String, String> = BTreeMap::new();
                refs(&mut request, |relative| {
                    if let Some(value) = hydrated.get(relative) {
                        return Ok(value.clone());
                    }
                    let asset = assets
                        .values()
                        .find(|a| a["path"] == relative)
                        .ok_or("Batch input missing from assets")?;
                    if !relative.starts_with("inputs/") {
                        return Err("Invalid batch input path".into());
                    }
                    let inputs_info =
                        fs::symlink_metadata(session.join("inputs")).map_err(|e| e.to_string())?;
                    if !inputs_info.is_dir() || inputs_info.file_type().is_symlink() {
                        return Err("Batch inputs directory must be a real directory".into());
                    }
                    if Path::new(relative).components().count() != 2 {
                        return Err("Invalid batch input path".into());
                    }
                    let input = storage::confined(&session, relative)?;
                    let input_info =
                        fs::symlink_metadata(session.join(relative)).map_err(|e| e.to_string())?;
                    if !input_info.is_file() || input_info.file_type().is_symlink() {
                        return Err("Batch input must be a regular file".into());
                    }
                    if input_info.len() > 100_000_000 {
                        return Err("Batch input too large".into());
                    }
                    let bytes = fs::read(&input).map_err(|e| e.to_string())?;
                    if asset["sha256"] != storage::hash(&bytes) {
                        return Err("Batch input hash mismatch".into());
                    }
                    let (_, mime) = storage::decode(&bytes)?;
                    let value = storage::data(&bytes, mime);
                    hydrated.insert(relative.to_owned(), value.clone());
                    Ok(value)
                })?;
                let attempts = m["state"]["jobs"][&job_id]["attempts"]
                    .as_array_mut()
                    .ok_or("Invalid attempts")?;
                let attempt = attempts.len() + 1;
                attempts.push(json!({"number":attempt,"started_at":now(),"status":"running"}));
                m["state"]["jobs"][&job_id]["status"] = json!("running");
                m["state"]["updated_at"] = json!(now());
                let mut context = m["plan"]["jobs"][index]["context"].clone();
                if !context.is_object() {
                    return Err("Invalid batch context".into());
                }
                context["batch_id"] = json!(batch_id);
                context["job_id"] = json!(job_id);
                context["attempt"] = json!(attempt);
                context["batch_manifest"] = json!(format!("batches/{batch_id}.yaml"));
                write(&path, &m, true)?;
                Ok((
                    workspace.clone(),
                    path,
                    request,
                    context,
                    attempt,
                    config.server_url.clone(),
                ))
            })?
        };
        let outcome = self
            .render_for_server(session_id.clone(), request, context, Some(server_url))
            .await;
        let update = (|| -> Result<()> {
            let _guard = self.batch_lock.lock().unwrap();
            self.batch_scope(&session_id, |_, _| {
            let session = storage::session_path(&workspace, &session_id)?;
            let current = manifest_path(&session, &batch_id)?;
            if current != path {
                return Err("Batch path changed".into());
            }
            let mut m = load_owned(&path, &session_id, &batch_id)?;
            let job = &mut m["state"]["jobs"][&job_id];
            if job["status"] != "running"
                || job["attempts"].as_array().map(Vec::len) != Some(attempt)
            {
                return Err("Batch attempt state changed".into());
            }
            let status = if outcome.is_ok() {
                "succeeded"
            } else {
                "failed"
            };
            job["status"] = json!(status);
            job["attempts"][attempt - 1]["status"] = json!(status);
            job["attempts"][attempt - 1]["finished_at"] = json!(now());
            match &outcome {
                Ok(outputs) => {
                    job["outputs"] = json!(outputs.iter().map(|output| {
                        json!({
                            "image": Path::new(&output.path).file_name().unwrap().to_string_lossy(),
                            "metadata": Path::new(&output.metadata_path).file_name().unwrap().to_string_lossy(),
                            "seed": output.seed,
                            "sha256": output.metadata["output"]["sha256"]
                        })
                    }).collect::<Vec<_>>());
                }
                Err(error) => {
                    job["attempts"][attempt - 1]["error"] = json!(logging::sanitize(error, 8192))
                }
            };
            m["state"]["updated_at"] = json!(now());
            write(&path, &m, true)
            })
        })();
        if let Err(error) = update {
            return Err(format!(
                "Batch result could not be recorded: {error}. Render outcome: {}. Inspect image metadata batch_id/job_id for reconciliation.",
                if outcome.is_ok() {
                    "images were saved"
                } else {
                    "render failed"
                }
            ));
        }
        outcome
    }
}
