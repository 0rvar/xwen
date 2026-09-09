mod batch;
mod logging;
mod models;
mod state;
mod storage;
use models::*;
use state::Studio;
use tauri::{Manager, State};
#[tauri::command]
fn bootstrap(state: State<'_, Studio>) -> Result<Bootstrap, String> {
    logging::result("bootstrap", state.bootstrap(), true)
}
#[tauri::command]
fn save_config(state: State<'_, Studio>, config: Config) -> Result<Config, String> {
    logging::result("save_config", state.save_config(config), true)
}
#[tauri::command]
fn select_workspace(state: State<'_, Studio>, path: String) -> Result<Workspace, String> {
    logging::result("select_workspace", state.select_workspace(&path), true)
}
#[tauri::command(async)]
fn read_image(path: String) -> Result<InputImage, String> {
    logging::result(
        "read_image",
        storage::read_image(std::path::Path::new(&path)),
        false,
    )
}
#[tauri::command]
async fn list_loras(state: State<'_, Studio>) -> Result<Vec<LoraCandidate>, String> {
    logging::result("list_loras", state.list_loras().await, false)
}
#[tauri::command]
async fn check_server(
    state: State<'_, Studio>,
    config: Option<Config>,
) -> Result<serde_json::Value, String> {
    logging::result("check_server", state.check_server(config).await, false)
}
#[tauri::command]
async fn preprocess(
    state: State<'_, Studio>,
    image: String,
    kind: String,
) -> Result<Preprocessed, String> {
    logging::result("preprocess", state.preprocess(image, kind).await, true)
}
#[tauri::command]
async fn render(
    state: State<'_, Studio>,
    session_id: String,
    request: RenderRequest,
    context: serde_json::Value,
) -> Result<Vec<SavedImage>, String> {
    logging::result(
        "render",
        state.render(session_id, request, context).await,
        true,
    )
}
#[tauri::command(async)]
fn list_images(state: State<'_, Studio>) -> Result<Vec<SavedImage>, String> {
    logging::result("list_images", state.list_images(), false)
}
#[tauri::command(async)]
fn list_sessions(state: State<'_, Studio>) -> Result<Vec<SessionSummary>, String> {
    logging::result("list_sessions", state.list_sessions(), false)
}
#[tauri::command(async)]
fn delete_image(
    state: State<'_, Studio>,
    workspace_path: String,
    session_id: String,
    image_id: String,
) -> Result<(), String> {
    logging::result(
        "delete_image",
        state.delete_image(workspace_path, session_id, image_id),
        true,
    )
}
#[tauri::command(async)]
fn delete_session(
    state: State<'_, Studio>,
    workspace_path: String,
    session_id: String,
) -> Result<Workspace, String> {
    logging::result(
        "delete_session",
        state.delete_session(workspace_path, session_id),
        true,
    )
}
#[tauri::command]
async fn generate_prompt(state: State<'_, Studio>, idea: String) -> Result<String, String> {
    logging::result("generate_prompt", state.generate_prompt(idea).await, true)
}
#[tauri::command]
async fn chat(
    state: State<'_, Studio>,
    messages: serde_json::Value,
    tools: serde_json::Value,
) -> Result<serde_json::Value, String> {
    logging::result("chat", state.chat(messages, tools).await, true)
}
#[tauri::command(async)]
fn reveal(path: String) -> Result<(), String> {
    logging::result(
        "reveal",
        (|| {
            let path = std::path::Path::new(&path)
                .canonicalize()
                .map_err(|e| e.to_string())?;
            let status = std::process::Command::new("open")
                .arg("-R")
                .arg(path)
                .status()
                .map_err(|e| e.to_string())?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("Finder reveal failed: {status}"))
            }
        })(),
        false,
    )
}
#[tauri::command(async)]
fn create_batch(
    state: State<'_, Studio>,
    session_id: String,
    draft: batch::Draft,
) -> Result<batch::BatchRef, String> {
    logging::result("create_batch", state.create_batch(session_id, draft), true)
}
#[tauri::command]
async fn render_batch_job(
    state: State<'_, Studio>,
    session_id: String,
    batch_id: String,
    job_id: String,
) -> Result<Vec<SavedImage>, String> {
    logging::result(
        "render_batch_job",
        state.render_batch_job(session_id, batch_id, job_id).await,
        true,
    )
}
#[tauri::command(async)]
fn discard_batch_jobs(
    state: State<'_, Studio>,
    session_id: String,
    batch_id: String,
    job_ids: Vec<String>,
) -> Result<(), String> {
    logging::result(
        "discard_batch_jobs",
        state.discard_batch_jobs(session_id, batch_id, job_ids),
        true,
    )
}
#[tauri::command]
fn frontend_logs(entries: Vec<logging::Entry>) -> Result<(), String> {
    logging::result("frontend_logs", logging::frontend(entries), false)
}
#[tauri::command]
fn get_log_path() -> Result<String, String> {
    logging::result("get_log_path", logging::path(), false)
}
pub fn run() {
    let started = (|| -> Result<(), Box<dyn std::error::Error>> {
        let home = dirs::home_dir().ok_or("Home directory unavailable")?;
        let log_dir = home.join(".local/state/xwen/image-studio/logs");
        logging::secure_directory(&log_dir)?;
        let config_path = home.join(".config/xwen/image-studio.json");
        logging::register_config_file(&config_path);
        let app = tauri::Builder::default()
            .plugin(logging::builder(log_dir, 5 * 1024 * 1024).build())
            .plugin(tauri_plugin_dialog::init())
            .setup(move |app| {
                logging::panic_hook();
                log::info!(target:"image_studio::lifecycle","Image Studio starting");
                let state = Studio::new(config_path.clone(), std::env::args().skip(1).collect())
                    .map_err(|error| {
                        log::error!(target: "image_studio::lifecycle", "State initialization failed: {}", logging::sanitize(&error, 8192));
                        error
                    })?;
                app.manage(state);
                Ok(())
            })
            .invoke_handler(tauri::generate_handler![
                create_batch,render_batch_job,discard_batch_jobs,bootstrap, save_config, select_workspace, read_image, list_loras,
                check_server, preprocess, render, list_images, list_sessions,
                delete_image, delete_session, generate_prompt, chat, reveal, frontend_logs, get_log_path
            ])
            .build(tauri::generate_context!())?;
        app.run(|_, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                log::info!(target:"image_studio::lifecycle","Image Studio stopped");
                log::logger().flush();
            }
        });
        Ok(())
    })();
    if let Err(error) = started {
        eprintln!(
            "Image Studio startup failed: {}",
            logging::sanitize(&error.to_string(), 8192)
        );
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests;
