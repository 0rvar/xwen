mod models;
mod state;
mod storage;
use models::*;
use state::Studio;
use tauri::State;
#[tauri::command]
fn bootstrap(state: State<'_, Studio>) -> Result<Bootstrap, String> {
    state.bootstrap()
}
#[tauri::command]
fn save_config(state: State<'_, Studio>, config: Config) -> Result<Config, String> {
    state.save_config(config)
}
#[tauri::command]
fn select_workspace(state: State<'_, Studio>, path: String) -> Result<Workspace, String> {
    state.select_workspace(&path)
}
#[tauri::command(async)]
fn read_image(path: String) -> Result<InputImage, String> {
    storage::read_image(std::path::Path::new(&path))
}
#[tauri::command]
async fn list_loras(state: State<'_, Studio>) -> Result<Vec<LoraCandidate>, String> {
    state.list_loras().await
}
#[tauri::command]
async fn check_server(
    state: State<'_, Studio>,
    config: Option<Config>,
) -> Result<serde_json::Value, String> {
    state.check_server(config).await
}
#[tauri::command]
async fn preprocess(
    state: State<'_, Studio>,
    image: String,
    kind: String,
) -> Result<Preprocessed, String> {
    state.preprocess(image, kind).await
}
#[tauri::command]
async fn render(
    state: State<'_, Studio>,
    session_id: String,
    request: RenderRequest,
    context: serde_json::Value,
) -> Result<Vec<SavedImage>, String> {
    state.render(session_id, request, context).await
}
#[tauri::command(async)]
fn list_images(state: State<'_, Studio>) -> Result<Vec<SavedImage>, String> {
    state.list_images()
}
#[tauri::command(async)]
fn reveal(path: String) -> Result<(), String> {
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
}
pub fn run() {
    let home = dirs::home_dir().expect("Home directory unavailable");
    let state = Studio::new(
        home.join(".config/xwen/image-studio.json"),
        std::env::args().skip(1).collect(),
    )
    .expect("HTTP client initialization failed");
    tauri::Builder::default()
        .manage(state)
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            bootstrap,
            save_config,
            select_workspace,
            read_image,
            list_loras,
            check_server,
            preprocess,
            render,
            list_images,
            reveal
        ])
        .run(tauri::generate_context!())
        .expect("Image Studio failed");
}
#[cfg(test)]
mod tests;
