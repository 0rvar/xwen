use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};
static KEYS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
fn keys() -> &'static Mutex<Vec<String>> {
    KEYS.get_or_init(|| Mutex::new(Vec::new()))
}
pub fn register_key(key: &str) {
    if key.is_empty() {
        return;
    }
    let mut keys = keys().lock().unwrap_or_else(|e| e.into_inner());
    if !keys.iter().any(|old| old == key) {
        keys.push(key.into());
        keys.sort_by_key(|s| std::cmp::Reverse(s.len()));
    }
}
pub fn register_config_file(path: &Path) {
    if let Ok(bytes) = std::fs::read(path) {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(key) = value["api_key"].as_str() {
                register_key(key)
            }
        } else if let Ok(text) = std::str::from_utf8(&bytes) {
            if let Some((_, rest)) = text.split_once("\"api_key\"") {
                if let Some((_, value)) = rest.split_once(':') {
                    let mut reader = serde_json::Deserializer::from_str(value.trim_start());
                    if let Ok(key) = String::deserialize(&mut reader) {
                        register_key(&key)
                    }
                }
            }
        }
    }
}
fn clip(text: &str, max: usize) -> &str {
    let mut end = text.len().min(max);
    while !text.is_char_boundary(end) {
        end -= 1
    }
    &text[..end]
}
pub fn sanitize(text: &str, max: usize) -> String {
    let mut text = text.to_owned();
    while let Some(start) = text.find("data:image/") {
        let end = text[start..]
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>'))
            .map(|n| start + n)
            .unwrap_or(text.len());
        text.replace_range(start..end, "[image data redacted]");
    }
    for key in keys().lock().unwrap_or_else(|e| e.into_inner()).iter() {
        text = text.replace(key, "[redacted]");
    }
    clip(&text, max).to_owned()
}
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub level: Level,
    pub source: String,
    pub message: String,
}
pub fn frontend(entries: Vec<Entry>) -> Result<(), String> {
    if entries.len() > 50 {
        return Err("At most 50 frontend log entries are allowed".into());
    }
    for entry in entries {
        let source = sanitize(&entry.source, 100).replace(['\r', '\n'], " ");
        let message = sanitize(&entry.message, 8192);
        let level = match entry.level {
            Level::Debug => log::Level::Debug,
            Level::Info => log::Level::Info,
            Level::Warn => log::Level::Warn,
            Level::Error => log::Level::Error,
        };
        log::log!(target:"image_studio::frontend",level,"[{source}] {message}");
    }
    Ok(())
}
pub fn result<T>(operation: &str, result: Result<T, String>, important: bool) -> Result<T, String> {
    match &result {
        Ok(_) => {
            if important {
                log::info!(target:"image_studio::command","{operation} succeeded")
            } else {
                log::debug!(target:"image_studio::command","{operation} succeeded")
            }
        }
        Err(error) => {
            log::error!(target:"image_studio::command","{operation} failed: {}",sanitize(error,8192))
        }
    }
    result
}
pub fn directory() -> Result<PathBuf, String> {
    dirs::home_dir()
        .map(|p| p.join(".local/state/xwen/image-studio/logs"))
        .ok_or("Home directory unavailable".into())
}
pub fn path() -> Result<String, String> {
    Ok(directory()?
        .join("image-studio.log")
        .to_string_lossy()
        .into())
}
pub fn secure_directory(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .map_err(|e| e.to_string())
}
pub fn panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        log::error!(target:"image_studio::panic","panic at {location}: {}",sanitize(payload,8192));
        log::logger().flush();
        previous(info);
    }));
}

pub fn builder(directory: PathBuf, max_bytes: u128) -> tauri_plugin_log::Builder {
    use tauri_plugin_log::{RotationStrategy, Target, TargetKind};
    tauri_plugin_log::Builder::new()
        .clear_targets()
        .level(log::LevelFilter::Debug)
        .filter(|metadata| metadata.target().starts_with("image_studio"))
        .max_file_size(max_bytes)
        .rotation_strategy(RotationStrategy::KeepSome(1))
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] [{}] {}",
                chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                record.level(),
                record.target(),
                sanitize(&message.to_string(), 8400)
            ))
        })
        .target(Target::new(TargetKind::Folder {
            path: directory,
            file_name: Some("image-studio".into()),
        }))
}
