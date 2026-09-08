use crate::models::*;
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Component, Path},
};

pub type Result<T> = std::result::Result<T, String>;
pub fn unique() -> String {
    use std::io::Read;
    let mut bytes = [0u8; 12];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .expect("OS randomness");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn atomic(path: &Path, bytes: &[u8], replace: bool) -> Result<()> {
    let parent = path.parent().ok_or("file has no parent")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let tmp = parent.join(format!(".studio-{}.tmp", unique()));
    let result = (|| {
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp).map_err(|e| e.to_string())?;
        f.write_all(bytes)
            .and_then(|_| f.sync_all())
            .map_err(|e| e.to_string())?;
        if replace {
            fs::rename(&tmp, path)
        } else {
            match fs::hard_link(&tmp, path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(e),
                Err(_) => reserve_and_rename(&tmp, path),
            }
        }
        .map_err(|e| format!("save {}: {e}", path.display()))?;
        Ok(())
    })();
    let _ = fs::remove_file(tmp);
    result
}
pub(crate) fn reserve_and_rename(tmp: &Path, path: &Path) -> std::io::Result<()> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let reservation = opts.open(path)?;
    drop(reservation);
    let result = fs::rename(tmp, path);
    if result.is_err() {
        let _ = fs::remove_file(path);
    }
    result
}
pub fn decode(bytes: &[u8]) -> Result<(image::DynamicImage, &'static str)> {
    let format = image::guess_format(bytes).map_err(|e| e.to_string())?;
    let mime = match format {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        _ => return Err("Only PNG and JPEG images are supported".into()),
    };
    if bytes.len() > 100_000_000 {
        return Err("Image exceeds 100 MB".into());
    }
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let img = reader.decode().map_err(|e| e.to_string())?;
    Ok((img, mime))
}
pub fn data(bytes: &[u8], mime: &str) -> String {
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}
pub fn decode_data(value: &str) -> Result<Vec<u8>> {
    let encoded = if value.starts_with("data:") {
        let (prefix, data) = value.split_once(',').ok_or("Invalid image data URL")?;
        if !matches!(prefix, "data:image/png;base64" | "data:image/jpeg;base64") {
            return Err("Expected PNG/JPEG base64 data URL".into());
        }
        data
    } else {
        value
    };
    if encoded.len() > 140_000_000 {
        return Err("Image exceeds 100 MB".into());
    }
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|e| format!("Invalid base64 image: {e}"))?;
    decode(&bytes)?;
    Ok(bytes)
}
pub fn thumbnail(bytes: &[u8]) -> Result<String> {
    let (img, _) = decode(bytes)?;
    let mut preview = std::io::Cursor::new(Vec::new());
    img.thumbnail(512, 512)
        .write_to(&mut preview, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(data(preview.get_ref(), "image/png"))
}
pub fn read_image(path: &Path) -> Result<InputImage> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > 100_000_000 {
        return Err("Image exceeds 100 MB".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let (img, mime) = decode(&bytes)?;
    Ok(InputImage {
        name: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into(),
        data_url: data(&bytes, mime),
        width: img.width(),
        height: img.height(),
    })
}
pub fn snapshot(session: &Path, value: &str) -> Result<Value> {
    let bytes = decode_data(value)?;
    let (img, mime) = decode(&bytes)?;
    let sha = hash(&bytes);
    let rel = format!(
        "inputs/{sha}.{}",
        if mime == "image/png" { "png" } else { "jpg" }
    );
    let path = session.join(&rel);
    fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    if !path
        .parent()
        .unwrap()
        .canonicalize()
        .map_err(|e| e.to_string())?
        .starts_with(session.canonicalize().map_err(|e| e.to_string())?)
    {
        return Err("Snapshot directory escapes session".into());
    }
    if path.exists() {
        if fs::read(&path).map_err(|e| e.to_string())? != bytes {
            return Err("Snapshot collision".into());
        }
    } else {
        atomic(&path, &bytes, false)?
    }
    Ok(json!({"path":rel,"sha256":sha,"width":img.width(),"height":img.height()}))
}
pub fn confined(session: &Path, relative: &str) -> Result<std::path::PathBuf> {
    let path = Path::new(relative);
    if path
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err("Unsafe asset path".into());
    }
    let full = session
        .join(path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let root = session.canonicalize().map_err(|e| e.to_string())?;
    if !full.starts_with(root) {
        return Err("Asset escapes session".into());
    }
    Ok(full)
}
pub fn gallery(workspace: &Workspace) -> Result<Vec<SavedImage>> {
    let mut records = Vec::new();
    let mut candidates = Vec::new();
    let mut verified_assets = std::collections::HashSet::new();
    for dir in fs::read_dir(&workspace.path)
        .map_err(|e| e.to_string())?
        .flatten()
    {
        if !dir.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let session = dir.path();
        let Ok(entries) = fs::read_dir(&session) else {
            continue;
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false)
                || entry.path().extension().and_then(|x| x.to_str()) != Some("yaml")
            {
                continue;
            }
            if entry
                .metadata()
                .map(|m| m.len() > 4_000_000)
                .unwrap_or(true)
            {
                continue;
            }
            let Ok(bytes) = fs::read(entry.path()) else {
                continue;
            };
            let Ok(m) = serde_yaml_ng::from_slice::<Value>(&bytes) else {
                continue;
            };
            if m["app_id"] != APP_ID
                || m["schema_version"] != 1
                || m["session_id"].as_str() != dir.file_name().to_str()
            {
                continue;
            }
            candidates.push((session.clone(), entry.path(), m));
        }
    }
    candidates.sort_by(|a, b| b.2["created_at"].as_str().cmp(&a.2["created_at"].as_str()));
    for (session, metadata_path, m) in candidates {
        if records.len() == 200 {
            break;
        }
        let result = (|| -> Result<SavedImage> {
            let file = m["output"]["file"].as_str().ok_or("Missing output")?;
            if file
                != format!(
                    "{}.png",
                    metadata_path.file_stem().unwrap().to_string_lossy()
                )
            {
                return Err("Output filename mismatch".into());
            }
            let path = confined(&session, file)?;
            if fs::symlink_metadata(session.join(file))
                .map_err(|e| e.to_string())?
                .file_type()
                .is_symlink()
            {
                return Err("Symlink output".into());
            }
            if let Some(assets) = m["assets"].as_object() {
                for asset in assets.values() {
                    let relative = asset["path"].as_str().ok_or("Missing asset path")?;
                    let asset_path = confined(&session, relative)?;
                    let info =
                        fs::symlink_metadata(session.join(relative)).map_err(|e| e.to_string())?;
                    if info.file_type().is_symlink() || info.len() > 100_000_000 {
                        return Err("Invalid asset file".into());
                    }
                    let expected = asset["sha256"].as_str().ok_or("Missing asset hash")?;
                    use std::os::unix::fs::MetadataExt;
                    let identity = (
                        asset_path.clone(),
                        expected.to_owned(),
                        info.len(),
                        info.modified().ok(),
                        info.ino(),
                        info.ctime(),
                        info.ctime_nsec(),
                    );
                    if !verified_assets.contains(&identity) {
                        let bytes = fs::read(&asset_path).map_err(|e| e.to_string())?;
                        if expected != hash(&bytes) {
                            return Err("Asset hash mismatch".into());
                        }
                        verified_assets.insert(identity);
                    }
                }
            }
            for (key, reference) in [
                ("init_image", &m["request"]["init_image"]),
                ("mask", &m["request"]["mask"]),
                ("control.image", &m["request"]["control"]["image"]),
            ] {
                if let Some(relative) = reference.as_str() {
                    if m["assets"][key]["path"] != relative {
                        return Err("Request asset mismatch".into());
                    }
                    confined(&session, relative)?;
                }
            }
            if fs::metadata(&path).map_err(|e| e.to_string())?.len() > 100_000_000 {
                return Err("Image too large".into());
            }
            let png = fs::read(&path).map_err(|e| e.to_string())?;
            if m["output"]["sha256"] != hash(&png) {
                return Err("Output hash mismatch".into());
            }
            let (img, _) = decode(&png)?;
            let mut preview = std::io::Cursor::new(Vec::new());
            img.thumbnail(512, 512)
                .write_to(&mut preview, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            Ok(SavedImage {
                id: format!(
                    "{}/{}",
                    session.file_name().unwrap().to_string_lossy(),
                    file
                ),
                path: path.to_string_lossy().into(),
                metadata_path: metadata_path.to_string_lossy().into(),
                data_url: data(preview.get_ref(), "image/png"),
                seed: m["output"]["seed"].as_u64().ok_or("Missing seed")?,
                width: img.width(),
                height: img.height(),
                session_id: m["session_id"].as_str().unwrap().into(),
                prompt: m["request"]["prompt"].as_str().unwrap_or_default().into(),
                metadata: m,
            })
        })();
        if let Ok(item) = result {
            records.push(item)
        }
    }
    records.sort_by(|a, b| {
        b.metadata["created_at"]
            .as_str()
            .cmp(&a.metadata["created_at"].as_str())
    });
    records.truncate(200);
    Ok(records)
}

pub fn valid_session_id(id: &str) -> bool {
    let Some((date, random)) = id.rsplit_once('-') else {
        return false;
    };
    random.len() == 24
        && random
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        && chrono::NaiveDateTime::parse_from_str(date, "%Y%m%d-%H%M%S").is_ok()
        && date.len() == 15
}

pub fn session_path(workspace: &Workspace, id: &str) -> Result<std::path::PathBuf> {
    if !valid_session_id(id) {
        return Err("Invalid session id".into());
    }
    let root = Path::new(&workspace.path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if root != Path::new(&workspace.path) {
        return Err("Workspace path changed".into());
    }
    let path = root.join(id);
    match fs::symlink_metadata(&path) {
        Ok(info) => {
            if !info.is_dir()
                || info.file_type().is_symlink()
                || path.canonicalize().map_err(|e| e.to_string())?.parent() != Some(root.as_path())
            {
                return Err("Session must be a direct real directory in the workspace".into());
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && id == workspace.session_id => {}
        Err(e) => return Err(e.to_string()),
    }
    Ok(path)
}

pub fn owned_output(
    session: &Path,
    file: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    if Path::new(file).components().count() != 1
        || !matches!(
            Path::new(file).components().next(),
            Some(Component::Normal(_))
        )
        || !file.ends_with(".png")
    {
        return Err("Invalid image id".into());
    }
    let png = session.join(file);
    let yaml = png.with_extension("yaml");
    for path in [&png, &yaml] {
        let info = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !info.is_file() || info.file_type().is_symlink() {
            return Err("Image and metadata must be regular files".into());
        }
    }
    owned_metadata(session, file, &yaml)?;
    Ok((png, yaml))
}

fn owned_metadata(session: &Path, file: &str, yaml: &Path) -> Result<()> {
    if fs::metadata(&yaml).map_err(|e| e.to_string())?.len() > 4_000_000 {
        return Err("Metadata too large".into());
    }
    let metadata: Value = serde_yaml_ng::from_slice(&fs::read(&yaml).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if metadata["app_id"] != APP_ID
        || metadata["schema_version"] != 1
        || metadata["session_id"].as_str() != session.file_name().and_then(|s| s.to_str())
        || metadata["output"]["file"] != file
    {
        return Err("Image is not owned by this Image Studio session".into());
    }
    Ok(())
}

pub fn session_image_count(session: &Path) -> Result<usize> {
    let mut count = 0;
    for entry in fs::read_dir(session).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let name = name.to_str().ok_or("Unrecognized session file")?;
        if kind.is_symlink() {
            return Err("Session contains a symbolic link".into());
        }
        if kind.is_dir() && name == "inputs" {
            for input in fs::read_dir(entry.path()).map_err(|e| e.to_string())? {
                let input = input.map_err(|e| e.to_string())?;
                let kind = input.file_type().map_err(|e| e.to_string())?;
                let name = input.file_name();
                let name = name.to_str().ok_or("Invalid input snapshot")?;
                if kind.is_file() && name == ".DS_Store" {
                    continue;
                }
                let Some((hash, extension)) = name.rsplit_once('.') else {
                    return Err("Invalid input snapshot".into());
                };
                if !kind.is_file()
                    || hash.len() != 64
                    || !hash.bytes().all(|b| b.is_ascii_hexdigit())
                    || !matches!(extension, "png" | "jpg")
                {
                    return Err("Session contains unrecognized input files".into());
                }
            }
        } else if kind.is_file() && name == ".DS_Store" {
            continue;
        } else if kind.is_file() && name.ends_with(".png") {
            owned_output(session, name)?;
            count += 1;
        } else if kind.is_file() && name.ends_with(".yaml") {
            let file = format!("{}.png", name.strip_suffix(".yaml").unwrap());
            if session.join(&file).exists() {
                owned_output(session, &file)?;
            } else {
                owned_metadata(session, &file, &entry.path())?;
            }
        } else {
            return Err("Session contains unrecognized files".into());
        }
    }
    Ok(count)
}

pub fn sessions(workspace: &Workspace) -> Result<Vec<SessionSummary>> {
    let mut result = Vec::new();
    for entry in fs::read_dir(&workspace.path)
        .map_err(|e| e.to_string())?
        .flatten()
    {
        let Some(id) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(path) = session_path(workspace, &id) else {
            continue;
        };
        if let Ok(image_count) = session_image_count(&path) {
            result.push(SessionSummary {
                session_id: id,
                image_count,
            });
        }
    }
    if !result.iter().any(|s| s.session_id == workspace.session_id) {
        result.push(SessionSummary {
            session_id: workspace.session_id.clone(),
            image_count: 0,
        });
    }
    result.sort_by(|a, b| b.session_id.cmp(&a.session_id));
    Ok(result)
}
