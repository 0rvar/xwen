use super::*;
use serde_json::json;
use std::{fs, path::PathBuf};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("studio-test-{}", storage::unique()));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn studio(&self) -> Studio {
        Studio::new(self.0.join("config.json"), vec![]).unwrap()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn request() -> RenderRequest {
    serde_json::from_value(json!({"prompt":"A red cube","width":128,"height":128,"steps":8,"seed":47,"n":1,"loras":[]})).unwrap()
}
fn png(w: u32, h: u32) -> Vec<u8> {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        w,
        h,
        image::Rgb([45, 78, 112]),
    ));
    let mut out = std::io::Cursor::new(vec![]);
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}
#[test]
fn config_and_cli() {
    assert_eq!(
        state::normalize_url("https://example.test/prefix/v1/").unwrap(),
        "https://example.test/prefix"
    );
    assert!(state::normalize_url("https://secret@example.test").is_err());
    assert!(state::parse_args(&["--bad".into()]).is_err());
    assert_eq!(
        state::parse_args(&["--workspace".into(), "a b".into()]).unwrap(),
        Some("a b".into())
    );
    let t = Temp::new();
    let s = t.studio();
    assert!(s.bootstrap().unwrap().config.server_url.is_empty());
    let w = s
        .select_workspace(t.0.join("images").to_str().unwrap())
        .unwrap();
    assert!(!PathBuf::from(&w.session_path).exists());
    let mut valid = Config::default();
    valid.server_url = "http://127.0.0.1:5241".into();
    let c = s.save_config(valid).unwrap();
    assert_eq!(c.last_workspace.as_deref(), Some(w.path.as_str()));
    assert_eq!(
        s.bootstrap().unwrap().workspace.unwrap().session_id,
        w.session_id
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&s_config(&t)).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    fs::write(s_config(&t), "broken").unwrap();
    assert!(s.save_config(c).is_err());
    assert_eq!(fs::read_to_string(s_config(&t)).unwrap(), "broken");
}
fn s_config(t: &Temp) -> PathBuf {
    t.0.join("config.json")
}
#[test]
fn missing_last_is_warning_and_atomic_never_overwrites() {
    let t = Temp::new();
    let mut config = Config::default();
    config.last_workspace = Some(t.0.join("missing").to_string_lossy().into());
    fs::write(s_config(&t), serde_json::to_vec(&config).unwrap()).unwrap();
    let result = t.studio().bootstrap().unwrap();
    assert!(result.workspace.is_none());
    assert!(result.warning.is_some());
    let file = t.0.join("output");
    storage::atomic(&file, b"first", false).unwrap();
    assert!(storage::atomic(&file, b"second", false).is_err());
    assert_eq!(fs::read(file).unwrap(), b"first");
}
#[tokio::test]
async fn real_png_http_saves_provenance_and_confines_gallery() {
    use std::io::{Read, Write};
    let t = Temp::new();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let image = png(128, 128);
    let source = storage::data(&image, "image/png");
    let response=json!({"model":"Z-Image-Turbo","steps":8,"size":"128x128","data":[{"b64_json":source.split_once(',').unwrap().1,"seed":47,"start_step":3,"control_map":source.split_once(',').unwrap().1}]}).to_string();
    let thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buffer = vec![];
        let mut scratch = [0u8; 4096];
        loop {
            let n = stream.read(&mut scratch).unwrap();
            buffer.extend_from_slice(&scratch[..n]);
            if let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&buffer[..pos]);
                let len = header
                    .lines()
                    .find_map(|l| {
                        l.to_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|s| s.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if buffer.len() >= pos + 4 + len {
                    assert!(header.starts_with("POST /v1/images/render"));
                    assert!(
                        header
                            .to_lowercase()
                            .contains("authorization: bearer test-secret")
                    );
                    break;
                }
            }
        }
        write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",response.len(),response).unwrap();
    });
    let s = t.studio();
    s.bootstrap().unwrap();
    let mut config = Config::default();
    config.server_url = format!("http://{addr}");
    config.api_key = "test-secret".into();
    s.save_config(config).unwrap();
    let workspace = s
        .select_workspace(t.0.join("images").to_str().unwrap())
        .unwrap();
    assert!(
        s.render("wrong".into(), request(), json!({}))
            .await
            .is_err()
    );
    let mut req = request();
    req.init_image = Some(source.clone());
    req.strength = Some(0.6);
    let saved = s
        .render(
            workspace.session_id.clone(),
            req,
            json!({"mode":"img2img","api_key":"test-secret"}),
        )
        .await
        .unwrap();
    thread.join().unwrap();
    assert_eq!(saved.len(), 1);
    assert_eq!(fs::read(&saved[0].path).unwrap(), image);
    let yaml = fs::read_to_string(&saved[0].metadata_path).unwrap();
    assert!(!yaml.contains("test-secret"));
    assert!(!yaml.contains("base64"));
    assert_eq!(saved[0].metadata["output"]["seed"], 47);
    assert_eq!(saved[0].metadata["output"]["start_step"], 3);
    let relative = saved[0].metadata["request"]["init_image"].as_str().unwrap();
    assert_eq!(
        fs::read(
            storage::confined(PathBuf::from(&workspace.session_path).as_path(), relative).unwrap()
        )
        .unwrap(),
        image
    );
    assert_eq!(s.list_images().unwrap().len(), 1);
    let asset_path = PathBuf::from(&workspace.session_path).join(relative);
    fs::write(&asset_path, png(128, 64)).unwrap();
    assert!(s.list_images().unwrap().is_empty());
    fs::write(&asset_path, &image).unwrap();
    assert_eq!(s.list_images().unwrap().len(), 1);
    let mut metadata = saved[0].metadata.clone();
    metadata["assets"]["init_image"]["path"] = json!("../outside.png");
    fs::write(
        &saved[0].metadata_path,
        serde_yaml_ng::to_string(&metadata).unwrap(),
    )
    .unwrap();
    assert!(s.list_images().unwrap().is_empty());
}
#[tokio::test]
#[ignore = "requires explicitly configured running xwen server; uses real image models"]
async fn live_render_saves_png_yaml() {
    let url = std::env::var("XWEN_STUDIO_SMOKE_URL").expect("Set XWEN_STUDIO_SMOKE_URL");
    let t = Temp::new();
    let s = t.studio();
    s.bootstrap().unwrap();
    let mut config = Config::default();
    config.server_url = url;
    config.api_key = std::env::var("XWEN_STUDIO_SMOKE_KEY").unwrap_or_default();
    s.save_config(config).unwrap();
    let workspace = s
        .select_workspace(t.0.join("images").to_str().unwrap())
        .unwrap();
    let live_workspace_path = PathBuf::from(&workspace.path);
    let mut req = request();
    req.width = 512;
    req.height = 512;
    let result = s
        .render(
            workspace.session_id.clone(),
            req,
            json!({"test":"live smoke"}),
        )
        .await
        .unwrap();
    let item = &result[0];
    assert_eq!((item.width, item.height, item.seed), (512, 512, 47));
    let metadata: serde_json::Value =
        serde_yaml_ng::from_slice(&fs::read(&item.metadata_path).unwrap()).unwrap();
    assert_eq!(metadata["request"]["seed"], 47);
    assert_eq!(
        metadata["output"]["sha256"],
        storage::hash(&fs::read(&item.path).unwrap())
    );
    let mut edit = request();
    edit.width = 512;
    edit.height = 512;
    edit.init_image = Some(item.data_url.clone());
    edit.strength = Some(0.0);
    let edited = s
        .render(workspace.session_id, edit, json!({"test":"zero strength"}))
        .await
        .unwrap();
    assert_eq!(
        storage::decode(&fs::read(&edited[0].path).unwrap())
            .unwrap()
            .0
            .to_rgb8(),
        storage::decode(&fs::read(&item.path).unwrap())
            .unwrap()
            .0
            .to_rgb8()
    );
    if let Ok(destination) = std::env::var("XWEN_STUDIO_SMOKE_OUTPUT") {
        fn copy_tree(source: &std::path::Path, target: &std::path::Path) {
            fs::create_dir(target).expect("Smoke output must not already exist");
            for entry in fs::read_dir(source).unwrap() {
                let entry = entry.unwrap();
                let dest = target.join(entry.file_name());
                if entry.file_type().unwrap().is_dir() {
                    copy_tree(&entry.path(), &dest);
                } else {
                    fs::copy(entry.path(), dest).unwrap();
                }
            }
        }
        copy_tree(&live_workspace_path, std::path::Path::new(&destination));
        println!("Live smoke artifacts: {destination}");
    }
}

#[tokio::test]
async fn redirects_do_not_forward_credentials_and_server_errors_are_clear() {
    use std::io::{Read, Write};
    let t = Temp::new();
    let s = t.studio();
    s.bootstrap().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let thread = std::thread::spawn(move || {
        for (status, body, extra) in [
            (
                "302 Found",
                r#"{"error":{"message":"redirect rejected"}}"#,
                "Location: http://127.0.0.1:1/steal\r\n",
            ),
            (
                "403 Forbidden",
                r#"{"error":{"message":"invalid API key test-secret"}}"#,
                "",
            ),
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            let mut b = [0u8; 2048];
            stream.read(&mut b).unwrap();
            write!(
                stream,
                "HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        }
    });
    let mut c = Config::default();
    c.server_url = format!("http://{addr}");
    c.api_key = "test-secret".into();
    s.save_config(c).unwrap();
    assert!(s.check_server(None).await.unwrap_err().contains("302"));
    let error = s.check_server(None).await.unwrap_err();
    assert!(error.contains("403"));
    assert!(!error.contains("test-secret"));
    thread.join().unwrap();
}

#[tokio::test]
async fn workspace_and_config_cannot_switch_during_request() {
    use std::io::{Read, Write};
    let t = Temp::new();
    let s = t.studio();
    s.bootstrap().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finish_tx, finish_rx) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut b = [0u8; 4096];
        stream.read(&mut b).unwrap();
        started_tx.send(()).unwrap();
        finish_rx.recv().unwrap();
        let body = r#"{"error":{"message":"test failure"}}"#;
        write!(stream,"HTTP/1.1 500 Internal Server Error\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
    });
    let mut c = Config::default();
    c.server_url = format!("http://{addr}");
    s.save_config(c.clone()).unwrap();
    let w = s
        .select_workspace(t.0.join("images").to_str().unwrap())
        .unwrap();
    let render = s.render(w.session_id, request(), json!({}));
    let assertions = async {
        started_rx.await.unwrap();
        assert!(s.save_config(c.clone()).is_err());
        assert!(
            s.select_workspace(t.0.join("other").to_str().unwrap())
                .is_err()
        );
        finish_tx.send(()).unwrap();
    };
    let (result, _) = tokio::join!(render, assertions);
    assert!(result.is_err());
    assert!(s.save_config(c).is_ok());
    thread.join().unwrap();
}

#[test]
fn cli_workspace_overrides_saved_last() {
    let t = Temp::new();
    let s = t.studio();
    s.bootstrap().unwrap();
    let old = s
        .select_workspace(t.0.join("old").to_str().unwrap())
        .unwrap();
    let new = t.0.join("new");
    let launched = Studio::new(
        s_config(&t),
        vec!["--workspace".into(), new.to_string_lossy().into()],
    )
    .unwrap();
    let result = launched.bootstrap().unwrap();
    let workspace = result.workspace.unwrap();
    assert_ne!(workspace.path, old.path);
    assert_eq!(PathBuf::from(&workspace.path), new.canonicalize().unwrap());
    assert_eq!(
        result.config.last_workspace.as_deref(),
        Some(workspace.path.as_str())
    );
    assert!(!PathBuf::from(workspace.session_path).exists());
}

#[test]
fn fallback_atomic_publish_and_thumbnail_and_server_size_limits() {
    let t = Temp::new();
    let temp = t.0.join("temp");
    let out = t.0.join("output");
    fs::write(&temp, b"first").unwrap();
    storage::reserve_and_rename(&temp, &out).unwrap();
    fs::write(&temp, b"second").unwrap();
    assert!(storage::reserve_and_rename(&temp, &out).is_err());
    assert_eq!(fs::read(&out).unwrap(), b"first");
    let thumb = storage::thumbnail(&png(1024, 512)).unwrap();
    let bytes = storage::decode_data(&thumb).unwrap();
    let image = storage::decode(&bytes).unwrap().0;
    assert_eq!((image.width(), image.height()), (512, 256));
    let mut req = request();
    req.width = 4096;
    req.height = 1024;
    assert!(state::validate(&req).is_ok());
    req.width = 8208;
    assert!(state::validate(&req).is_err());
}

#[test]
fn shared_gallery_assets_keep_expected_hash_and_refresh_between_scans() {
    let t = Temp::new();
    let s = t.studio();
    s.bootstrap().unwrap();
    let workspace = s
        .select_workspace(t.0.join("images").to_str().unwrap())
        .unwrap();
    let session = PathBuf::from(&workspace.session_path);
    fs::create_dir_all(&session).unwrap();
    let image = png(128, 128);
    let asset = storage::snapshot(&session, &storage::data(&image, "image/png")).unwrap();
    for index in 0..3 {
        let name = format!("output-{index}");
        let mut reference = asset.clone();
        if index == 2 {
            reference["sha256"] = json!("incorrect");
        }
        let metadata = json!({"app_id":models::APP_ID,"schema_version":1,"session_id":workspace.session_id,"created_at":format!("2026-09-08T12:00:0{index}Z"),"request":{"prompt":"Shared source","init_image":reference["path"]},"assets":{"init_image":reference},"output":{"file":format!("{name}.png"),"sha256":storage::hash(&image),"seed":index}});
        fs::write(session.join(format!("{name}.png")), &image).unwrap();
        fs::write(
            session.join(format!("{name}.yaml")),
            serde_yaml_ng::to_string(&metadata).unwrap(),
        )
        .unwrap();
    }
    assert_eq!(s.list_images().unwrap().len(), 2);
    fs::write(session.join(asset["path"].as_str().unwrap()), png(128, 64)).unwrap();
    assert!(s.list_images().unwrap().is_empty());
}

#[tokio::test]
async fn checking_draft_connection_does_not_save_settings() {
    use std::io::{Read, Write};
    let t = Temp::new();
    let s = t.studio();
    s.bootstrap().unwrap();
    let mut saved = Config::default();
    saved.server_url = "http://127.0.0.1:1".into();
    saved.api_key = "saved-secret".into();
    s.save_config(saved.clone()).unwrap();
    let before = fs::read(s_config(&t)).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut bytes = [0u8; 4096];
        let n = stream.read(&mut bytes).unwrap();
        let request = String::from_utf8_lossy(&bytes[..n]);
        assert!(request.starts_with("GET /prefix/health"));
        assert!(
            request
                .to_lowercase()
                .contains("authorization: bearer draft-secret")
        );
        let body = r#"{"status":"ok"}"#;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
    });
    let mut draft = saved.clone();
    draft.server_url = format!("http://{addr}/prefix/v1/");
    draft.api_key = "draft-secret".into();
    assert_eq!(s.check_server(Some(draft)).await.unwrap()["status"], "ok");
    thread.join().unwrap();
    assert_eq!(fs::read(s_config(&t)).unwrap(), before);
    let current = s.bootstrap().unwrap().config;
    assert_eq!(current.server_url, saved.server_url);
    assert_eq!(current.api_key, saved.api_key);
    let mut invalid = saved;
    invalid.api_key = "bad\nheader".into();
    assert!(s.check_server(Some(invalid)).await.is_err());
    assert_eq!(fs::read(s_config(&t)).unwrap(), before);
}
