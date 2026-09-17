//! The server's operational log on disk.
//!
//! Every line `xwen serve` reports — the lazy load's resident-memory figure, a
//! cache clear that failed, the failure a request was answered with — reaches
//! whichever sink is drawing the server, and on a `--tui` run that sink is a
//! frame that is gone as soon as it is redrawn. This module is the copy that
//! outlives the frame: one file under the XDG state directory, beside the
//! metrics history, holding the same lines with a timestamp in front of each.
//!
//! Only `serve` installs it. A `generate` or `chat` run owns its terminal and
//! its lines belong there, so with nothing installed every append is a no-op.
//!
//! It is BOUNDED, unlike the metrics history: a log of what happened is worth
//! its most recent 16 MiB and a server left running for a month must not fill a
//! disk. Past the cap the file is truncated in place rather than rotated, which
//! keeps the inode a `tail -f` is following.

use std::ffi::OsStr;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};

/// The default log under `$HOME`, which is the path the README names and
/// [`SERVE_LOG_ENV`] overrides.
pub const SERVE_LOG_RELATIVE_PATH: &str = ".local/state/xwen/serve.log";

/// Names the file to log into, or `off` (in any casing) to log nothing. An empty
/// value counts as unset and resolves the default path — the same rule
/// [`crate::metrics::METRICS_ENV`] follows, so an operator who knows one knows
/// the other.
pub const SERVE_LOG_ENV: &str = "XWEN_SERVE_LOG";

/// How large the log grows before it is truncated in place.
pub const SERVE_LOG_MAX_BYTES: u64 = 16 * 1024 * 1024;

/// The file this process logs into, or `None` when logging is off: either
/// `XWEN_SERVE_LOG` says `off`, or there is no `$HOME` to resolve the default
/// under.
pub fn serve_log_path() -> Option<PathBuf> {
    serve_log_path_from(
        std::env::var_os(SERVE_LOG_ENV).as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// [`serve_log_path`] over a value rather than the process environment, which is
/// what makes the rule testable: a test that set the variable would be changing
/// state every other thread in the runner shares.
pub fn serve_log_path_from(file_env: Option<&OsStr>, home: Option<&OsStr>) -> Option<PathBuf> {
    if file_env
        .and_then(OsStr::to_str)
        .is_some_and(|value| value.eq_ignore_ascii_case("off"))
    {
        return None;
    }
    match file_env {
        Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => home.map(|home| PathBuf::from(home).join(SERVE_LOG_RELATIVE_PATH)),
    }
}

/// An append-only handle on the log, opened once and kept for the life of the
/// server: a line costs one write and one flush, never an open.
pub(crate) struct HostLogFile {
    path: PathBuf,
    file: File,
}

impl HostLogFile {
    /// Open the log, creating the file and, if it is missing, the directory
    /// holding it.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = match open_for_append(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if let Some(parent) = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                open_for_append(path).with_context(|| format!("opening {}", path.display()))?
            }
            other => other.with_context(|| format!("opening {}", path.display()))?,
        };
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Append one line, timestamped, truncating the file first when it has
    /// reached [`SERVE_LOG_MAX_BYTES`].
    ///
    /// The write is one call on an `O_APPEND` handle, which is what makes the
    /// truncation safe with a second server logging to the same file: every
    /// append lands at the end as the kernel finds it, so a file another process
    /// has just emptied is written from zero rather than leaving a hole.
    pub(crate) fn append(&mut self, line: &str) -> Result<()> {
        if self
            .file
            .metadata()
            .with_context(|| format!("sizing {}", self.path.display()))?
            .len()
            >= SERVE_LOG_MAX_BYTES
        {
            self.file
                .set_len(0)
                .with_context(|| format!("truncating {}", self.path.display()))?;
        }
        let mut row = String::with_capacity(line.len() + 22);
        row.push_str(&timestamp(crate::metrics::now_secs()));
        row.push(' ');
        // One entry is one line, so that `grep` over the log answers about
        // entries: a line's author may have wrapped it, and the breaks become
        // spaces rather than rows nothing identifies.
        row.extend(line.chars().map(|ch| match ch {
            '\n' | '\r' => ' ',
            other => other,
        }));
        row.push('\n');
        self.file
            .write_all(row.as_bytes())
            .with_context(|| format!("appending to {}", self.path.display()))?;
        self.file.flush()?;
        Ok(())
    }
}

fn open_for_append(path: &Path) -> std::io::Result<File> {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
}

/// A UTC instant as the log spells it, `2026-09-17T09:41:07Z`. UTC rather than
/// local time because the offset costs a subprocess to read
/// ([`crate::metrics::read_utc_offset`]) and this runs once per line; a reader
/// comparing the log against a client's own timestamps wants an unambiguous one
/// either way.
fn timestamp(secs: u64) -> String {
    let (year, month, day) = crate::metrics::civil_from_days((secs / 86_400) as i64);
    let time = secs % 86_400;
    let (hour, minute, second) = (time / 3600, (time % 3600) / 60, time % 60);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// The log this process appends to, once `serve` has opened one.
static FILE: OnceLock<Mutex<HostLogFile>> = OnceLock::new();

/// Whether this process has already said that logging is failing.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Open this process's log, returning the one line to report when it cannot be
/// opened. Called by `serve` and by nothing else; `None` also means logging is
/// turned off, which is a setting rather than a failure.
///
/// The first caller wins, so a second server in one process keeps the first's
/// file.
pub(crate) fn install() -> Option<String> {
    install_at(&serve_log_path()?)
}

/// [`install`] into a named file rather than the one the environment
/// resolves, which is what lets a test drive the whole path from a logged event
/// to the file.
fn install_at(path: &Path) -> Option<String> {
    match HostLogFile::open(path) {
        Ok(file) => {
            let _ = FILE.set(Mutex::new(file));
            None
        }
        Err(error) => Some(format!(
            "warning: serve log not written ({}): {error:#}",
            path.display()
        )),
    }
}

/// Whether this process has a log to append to, which is what lets a caller
/// skip wording a line nobody will read.
pub(super) fn logging() -> bool {
    FILE.get().is_some()
}

/// Append one line, returning the single line this process will ever say about
/// failing to.
///
/// The log is a side effect of serving and never a reason for a request to fail,
/// and a full disk would otherwise repeat itself once per line for the life of
/// the server. The warning is returned rather than logged here, so that the
/// caller reports it without reaching back through this function.
pub(super) fn append_warning(line: &str) -> Option<String> {
    let mut file = FILE.get()?.lock().unwrap_or_else(|e| e.into_inner());
    let error = file.append(line).err()?;
    if WARNED.swap(true, Ordering::Relaxed) {
        return None;
    }
    Some(format!(
        "warning: serve log not written ({}): {error:#}",
        file.path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    /// A log file of this test's own, under a name no other test uses.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xwen-servelog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch dir");
        let path = dir.join(name);
        let _ = std::fs::remove_file(&path);
        path
    }

    /// `off` in any casing turns the log off; an empty value is how a shell
    /// spells "unset" by accident and resolves the default path.
    #[test]
    fn the_env_variable_names_a_file_or_turns_the_log_off() {
        let home = std::ffi::OsString::from("/home/someone");
        assert_eq!(
            serve_log_path_from(None, Some(&home)),
            Some(PathBuf::from("/home/someone/.local/state/xwen/serve.log"))
        );
        assert_eq!(
            serve_log_path_from(Some(OsStr::new("")), Some(&home)),
            Some(PathBuf::from("/home/someone/.local/state/xwen/serve.log"))
        );
        assert_eq!(
            serve_log_path_from(Some(OsStr::new("/tmp/elsewhere.log")), Some(&home)),
            Some(PathBuf::from("/tmp/elsewhere.log"))
        );
        assert_eq!(
            serve_log_path_from(Some(OsStr::new("off")), Some(&home)),
            None
        );
        assert_eq!(
            serve_log_path_from(Some(OsStr::new("OFF")), Some(&home)),
            None
        );
        // Nowhere to resolve the default under is a log that cannot be pointed
        // anywhere, not one that was turned off.
        assert_eq!(serve_log_path_from(None, None), None);
    }

    /// Lines accumulate in the order they were appended, each stamped and each
    /// on one row however its author wrapped it.
    #[test]
    fn appending_stamps_every_line_and_keeps_one_entry_to_a_row() {
        let path = scratch("appends.log");
        let mut log = HostLogFile::open(&path).expect("a log file");
        log.append("xwen serve: model loaded in 4.2s")
            .expect("a line");
        log.append("xwen serve: cache clear failed\nit panicked")
            .expect("a second line");

        let text = std::fs::read_to_string(&path).expect("the log reads back");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "one row per entry: {text:?}");
        assert!(
            lines[0].ends_with(" xwen serve: model loaded in 4.2s"),
            "the line follows its stamp: {:?}",
            lines[0]
        );
        // `2026-09-17T09:41:07Z` — twenty columns, then the separating space.
        assert_eq!(
            lines[0].len() - " xwen serve: model loaded in 4.2s".len(),
            20
        );
        assert!(
            lines[1].ends_with(" xwen serve: cache clear failed it panicked"),
            "a wrapped line becomes one row: {:?}",
            lines[1]
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Past the cap the log starts again in the same file: a `tail -f` on it
    /// keeps reading, which a rotation would end.
    #[test]
    fn a_log_past_the_cap_is_truncated_in_place() {
        let path = scratch("bounded.log");
        let mut log = HostLogFile::open(&path).expect("a log file");
        log.append("xwen serve: the line before the cap")
            .expect("a line");
        let inode = std::fs::metadata(&path).expect("metadata").ino();

        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("the log opens for a resize")
            .set_len(SERVE_LOG_MAX_BYTES)
            .expect("a log at the cap");
        log.append("xwen serve: the line after the cap")
            .expect("a line past the cap");

        let text = std::fs::read_to_string(&path).expect("the log reads back");
        assert_eq!(text.lines().count(), 1, "the cap emptied it: {text:?}");
        assert!(text.ends_with(" xwen serve: the line after the cap\n"));
        assert_eq!(
            std::fs::metadata(&path).expect("metadata").ino(),
            inode,
            "the truncation keeps the inode"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_stamp_is_utc_seconds_in_iso_order() {
        assert_eq!(timestamp(1_757_030_400), "2025-09-05T00:00:00Z");
        assert_eq!(timestamp(1_757_030_400 + 34_867), "2025-09-05T09:41:07Z");
    }

    /// The whole path from a logged event to the file. The install is
    /// process-wide and permanent, so this is the only test that may claim it,
    /// and what it asserts is that the line is THERE — every other test in the
    /// binary that logs a line is appending to the same scratch file.
    #[test]
    fn a_logged_line_reaches_the_installed_file() {
        let path = scratch("installed.log");
        assert_eq!(install_at(&path), None, "the log opens");
        let (logger, _events) = crate::serve::log::collecting();
        logger.log(crate::serve::log::ServeLog::HostLine(
            "xwen: weights 12.3GB + KV 1.0GB = 13.3GB resident".to_string(),
        ));
        let text = std::fs::read_to_string(&path).expect("the log reads back");
        assert!(
            text.contains(" xwen: weights 12.3GB + KV 1.0GB = 13.3GB resident\n"),
            "the logged line was written: {text:?}"
        );
    }
}
