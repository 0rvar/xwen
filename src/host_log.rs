//! Where an operational line printed by the engine's core goes.
//!
//! The model layer sits below `serve` and cannot name its log types, yet a
//! process that owns the terminal — a dashboard redrawing a frame, say — cannot
//! have a raw `eprintln!` appear in the middle of it. This module is the one
//! string-shaped seam between the two: core code calls [`host_line`], and
//! whoever owns the terminal installs a consumer for it. With nothing installed
//! — every plain CLI run — a line reaches stderr exactly as `eprintln!` would
//! have put it there.

use std::sync::OnceLock;

/// The process-wide consumer, once something has claimed the terminal.
static SINK: OnceLock<Box<dyn Fn(String) + Send + Sync>> = OnceLock::new();

/// Claim [`host_line`]'s output for this process. The first caller wins, so a
/// second server in one process keeps the first's consumer.
pub(crate) fn set_host_line_sink(sink: impl Fn(String) + Send + Sync + 'static) {
    let _ = SINK.set(Box::new(sink));
}

/// Report one line from code that sits below the server, without knowing who is
/// listening.
pub(crate) fn host_line(line: String) {
    match SINK.get() {
        Some(sink) => sink(line),
        None => eprintln!("{line}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

    /// An installed consumer receives the line verbatim, instead of stderr.
    /// The install is process-wide and permanent, so this is the only test that
    /// may claim it.
    #[test]
    fn an_installed_sink_receives_the_line() {
        set_host_line_sink(|line| CAPTURED.lock().unwrap().push(line));
        host_line("xwen: weights 63.5GB + KV 4.5GB = 68.0GB resident (max_ctx 4096)".to_string());
        assert_eq!(
            CAPTURED.lock().unwrap().as_slice(),
            ["xwen: weights 63.5GB + KV 4.5GB = 68.0GB resident (max_ctx 4096)"]
        );
    }
}
