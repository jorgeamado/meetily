#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use log;
use env_logger;

/// GUI-app stderr reaches neither the terminal nor the macOS unified log, so
/// mirror env_logger output to a file next to the app's data. Returns None
/// when the file can't be prepared (logging then stays on stderr).
#[cfg(target_os = "macos")]
fn log_file() -> Option<std::fs::File> {
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::Path::new(&home).join("Library/Application Support/com.meetily.ai/logs");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join("meetily.log");
    // Crude rotation: start over past 10 MB, keeping one previous file
    if std::fs::metadata(&path).map(|m| m.len() > 10 * 1024 * 1024).unwrap_or(false) {
        let _ = std::fs::rename(&path, dir.join("meetily.log.old"));
    }
    std::fs::OpenOptions::new().create(true).append(true).open(path).ok()
}

fn main() {
    std::env::set_var("RUST_LOG", "info");
    let mut builder = env_logger::Builder::from_default_env();
    #[cfg(target_os = "macos")]
    if let Some(file) = log_file() {
        builder.target(env_logger::Target::Pipe(Box::new(file)));
    }
    builder.init();

    // Async logger will be initialized lazily when first needed (after Tauri runtime starts)
    log::info!("Starting application...");
    app_lib::run();
}
