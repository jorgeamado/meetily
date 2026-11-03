// FluidAudio Parakeet (CoreML / Apple Neural Engine) speech-to-text sidecar client.
// Communicates with the `fluid-helper` Swift binary over line-delimited JSON on stdin/stdout.

pub mod commands;

use crate::audio::diarization::{build_wav_bytes, target_triple, WordSpan};
use anyhow::{anyhow, Context, Result};
use log::{debug, info, warn};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::Duration;
use tauri::{AppHandle, Manager, Runtime};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use uuid::Uuid;

const MACOS_ONLY_ERR: &str = "FluidAudio transcription is only available on macOS";
const LOAD_LINE_TIMEOUT: Duration = Duration::from_secs(600);
const STATUS_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_WAIT_TIMEOUT: Duration = Duration::from_secs(2);
const MIN_TRANSCRIBE_TIMEOUT_SECS: f64 = 60.0;
const TRANSCRIBE_DURATION_MULTIPLIER: f64 = 10.0;
const SAMPLE_RATE_HZ: f64 = 16000.0;
/// FluidAudio rejects audio shorter than 300 ms ("Must be at least 300ms of 16kHz audio").
const MIN_TRANSCRIBE_SAMPLES: usize = 4800;

/// Global FluidAudio engine instance shared across batch retranscription and audio import.
pub static FLUID_ENGINE: Lazy<Arc<FluidAudioEngine>> =
    Lazy::new(|| Arc::new_cyclic(|weak| FluidAudioEngine::with_weak(weak.clone())));

/// Supported FluidAudio Parakeet model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FluidModel {
    V3,
    Ultra,
}

impl Default for FluidModel {
    fn default() -> Self {
        Self::V3
    }
}

impl FluidModel {
    /// Parse a model identifier (`"fluid-parakeet-v3"`, `"fluid-parakeet-ultra"`)
    /// or bare wire version (`"v3"`, `"ultra"`).
    pub fn from_model_id(id: &str) -> Option<Self> {
        match id {
            "fluid-parakeet-v3" | "v3" => Some(Self::V3),
            "fluid-parakeet-ultra" | "ultra" => Some(Self::Ultra),
            _ => None,
        }
    }

    /// Canonical application model identifier stored in configuration.
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::V3 => "fluid-parakeet-v3",
            Self::Ultra => "fluid-parakeet-ultra",
        }
    }

    /// Wire protocol version string expected by `fluid-helper`.
    pub const fn wire_version(self) -> &'static str {
        match self {
            Self::V3 => "v3",
            Self::Ultra => "ultra",
        }
    }

    /// Human-readable display name.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::V3 => "Parakeet v3",
            Self::Ultra => "Parakeet Ultra",
        }
    }
}

impl std::fmt::Display for FluidModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.model_id())
    }
}

/// Progress stage reported by `fluid-helper` while loading a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FluidLoadStage {
    Download,
    Compile,
}

pub use FluidLoadStage::{Compile, Download};

/// Progress event emitted during `FluidAudioEngine::load_model`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidLoadProgress {
    pub stage: FluidLoadStage,
    pub fraction: f64,
}

/// Transcription output returned by `FluidAudioEngine::transcribe`.
#[derive(Debug, Clone)]
pub struct FluidTranscript {
    pub text: String,
    pub confidence: f32,
    pub words: Vec<WordSpan>,
}

/// On-disk model readiness status returned by `FluidAudioEngine::model_status`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FluidModelStatus {
    pub downloaded: bool,
    pub path: PathBuf,
}

// ============================================================================
// Wire Protocol Types
// ============================================================================

#[derive(Debug, Serialize)]
pub(crate) struct LoadRequest<'a> {
    #[serde(rename = "type")]
    pub req_type: &'static str,
    pub version: &'a str,
    pub models_dir: &'a str,
}

#[derive(Debug, Serialize)]
pub(crate) struct StatusRequest<'a> {
    #[serde(rename = "type")]
    pub req_type: &'static str,
    pub version: &'a str,
    pub models_dir: &'a str,
}

#[derive(Debug, Serialize)]
pub(crate) struct TranscribeRequest<'a> {
    #[serde(rename = "type")]
    pub req_type: &'static str,
    pub id: i64,
    pub wav_path: &'a str,
}

#[derive(Debug, Serialize)]
pub(crate) struct PingRequest {
    #[serde(rename = "type")]
    pub req_type: &'static str,
}

#[derive(Debug, Serialize)]
pub(crate) struct ShutdownRequest {
    #[serde(rename = "type")]
    pub req_type: &'static str,
}

/// Word-level timing item in a `fluid-helper` `result` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluidWord {
    pub w: String,
    pub start: f64,
    pub end: f64,
}

/// Token-level timing and confidence item in a `fluid-helper` `result` response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluidToken {
    pub t: String,
    pub start: f64,
    pub end: f64,
    #[serde(default = "default_token_confidence")]
    pub conf: f32,
}

fn default_token_confidence() -> f32 {
    1.0
}

fn deserialize_null_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    T: Default + Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

/// Outbound JSON line emitted by `fluid-helper` on stdout.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum HelperResponse {
    Progress {
        stage: FluidLoadStage,
        fraction: f64,
    },
    Loaded {
        #[serde(default)]
        version: Option<String>,
        #[serde(default)]
        load_secs: Option<f64>,
    },
    Status {
        #[serde(default)]
        version: Option<String>,
        downloaded: bool,
        path: String,
    },
    Result {
        id: i64,
        text: String,
        #[serde(default)]
        confidence: f32,
        #[serde(default)]
        duration: f64,
        #[serde(default)]
        processing_secs: f64,
        #[serde(default, deserialize_with = "deserialize_null_default")]
        words: Vec<FluidWord>,
        #[serde(default, deserialize_with = "deserialize_null_default")]
        tokens: Vec<FluidToken>,
    },
    Pong,
    Goodbye,
    Error {
        #[serde(default)]
        id: Option<i64>,
        message: String,
    },
}

// ============================================================================
// Token -> Word Merge
// ============================================================================

fn is_word_boundary_char(c: char) -> bool {
    c.is_whitespace() || c == '▁'
}

/// Merge decoded tokens from `fluid-helper` into word spans with millisecond timestamps.
///
/// - A token starting with whitespace or `▁` (U+2581) begins a new word; other tokens continue
///   the current word (the first non-empty token always begins the first word, with or without a
///   leading space).
/// - A token consisting only of whitespace or `▁` marks a word boundary and contributes no text.
/// - Every output `WordSpan.text` except the first carries one leading ASCII space so concatenating
///   `text` across words reproduces the sentence (`diarization::SpeakerTurn` convention).
/// - If `tokens` is empty, falls back to `fallback_words` with `prob = confidence`.
pub(crate) fn tokens_to_words(
    tokens: &[FluidToken],
    fallback_words: &[FluidWord],
    confidence: f32,
) -> Vec<WordSpan> {
    if tokens.is_empty() {
        return fallback_words
            .iter()
            .filter_map(|fw| {
                let clean = fw.w.replace('▁', " ");
                let trimmed = clean.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some((trimmed.to_string(), fw.start * 1000.0, fw.end * 1000.0))
                }
            })
            .enumerate()
            .map(|(i, (w, start_ms, end_ms))| WordSpan {
                text: if i == 0 { w } else { format!(" {}", w) },
                start_ms,
                end_ms,
                prob: confidence,
            })
            .collect();
    }

    let mut words: Vec<WordSpan> = Vec::new();
    let mut current: Option<WordSpan> = None;
    let mut at_boundary = true;

    for token in tokens {
        let starts_with_marker = token.t.starts_with(is_word_boundary_char);
        let stripped = token.t.trim_start_matches(is_word_boundary_char);

        // A whitespace-only or `▁`-only token marks a word boundary and contributes no text.
        if stripped.is_empty() {
            if starts_with_marker {
                if let Some(word) = current.take() {
                    words.push(word);
                }
                at_boundary = true;
            }
            continue;
        }

        let piece = stripped.replace('▁', " ");
        let start_ms = token.start * 1000.0;
        let end_ms = token.end * 1000.0;

        if at_boundary || starts_with_marker || current.is_none() {
            if let Some(word) = current.take() {
                words.push(word);
            }
            let text = if words.is_empty() {
                piece
            } else {
                format!(" {}", piece)
            };
            current = Some(WordSpan {
                text,
                start_ms,
                end_ms,
                prob: token.conf,
            });
            at_boundary = false;
        } else if let Some(word) = current.as_mut() {
            word.text.push_str(&piece);
            word.end_ms = end_ms;
            word.prob = word.prob.min(token.conf);
        }
    }

    if let Some(word) = current.take() {
        words.push(word);
    }

    words
}

// ============================================================================
// Binary & Directory Resolution
// ============================================================================

/// Resolve the FluidAudio models directory (`<app_data_dir>/models/fluidaudio`), creating it if needed.
pub fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf> {
    if !cfg!(target_os = "macos") {
        return Err(anyhow!(MACOS_ONLY_ERR));
    }
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    let dir = base.join("models").join("fluidaudio");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create FluidAudio models dir: {}", dir.display()))?;
    Ok(dir)
}

/// Resolve the `fluid-helper` sidecar binary path.
pub(crate) fn resolve_fluid_helper_binary() -> Result<PathBuf> {
    if !cfg!(target_os = "macos") {
        return Err(anyhow!(MACOS_ONLY_ERR));
    }

    if let Ok(env_path) = std::env::var("MEETILY_FLUID_HELPER") {
        if !env_path.is_empty() {
            let path = PathBuf::from(env_path);
            if path.exists() {
                info!(
                    "Using fluid-helper from MEETILY_FLUID_HELPER: {}",
                    path.display()
                );
                return Ok(path);
            }
        }
    }

    let triple = target_triple();
    let binary_name = if cfg!(windows) {
        format!("fluid-helper-{}.exe", triple)
    } else {
        format!("fluid-helper-{}", triple)
    };

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let bundled = exe_dir.join(&binary_name);
            if bundled.exists() {
                return Ok(bundled);
            }
            if let Ok(entries) = std::fs::read_dir(exe_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("fluid-helper") && !name.ends_with(".d") {
                            return Ok(path);
                        }
                    }
                }
            }
        }
    }

    if let Ok(resource_dir) = std::env::var("RESOURCE_DIR") {
        let resource_path = PathBuf::from(&resource_dir);
        let bundled = resource_path.join(&binary_name);
        if bundled.exists() {
            return Ok(bundled);
        }
        if let Ok(entries) = std::fs::read_dir(&resource_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("fluid-helper") && !name.ends_with(".d") {
                        return Ok(path);
                    }
                }
            }
        }
    }

    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidate = PathBuf::from(&manifest_dir)
            .join("binaries")
            .join(&binary_name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(anyhow!(
        "fluid-helper binary not found. Build with 'cd fluid-helper && swift build -c release --product fluid-helper' or set MEETILY_FLUID_HELPER env var."
    ))
}

/// Check whether FluidAudio transcription is available on this system (macOS and sidecar resolves).
pub fn is_available() -> bool {
    cfg!(target_os = "macos") && resolve_fluid_helper_binary().is_ok()
}

// ============================================================================
// Job Ownership & FluidAudioEngine
// ============================================================================

/// Tracks active batch jobs sharing `FluidAudioEngine` and the model they require.
#[derive(Debug, Default)]
pub(crate) struct FluidJobBookkeeping {
    active_jobs: usize,
    active_model: Option<FluidModel>,
}

impl FluidJobBookkeeping {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Verify that a job using `model` may start alongside any currently active jobs.
    pub(crate) fn check_can_start(&self, model: FluidModel) -> Result<()> {
        if self.active_jobs > 0 {
            if let Some(active) = self.active_model {
                if active != model {
                    return Err(anyhow!(
                        "FluidAudio is busy with {} in another job; try again when it finishes",
                        active.model_id()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Record a newly started job for `model`, failing if another model is currently active.
    pub(crate) fn begin(&mut self, model: FluidModel) -> Result<()> {
        self.check_can_start(model)?;
        self.active_jobs += 1;
        self.active_model = Some(model);
        Ok(())
    }

    /// Decrement the active-job count and clear the active model when the last job finishes.
    pub(crate) fn finish(&mut self) {
        self.active_jobs = self.active_jobs.saturating_sub(1);
        if self.active_jobs == 0 {
            self.active_model = None;
        }
    }

    pub(crate) fn active_jobs(&self) -> usize {
        self.active_jobs
    }

    pub(crate) fn active_model(&self) -> Option<FluidModel> {
        self.active_model
    }

    pub(crate) fn is_model_in_use(&self, model: FluidModel) -> bool {
        self.active_jobs > 0 && self.active_model == Some(model)
    }
}

/// RAII guard representing an active batch transcription job on `FluidAudioEngine`.
/// Holds an `Arc<FluidAudioEngine>`, exposes `transcribe`, and decrements the active-job count on drop.
pub struct FluidJobGuard {
    engine: Arc<FluidAudioEngine>,
}

impl FluidJobGuard {
    /// Transcribe 16 kHz mono f32 audio samples using the guarded FluidAudio job.
    pub async fn transcribe(&self, samples: &[f32]) -> Result<FluidTranscript> {
        self.engine.transcribe(samples).await
    }

    /// Access the underlying `FluidAudioEngine`.
    pub fn engine(&self) -> &Arc<FluidAudioEngine> {
        &self.engine
    }
}

impl Drop for FluidJobGuard {
    fn drop(&mut self) {
        let mut jobs = self
            .engine
            .inner
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        jobs.finish();
    }
}

struct HelperProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct TempWavGuard(PathBuf);

impl Drop for TempWavGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

enum TranscribeAttemptError {
    ProcessExited(anyhow::Error),
    Timeout(Duration),
    ProtocolError(anyhow::Error),
}

struct FluidEngineInner {
    process: Mutex<Option<HelperProcess>>,
    loaded_model: Mutex<Option<FluidModel>>,
    models_dir: Mutex<Option<PathBuf>>,
    jobs: StdMutex<FluidJobBookkeeping>,
    next_request_id: AtomicI64,
}

/// Persistent client for the `fluid-helper` sidecar process.
/// Serialises all requests behind a single async mutex so at most one CoreML/ANE request is in flight.
pub struct FluidAudioEngine {
    inner: Arc<FluidEngineInner>,
    self_weak: Weak<FluidAudioEngine>,
}

impl Default for FluidAudioEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FluidAudioEngine {
    pub fn new() -> Self {
        Self::with_weak(Weak::new())
    }

    fn with_weak(self_weak: Weak<FluidAudioEngine>) -> Self {
        Self {
            inner: Arc::new(FluidEngineInner {
                process: Mutex::new(None),
                loaded_model: Mutex::new(None),
                models_dir: Mutex::new(None),
                jobs: StdMutex::new(FluidJobBookkeeping::new()),
                next_request_id: AtomicI64::new(1),
            }),
            self_weak,
        }
    }

    fn as_arc(&self) -> Arc<FluidAudioEngine> {
        self.self_weak.upgrade().unwrap_or_else(|| {
            Arc::new(FluidAudioEngine {
                inner: Arc::clone(&self.inner),
                self_weak: Weak::new(),
            })
        })
    }

    /// Check whether FluidAudio transcription is available on this system.
    pub fn is_available() -> bool {
        is_available()
    }

    /// Return the number of currently active batch jobs on this engine.
    pub fn active_job_count(&self) -> usize {
        self.inner
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_jobs()
    }

    /// Check whether `model` is currently in use by one or more active batch jobs.
    pub fn is_model_in_use(&self, model: FluidModel) -> bool {
        self.inner
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_model_in_use(model)
    }

    fn spawn_helper() -> Result<HelperProcess> {
        if !cfg!(target_os = "macos") {
            return Err(anyhow!(MACOS_ONLY_ERR));
        }

        let binary = resolve_fluid_helper_binary()?;
        info!("Spawning fluid-helper sidecar: {}", binary.display());

        let mut command = Command::new(&binary);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command
            .spawn()
            .with_context(|| format!("Failed to spawn fluid-helper at {}", binary.display()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to capture fluid-helper stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture fluid-helper stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("Failed to capture fluid-helper stderr"))?;

        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                debug!("fluid-helper: {}", line);
            }
        });

        Ok(HelperProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    async fn clear_state_and_kill(&self, proc_guard: &mut Option<HelperProcess>) {
        if let Some(mut proc) = proc_guard.take() {
            let _ = proc.child.kill().await;
        }
        *self.inner.loaded_model.lock().await = None;
        *self.inner.models_dir.lock().await = None;
    }

    async fn send_load_command(
        proc: &mut HelperProcess,
        model: FluidModel,
        models_dir: &Path,
        on_progress: &(impl Fn(FluidLoadProgress) + Send + Sync),
    ) -> std::result::Result<(), (bool, anyhow::Error)> {
        let models_dir_str = models_dir.to_string_lossy();
        let req = LoadRequest {
            req_type: "load",
            version: model.wire_version(),
            models_dir: &models_dir_str,
        };
        let mut payload = serde_json::to_string(&req)
            .map_err(|e| (false, anyhow!("Failed to serialize load request: {}", e)))?;
        payload.push('\n');

        proc.stdin
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| (true, anyhow!("Failed to write load request to fluid-helper: {}", e)))?;
        proc.stdin
            .flush()
            .await
            .map_err(|e| (true, anyhow!("Failed to flush fluid-helper stdin: {}", e)))?;

        let mut line = String::new();
        loop {
            line.clear();
            match tokio::time::timeout(LOAD_LINE_TIMEOUT, proc.stdout.read_line(&mut line)).await {
                Ok(Ok(0)) => {
                    return Err((
                        true,
                        anyhow!("fluid-helper exited unexpectedly (EOF) while loading model '{}'", model.model_id()),
                    ));
                }
                Ok(Ok(_)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<HelperResponse>(trimmed) {
                        Ok(HelperResponse::Progress { stage, fraction }) => {
                            on_progress(FluidLoadProgress { stage, fraction });
                        }
                        Ok(HelperResponse::Loaded { load_secs, .. }) => {
                            info!(
                                "FluidAudio model '{}' loaded in {:.2}s",
                                model.model_id(),
                                load_secs.unwrap_or(0.0)
                            );
                            return Ok(());
                        }
                        Ok(HelperResponse::Error { message, .. }) => {
                            return Err((
                                false,
                                anyhow!(
                                    "fluid-helper failed to load model '{}': {}",
                                    model.model_id(),
                                    message
                                ),
                            ));
                        }
                        Ok(other) => {
                            debug!(
                                "Ignoring unexpected fluid-helper response during load: {:?}",
                                other
                            );
                        }
                        Err(e) => {
                            debug!(
                                "Ignoring malformed fluid-helper line during load ({}): {}",
                                e, trimmed
                            );
                        }
                    }
                }
                Ok(Err(e)) => {
                    return Err((
                        true,
                        anyhow!("Failed reading from fluid-helper stdout during load: {}", e),
                    ));
                }
                Err(_) => {
                    return Err((
                        true,
                        anyhow!(
                            "Timed out after {:?} waiting for fluid-helper to load model '{}'",
                            LOAD_LINE_TIMEOUT,
                            model.model_id()
                        ),
                    ));
                }
            }
        }
    }

    async fn load_model_locked(
        &self,
        proc_guard: &mut Option<HelperProcess>,
        model: FluidModel,
        models_dir: &Path,
        on_progress: &(impl Fn(FluidLoadProgress) + Send + Sync),
    ) -> Result<()> {
        if let Some(proc) = proc_guard.as_mut() {
            match proc.child.try_wait() {
                Ok(None) => {
                    let current_model = *self.inner.loaded_model.lock().await;
                    let current_dir = self.inner.models_dir.lock().await.clone();
                    if current_model == Some(model)
                        && current_dir.as_deref() == Some(models_dir)
                    {
                        debug!(
                            "FluidAudio model '{}' already loaded in live helper",
                            model.model_id()
                        );
                        return Ok(());
                    }
                }
                Ok(Some(status)) => {
                    info!(
                        "fluid-helper had exited with status {}; respawning for load",
                        status
                    );
                    *proc_guard = None;
                    *self.inner.loaded_model.lock().await = None;
                }
                Err(e) => {
                    warn!("Failed to check fluid-helper status ({}); respawning", e);
                    *proc_guard = None;
                    *self.inner.loaded_model.lock().await = None;
                }
            }
        }

        std::fs::create_dir_all(models_dir).with_context(|| {
            format!(
                "Failed to create FluidAudio models dir: {}",
                models_dir.display()
            )
        })?;

        if proc_guard.is_none() {
            *proc_guard = Some(Self::spawn_helper()?);
        }

        let proc = proc_guard.as_mut().expect("helper process spawned above");
        match Self::send_load_command(proc, model, models_dir, on_progress).await {
            Ok(()) => {
                *self.inner.loaded_model.lock().await = Some(model);
                *self.inner.models_dir.lock().await = Some(models_dir.to_path_buf());
                Ok(())
            }
            Err((kill_and_clear, err)) => {
                if kill_and_clear {
                    self.clear_state_and_kill(proc_guard).await;
                } else {
                    *self.inner.loaded_model.lock().await = None;
                    *self.inner.models_dir.lock().await = None;
                }
                Err(err)
            }
        }
    }

    /// Ensure `fluid-helper` is running and has `model` loaded from `models_dir`.
    /// No-op if the same model is already loaded in a live helper process.
    pub async fn load_model(
        &self,
        model: FluidModel,
        models_dir: &Path,
        on_progress: impl Fn(FluidLoadProgress) + Send + Sync,
    ) -> Result<()> {
        if !cfg!(target_os = "macos") {
            return Err(anyhow!(MACOS_ONLY_ERR));
        }

        let mut proc_guard = self.inner.process.lock().await;
        {
            let jobs = self
                .inner
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            jobs.check_can_start(model)?;
        }

        self.load_model_locked(&mut proc_guard, model, models_dir, &on_progress)
            .await
    }

    /// Begin a batch transcription job using `model`.
    /// Loads `model` (if not already loaded) and increments the active-job count.
    /// Returns an error if another job is currently active with a different model.
    pub async fn begin_job(
        &self,
        model: FluidModel,
        models_dir: &Path,
        on_progress: impl Fn(FluidLoadProgress) + Send + Sync,
    ) -> Result<FluidJobGuard> {
        if !cfg!(target_os = "macos") {
            return Err(anyhow!(MACOS_ONLY_ERR));
        }

        let mut proc_guard = self.inner.process.lock().await;
        {
            let jobs = self
                .inner
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            jobs.check_can_start(model)?;
        }

        self.load_model_locked(&mut proc_guard, model, models_dir, &on_progress)
            .await?;

        {
            let mut jobs = self
                .inner
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            jobs.begin(model)?;
        }

        Ok(FluidJobGuard {
            engine: self.as_arc(),
        })
    }

    async fn send_status_command(
        proc: &mut HelperProcess,
        model: FluidModel,
        models_dir: &Path,
    ) -> std::result::Result<FluidModelStatus, (bool, anyhow::Error)> {
        let models_dir_str = models_dir.to_string_lossy();
        let req = StatusRequest {
            req_type: "status",
            version: model.wire_version(),
            models_dir: &models_dir_str,
        };
        let mut payload = serde_json::to_string(&req)
            .map_err(|e| (false, anyhow!("Failed to serialize status request: {}", e)))?;
        payload.push('\n');

        let io_fut = async {
            proc.stdin
                .write_all(payload.as_bytes())
                .await
                .map_err(|e| {
                    (
                        true,
                        anyhow!("Failed to write status request to fluid-helper: {}", e),
                    )
                })?;
            proc.stdin
                .flush()
                .await
                .map_err(|e| (true, anyhow!("Failed to flush fluid-helper stdin: {}", e)))?;

            let mut line = String::new();
            loop {
                line.clear();
                let bytes_read = proc.stdout.read_line(&mut line).await.map_err(|e| {
                    (
                        true,
                        anyhow!(
                            "Failed reading from fluid-helper stdout during status: {}",
                            e
                        ),
                    )
                })?;
                if bytes_read == 0 {
                    return Err((
                        true,
                        anyhow!(
                            "fluid-helper exited unexpectedly (EOF) while checking status for '{}'",
                            model.model_id()
                        ),
                    ));
                }

                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                match serde_json::from_str::<HelperResponse>(trimmed) {
                    Ok(HelperResponse::Status {
                        downloaded, path, ..
                    }) => {
                        return Ok(FluidModelStatus {
                            downloaded,
                            path: PathBuf::from(path),
                        });
                    }
                    Ok(HelperResponse::Error { message, .. }) => {
                        return Err((
                            false,
                            anyhow!(
                                "fluid-helper failed to check status for model '{}': {}",
                                model.model_id(),
                                message
                            ),
                        ));
                    }
                    Ok(other) => {
                        debug!(
                            "Ignoring unexpected fluid-helper response during status: {:?}",
                            other
                        );
                    }
                    Err(e) => {
                        debug!(
                            "Ignoring malformed fluid-helper line during status ({}): {}",
                            e, trimmed
                        );
                    }
                }
            }
        };

        match tokio::time::timeout(STATUS_TIMEOUT, io_fut).await {
            Ok(res) => res,
            Err(_) => Err((
                true,
                anyhow!(
                    "Timed out after {:?} waiting for fluid-helper status of '{}'",
                    STATUS_TIMEOUT,
                    model.model_id()
                ),
            )),
        }
    }

    async fn model_status_locked(
        &self,
        proc_guard: &mut Option<HelperProcess>,
        model: FluidModel,
        models_dir: &Path,
    ) -> Result<FluidModelStatus> {
        if !cfg!(target_os = "macos") {
            return Err(anyhow!(MACOS_ONLY_ERR));
        }

        if let Some(proc) = proc_guard.as_mut() {
            match proc.child.try_wait() {
                Ok(None) => {}
                Ok(Some(status)) => {
                    info!(
                        "fluid-helper had exited with status {}; respawning for status",
                        status
                    );
                    *proc_guard = None;
                    *self.inner.loaded_model.lock().await = None;
                    *self.inner.models_dir.lock().await = None;
                }
                Err(e) => {
                    warn!("Failed to check fluid-helper status ({}); respawning", e);
                    *proc_guard = None;
                    *self.inner.loaded_model.lock().await = None;
                    *self.inner.models_dir.lock().await = None;
                }
            }
        }

        std::fs::create_dir_all(models_dir).with_context(|| {
            format!(
                "Failed to create FluidAudio models dir: {}",
                models_dir.display()
            )
        })?;

        if proc_guard.is_none() {
            *proc_guard = Some(Self::spawn_helper()?);
        }

        let proc = proc_guard.as_mut().expect("helper process spawned above");
        match Self::send_status_command(proc, model, models_dir).await {
            Ok(status) => Ok(status),
            Err((kill_and_clear, err)) => {
                if kill_and_clear {
                    self.clear_state_and_kill(proc_guard).await;
                }
                Err(err)
            }
        }
    }

    /// Query `fluid-helper` for on-disk readiness and resolved path of `model` under `models_dir`.
    /// Spawns `fluid-helper` if it is not already running, without loading any model.
    pub async fn model_status(
        &self,
        model: FluidModel,
        models_dir: &Path,
    ) -> Result<FluidModelStatus> {
        if !cfg!(target_os = "macos") {
            return Err(anyhow!(MACOS_ONLY_ERR));
        }

        let mut proc_guard = self.inner.process.lock().await;
        self.model_status_locked(&mut proc_guard, model, models_dir)
            .await
    }

    /// Atomically check that `model` is not in use by an active job, shut down the helper if `model`
    /// is currently loaded and idle, query its on-disk path via `status`, run `verify` on the path,
    /// and delete the directory — all while holding the process mutex so no job can start in between.
    pub async fn delete_model_files(
        &self,
        model: FluidModel,
        models_dir: &Path,
        verify: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        let mut proc_guard = self.inner.process.lock().await;

        let active_jobs = {
            let jobs = self
                .inner
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if jobs.is_model_in_use(model) {
                return Err(anyhow!(
                    "Cannot delete '{}' while it is in use by an active transcription job",
                    model.model_id()
                ));
            }
            jobs.active_jobs()
        };

        if !cfg!(target_os = "macos") {
            return Err(anyhow!(MACOS_ONLY_ERR));
        }

        let is_loaded_model = *self.inner.loaded_model.lock().await == Some(model);
        if is_loaded_model && active_jobs == 0 {
            self.shutdown_locked(&mut proc_guard).await;
        }

        let status = self
            .model_status_locked(&mut proc_guard, model, models_dir)
            .await
            .map_err(|e| anyhow!("Failed to inspect model '{}': {}", model.model_id(), e))?;

        verify(&status.path)?;

        if status.path.exists() {
            std::fs::remove_dir_all(&status.path).map_err(|e| {
                anyhow!(
                    "Failed to remove FluidAudio model directory '{}': {}",
                    status.path.display(),
                    e
                )
            })?;
            info!(
                "Deleted FluidAudio model '{}' at '{}'",
                model.model_id(),
                status.path.display()
            );
        } else {
            info!(
                "FluidAudio model '{}' directory '{}' was already absent",
                model.model_id(),
                status.path.display()
            );
        }

        Ok(())
    }

    async fn run_transcribe_attempt(
        proc: &mut HelperProcess,
        req_id: i64,
        wav_path: &Path,
        timeout_duration: Duration,
    ) -> std::result::Result<FluidTranscript, TranscribeAttemptError> {
        let wav_path_str = wav_path.to_string_lossy();
        let req = TranscribeRequest {
            req_type: "transcribe",
            id: req_id,
            wav_path: &wav_path_str,
        };
        let mut payload = serde_json::to_string(&req).map_err(|e| {
            TranscribeAttemptError::ProtocolError(anyhow!(
                "Failed to serialize transcribe request: {}",
                e
            ))
        })?;
        payload.push('\n');

        let io_fut = async {
            proc.stdin
                .write_all(payload.as_bytes())
                .await
                .map_err(|e| {
                    TranscribeAttemptError::ProcessExited(anyhow!(
                        "Failed writing transcribe request to fluid-helper: {}",
                        e
                    ))
                })?;
            proc.stdin.flush().await.map_err(|e| {
                TranscribeAttemptError::ProcessExited(anyhow!(
                    "Failed flushing fluid-helper stdin: {}",
                    e
                ))
            })?;

            let mut line = String::new();
            loop {
                line.clear();
                let bytes_read = proc.stdout.read_line(&mut line).await.map_err(|e| {
                    TranscribeAttemptError::ProcessExited(anyhow!(
                        "Failed reading from fluid-helper stdout: {}",
                        e
                    ))
                })?;
                if bytes_read == 0 {
                    return Err(TranscribeAttemptError::ProcessExited(anyhow!(
                        "fluid-helper closed stdout (EOF) during transcribe request {}",
                        req_id
                    )));
                }

                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                match serde_json::from_str::<HelperResponse>(trimmed) {
                    Ok(HelperResponse::Result {
                        id,
                        text,
                        confidence,
                        words,
                        tokens,
                        ..
                    }) if id == req_id => {
                        let word_spans = tokens_to_words(&tokens, &words, confidence);
                        return Ok(FluidTranscript {
                            text,
                            confidence,
                            words: word_spans,
                        });
                    }
                    Ok(HelperResponse::Error {
                        id: Some(id),
                        message,
                    }) if id == req_id => {
                        return Err(TranscribeAttemptError::ProtocolError(anyhow!(
                            "fluid-helper transcription error (id {}): {}",
                            id,
                            message
                        )));
                    }
                    Ok(other) => {
                        debug!(
                            "Ignoring fluid-helper message while awaiting id {}: {:?}",
                            req_id, other
                        );
                    }
                    Err(e) => {
                        debug!(
                            "Ignoring non-matching fluid-helper line while awaiting id {} ({}): {}",
                            req_id, e, trimmed
                        );
                    }
                }
            }
        };

        match tokio::time::timeout(timeout_duration, io_fut).await {
            Ok(res) => res,
            Err(_) => Err(TranscribeAttemptError::Timeout(timeout_duration)),
        }
    }

    async fn respawn_and_reload(
        proc_guard: &mut Option<HelperProcess>,
        model: FluidModel,
        models_dir: &Path,
    ) -> Result<()> {
        if let Some(mut old_proc) = proc_guard.take() {
            let _ = old_proc.child.kill().await;
        }
        let mut new_proc = Self::spawn_helper()?;
        if let Err((_, err)) = Self::send_load_command(&mut new_proc, model, models_dir, &|_| {}).await
        {
            let _ = new_proc.child.kill().await;
            return Err(err);
        }
        *proc_guard = Some(new_proc);
        Ok(())
    }

    /// Transcribe 16 kHz mono f32 audio samples using the currently loaded FluidAudio model.
    /// Word timestamps in the returned `FluidTranscript` are in milliseconds relative to the start of `samples`.
    pub async fn transcribe(&self, samples: &[f32]) -> Result<FluidTranscript> {
        if !cfg!(target_os = "macos") {
            return Err(anyhow!(MACOS_ONLY_ERR));
        }

        let mut proc_guard = self.inner.process.lock().await;

        let model = (*self.inner.loaded_model.lock().await)
            .ok_or_else(|| anyhow!("No FluidAudio model is loaded; call load_model first"))?;
        let models_dir = self
            .inner
            .models_dir
            .lock()
            .await
            .clone()
            .ok_or_else(|| anyhow!("No FluidAudio models directory configured"))?;

        let samples = pad_to_min_transcribe_len(samples);
        let samples = samples.as_ref();
        let wav_bytes = build_wav_bytes(samples);
        let temp_path = std::env::temp_dir().join(format!("meetily-fluid-{}.wav", Uuid::new_v4()));
        let _temp_guard = TempWavGuard(temp_path.clone());

        tokio::fs::write(&temp_path, &wav_bytes)
            .await
            .with_context(|| format!("Failed to write temp FluidAudio WAV: {}", temp_path.display()))?;

        let audio_secs = samples.len() as f64 / SAMPLE_RATE_HZ;
        let timeout_duration = Duration::from_secs_f64(
            (audio_secs * TRANSCRIBE_DURATION_MULTIPLIER).max(MIN_TRANSCRIBE_TIMEOUT_SECS),
        );

        // Pre-flight check: if the helper process has already exited (e.g. after 300s idle timeout),
        // respawn it now without consuming the one in-request retry budget.
        let needs_preflight_respawn = match proc_guard.as_mut() {
            Some(proc) => match proc.child.try_wait() {
                Ok(None) => false,
                Ok(Some(status)) => {
                    info!(
                        "fluid-helper exited before transcribe (status {}); respawning and reloading",
                        status
                    );
                    true
                }
                Err(e) => {
                    warn!(
                        "Failed to poll fluid-helper status ({}); respawning and reloading",
                        e
                    );
                    true
                }
            },
            None => true,
        };

        if needs_preflight_respawn {
            if let Err(e) = Self::respawn_and_reload(&mut proc_guard, model, &models_dir).await {
                self.clear_state_and_kill(&mut proc_guard).await;
                let _ = tokio::fs::remove_file(&temp_path).await;
                return Err(e);
            }
        }

        let mut retried_in_request = false;
        loop {
            let req_id = self.inner.next_request_id.fetch_add(1, Ordering::SeqCst);
            let proc = proc_guard.as_mut().expect("helper process must be present");

            match Self::run_transcribe_attempt(proc, req_id, &temp_path, timeout_duration).await {
                Ok(transcript) => {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Ok(transcript);
                }
                Err(TranscribeAttemptError::ProcessExited(err)) if !retried_in_request => {
                    warn!(
                        "fluid-helper exited or broke pipe during transcribe ({}); respawning and retrying once",
                        err
                    );
                    if let Err(reload_err) =
                        Self::respawn_and_reload(&mut proc_guard, model, &models_dir).await
                    {
                        self.clear_state_and_kill(&mut proc_guard).await;
                        let _ = tokio::fs::remove_file(&temp_path).await;
                        return Err(reload_err);
                    }
                    retried_in_request = true;
                }
                Err(TranscribeAttemptError::ProcessExited(err)) => {
                    self.clear_state_and_kill(&mut proc_guard).await;
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Err(err);
                }
                Err(TranscribeAttemptError::Timeout(timeout)) => {
                    warn!(
                        "fluid-helper transcription timed out after {:?}; killing process and clearing state",
                        timeout
                    );
                    self.clear_state_and_kill(&mut proc_guard).await;
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Err(anyhow!(
                        "FluidAudio transcription timed out after {:.1}s",
                        timeout.as_secs_f64()
                    ));
                }
                Err(TranscribeAttemptError::ProtocolError(err)) => {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Err(err);
                }
            }
        }
    }

    async fn shutdown_locked(&self, proc_guard: &mut Option<HelperProcess>) {
        *self.inner.loaded_model.lock().await = None;
        *self.inner.models_dir.lock().await = None;

        if let Some(mut proc) = proc_guard.take() {
            info!("Shutting down fluid-helper sidecar");
            let req = ShutdownRequest {
                req_type: "shutdown",
            };
            if let Ok(mut payload) = serde_json::to_string(&req) {
                payload.push('\n');
                let _ = proc.stdin.write_all(payload.as_bytes()).await;
                let _ = proc.stdin.flush().await;
            }

            match tokio::time::timeout(SHUTDOWN_WAIT_TIMEOUT, proc.child.wait()).await {
                Ok(Ok(status)) => {
                    debug!("fluid-helper exited with status {}", status);
                }
                Ok(Err(e)) => {
                    warn!("Failed waiting for fluid-helper exit ({}), killing", e);
                    let _ = proc.child.kill().await;
                }
                Err(_) => {
                    warn!(
                        "fluid-helper did not exit within {:?}, killing",
                        SHUTDOWN_WAIT_TIMEOUT
                    );
                    let _ = proc.child.kill().await;
                }
            }
        }
    }

    /// Shut down the `fluid-helper` sidecar only when no batch jobs are active.
    /// Checks the active-job count under the process mutex so a job cannot start
    /// between the check and the shutdown.
    pub async fn unload_if_idle(&self) {
        let mut proc_guard = self.inner.process.lock().await;
        let active = {
            let jobs = self
                .inner
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            jobs.active_jobs()
        };
        if active > 0 {
            info!(
                "Skipping FluidAudio unload: {} batch job(s) still active",
                active
            );
            return;
        }
        self.shutdown_locked(&mut proc_guard).await;
    }

    /// Gracefully shut down the `fluid-helper` sidecar unconditionally and clear the loaded model state.
    pub async fn unload_model(&self) {
        let mut proc_guard = self.inner.process.lock().await;
        self.shutdown_locked(&mut proc_guard).await;
    }

    /// Check if a model is currently loaded.
    pub async fn is_model_loaded(&self) -> bool {
        self.inner.loaded_model.lock().await.is_some()
    }

    /// Return the canonical model identifier of the currently loaded model, if any.
    pub async fn get_current_model(&self) -> Option<String> {
        self.inner
            .loaded_model
            .lock()
            .await
            .map(|m| m.model_id().to_string())
    }
}

#[async_trait::async_trait]
impl crate::audio::transcription::provider::TranscriptionProvider for FluidAudioEngine {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        _language: Option<String>,
    ) -> std::result::Result<
        crate::audio::transcription::provider::TranscriptResult,
        crate::audio::transcription::provider::TranscriptionError,
    > {
        use crate::audio::transcription::provider::{TranscriptResult, TranscriptionError};

        if !self.is_model_loaded().await {
            return Err(TranscriptionError::ModelNotLoaded);
        }

        let transcript = FluidAudioEngine::transcribe(self, &audio)
            .await
            .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))?;

        Ok(TranscriptResult {
            text: transcript.text,
            confidence: Some(transcript.confidence),
            is_partial: false,
            words: Some(transcript.words),
        })
    }

    async fn is_model_loaded(&self) -> bool {
        FluidAudioEngine::is_model_loaded(self).await
    }

    async fn get_current_model(&self) -> Option<String> {
        FluidAudioEngine::get_current_model(self).await
    }

    fn provider_name(&self) -> &'static str {
        "FluidAudio Parakeet"
    }
}

// ============================================================================
// Tests
// ============================================================================

/// Pad short clips with trailing silence up to FluidAudio's minimum length. Trailing
/// padding leaves the word timestamps (relative to the clip start) unchanged.
fn pad_to_min_transcribe_len(samples: &[f32]) -> std::borrow::Cow<'_, [f32]> {
    if samples.len() >= MIN_TRANSCRIBE_SAMPLES {
        return std::borrow::Cow::Borrowed(samples);
    }
    let mut padded = Vec::with_capacity(MIN_TRANSCRIBE_SAMPLES);
    padded.extend_from_slice(samples);
    padded.resize(MIN_TRANSCRIBE_SAMPLES, 0.0);
    std::borrow::Cow::Owned(padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_clips_are_padded_to_fluidaudio_minimum() {
        let short = vec![0.5f32; 1600];
        let padded = pad_to_min_transcribe_len(&short);
        assert_eq!(padded.len(), MIN_TRANSCRIBE_SAMPLES);
        assert_eq!(&padded[..1600], &short[..]);
        assert!(padded[1600..].iter().all(|&x| x == 0.0));

        let long = vec![0.5f32; MIN_TRANSCRIBE_SAMPLES + 1];
        assert!(matches!(
            pad_to_min_transcribe_len(&long),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn parses_fluid_model_ids_and_wire_versions() {
        assert_eq!(FluidModel::default(), FluidModel::V3);

        assert_eq!(
            FluidModel::from_model_id("fluid-parakeet-v3"),
            Some(FluidModel::V3)
        );
        assert_eq!(FluidModel::from_model_id("v3"), Some(FluidModel::V3));
        assert_eq!(
            FluidModel::from_model_id("fluid-parakeet-ultra"),
            Some(FluidModel::Ultra)
        );
        assert_eq!(FluidModel::from_model_id("ultra"), Some(FluidModel::Ultra));

        assert_eq!(FluidModel::from_model_id(""), None);
        assert_eq!(FluidModel::from_model_id("parakeet-tdt-0.6b-v3-int8"), None);
        assert_eq!(FluidModel::from_model_id("V3"), None);

        assert_eq!(FluidModel::V3.model_id(), "fluid-parakeet-v3");
        assert_eq!(FluidModel::V3.wire_version(), "v3");
        assert_eq!(FluidModel::V3.display_name(), "Parakeet v3");

        assert_eq!(FluidModel::Ultra.model_id(), "fluid-parakeet-ultra");
        assert_eq!(FluidModel::Ultra.wire_version(), "ultra");
        assert_eq!(FluidModel::Ultra.display_name(), "Parakeet Ultra");
    }

    #[test]
    fn deserializes_all_readme_response_types_and_ignores_unknown_fields() {
        // 1. progress (download and compile)
        let dl_raw = r#"{"type":"progress","stage":"download","fraction":0.45,"extra_field":"ignored"}"#;
        let dl: HelperResponse = serde_json::from_str(dl_raw).unwrap();
        assert_eq!(
            dl,
            HelperResponse::Progress {
                stage: FluidLoadStage::Download,
                fraction: 0.45,
            }
        );

        let cmp_raw = r#"{"type":"progress","stage":"compile","fraction":0.75}"#;
        let cmp: HelperResponse = serde_json::from_str(cmp_raw).unwrap();
        assert_eq!(
            cmp,
            HelperResponse::Progress {
                stage: FluidLoadStage::Compile,
                fraction: 0.75,
            }
        );

        // 2. loaded
        let loaded_raw = r#"{"type":"loaded","version":"v3","load_secs":12.3,"unknown":123}"#;
        let loaded: HelperResponse = serde_json::from_str(loaded_raw).unwrap();
        assert_eq!(
            loaded,
            HelperResponse::Loaded {
                version: Some("v3".to_string()),
                load_secs: Some(12.3),
            }
        );

        // 2b. status
        let status_raw = r#"{"type":"status","version":"v3","downloaded":true,"path":"/abs/models/parakeet-tdt-0.6b-v3","extra":1}"#;
        let status: HelperResponse = serde_json::from_str(status_raw).unwrap();
        assert_eq!(
            status,
            HelperResponse::Status {
                version: Some("v3".to_string()),
                downloaded: true,
                path: "/abs/models/parakeet-tdt-0.6b-v3".to_string(),
            }
        );

        let status_missing_raw = r#"{"type":"status","version":"ultra","downloaded":false,"path":"/abs/models/parakeet-ultra"}"#;
        let status_missing: HelperResponse = serde_json::from_str(status_missing_raw).unwrap();
        assert_eq!(
            status_missing,
            HelperResponse::Status {
                version: Some("ultra".to_string()),
                downloaded: false,
                path: "/abs/models/parakeet-ultra".to_string(),
            }
        );

        // 3. result
        let result_raw = r#"{"type":"result","id":7,"text":"...","confidence":0.93,"duration":120.1,"processing_secs":1.6,"words":[{"w":"forecast","start":12.34,"end":12.8,"extra":true}],"tokens":[{"t":" fore","start":12.34,"end":12.5,"conf":0.98,"extra":1}],"unexpected":"ok"}"#;
        let res: HelperResponse = serde_json::from_str(result_raw).unwrap();
        match res {
            HelperResponse::Result {
                id,
                text,
                confidence,
                duration,
                processing_secs,
                words,
                tokens,
            } => {
                assert_eq!(id, 7);
                assert_eq!(text, "...");
                assert!((confidence - 0.93).abs() < 1e-5);
                assert!((duration - 120.1).abs() < 1e-6);
                assert!((processing_secs - 1.6).abs() < 1e-6);
                assert_eq!(
                    words,
                    vec![FluidWord {
                        w: "forecast".to_string(),
                        start: 12.34,
                        end: 12.8,
                    }]
                );
                assert_eq!(
                    tokens,
                    vec![FluidToken {
                        t: " fore".to_string(),
                        start: 12.34,
                        end: 12.5,
                        conf: 0.98,
                    }]
                );
            }
            other => panic!("Expected Result, got {:?}", other),
        }

        // 4. pong & goodbye
        let pong: HelperResponse =
            serde_json::from_str(r#"{"type":"pong","extra":false}"#).unwrap();
        assert_eq!(pong, HelperResponse::Pong);

        let goodbye: HelperResponse =
            serde_json::from_str(r#"{"type":"goodbye","extra":null}"#).unwrap();
        assert_eq!(goodbye, HelperResponse::Goodbye);

        // 5. error with id and with id: null
        let err_with_id: HelperResponse =
            serde_json::from_str(r#"{"type":"error","id":7,"message":"...","code":"E"}"#).unwrap();
        assert_eq!(
            err_with_id,
            HelperResponse::Error {
                id: Some(7),
                message: "...".to_string(),
            }
        );

        let err_null_id: HelperResponse =
            serde_json::from_str(r#"{"type":"error","id":null,"message":"Malformed JSON"}"#)
                .unwrap();
        assert_eq!(
            err_null_id,
            HelperResponse::Error {
                id: None,
                message: "Malformed JSON".to_string(),
            }
        );
    }

    fn tok(t: &str, start: f64, end: f64, conf: f32) -> FluidToken {
        FluidToken {
            t: t.to_string(),
            start,
            end,
            conf,
        }
    }

    #[test]
    fn tokens_to_words_merges_25_token_code_switched_example() {
        // Synthetic sentence shaped like real fluid-helper output for Cyrillic + Latin
        // code-switched speech (25 tokens, leading space starts a word):
        // [' Это', ' по', 'ка', 'зы', 'вает', ' bu', 'ild', ' I', 'D', '.', ' М', 'ы',
        //  ' ви', 'д', 'и', 'м', ' но', 'вы', 'й', ' I', 'D', ',', ' bu', 'ild', ' I']
        let tokens = vec![
            tok(" Это", 0.00, 0.24, 0.96),
            tok(" по", 0.24, 0.32, 0.95),
            tok("ка", 0.32, 0.40, 0.82),
            tok("зы", 0.40, 0.48, 0.91),
            tok("вает", 0.48, 0.64, 0.89),
            tok(" bu", 0.68, 0.80, 0.94),
            tok("ild", 0.80, 0.92, 0.77),
            tok(" I", 0.96, 1.04, 0.93),
            tok("D", 1.04, 1.12, 0.85),
            tok(".", 1.12, 1.16, 0.65),
            tok(" М", 1.30, 1.38, 0.97),
            tok("ы", 1.38, 1.46, 0.88),
            tok(" ви", 1.50, 1.58, 0.92),
            tok("д", 1.58, 1.66, 0.84),
            tok("и", 1.66, 1.72, 0.71),
            tok("м", 1.72, 1.84, 0.90),
            tok(" но", 1.88, 1.96, 0.95),
            tok("вы", 1.96, 2.04, 0.86),
            tok("й", 2.04, 2.16, 0.79),
            tok(" I", 2.20, 2.28, 0.91),
            tok("D", 2.28, 2.36, 0.83),
            tok(",", 2.36, 2.40, 0.60),
            tok(" bu", 2.48, 2.60, 0.87),
            tok("ild", 2.60, 2.72, 0.94),
            tok(" I", 2.76, 2.84, 0.73),
        ];

        let words = tokens_to_words(&tokens, &[], 0.9);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "Это",
                " показывает",
                " build",
                " ID.",
                " Мы",
                " видим",
                " новый",
                " ID,",
                " build",
                " I",
            ]
        );

        let expected: &[(f64, f64, f32)] = &[
            (0.0, 240.0, 0.96),
            (240.0, 640.0, 0.82),
            (680.0, 920.0, 0.77),
            (960.0, 1160.0, 0.65),
            (1300.0, 1460.0, 0.88),
            (1500.0, 1840.0, 0.71),
            (1880.0, 2160.0, 0.79),
            (2200.0, 2400.0, 0.60),
            (2480.0, 2720.0, 0.87),
            (2760.0, 2840.0, 0.73),
        ];
        for (i, (start_ms, end_ms, prob)) in expected.iter().enumerate() {
            assert!(
                (words[i].start_ms - start_ms).abs() < 1e-5,
                "word {} start_ms: got {}, expected {}",
                i,
                words[i].start_ms,
                start_ms
            );
            assert!(
                (words[i].end_ms - end_ms).abs() < 1e-5,
                "word {} end_ms: got {}, expected {}",
                i,
                words[i].end_ms,
                end_ms
            );
            assert!(
                (words[i].prob - prob).abs() < 1e-5,
                "word {} prob: got {}, expected {}",
                i,
                words[i].prob,
                prob
            );
        }

        let sentence: String = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(
            sentence,
            "Это показывает build ID. Мы видим новый ID, build I"
        );
    }

    #[test]
    fn tokens_to_words_handles_first_token_without_space_and_punctuation_attachment() {
        let tokens = vec![
            tok("Hello", 0.0, 0.3, 0.95),
            tok(",", 0.3, 0.35, 0.70),
            tok(" world", 0.4, 0.7, 0.88),
            tok("!", 0.7, 0.75, 0.99),
        ];
        let words = tokens_to_words(&tokens, &[], 0.9);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello,");
        assert!((words[0].start_ms - 0.0).abs() < 1e-6);
        assert!((words[0].end_ms - 350.0).abs() < 1e-6);
        assert!((words[0].prob - 0.70).abs() < 1e-6);

        assert_eq!(words[1].text, " world!");
        assert!((words[1].start_ms - 400.0).abs() < 1e-6);
        assert!((words[1].end_ms - 750.0).abs() < 1e-6);
        assert!((words[1].prob - 0.88).abs() < 1e-6);
    }

    #[test]
    fn tokens_to_words_handles_whitespace_only_boundary_tokens_and_min_conf() {
        let tokens = vec![
            tok(" ", 0.0, 0.05, 0.10),
            tok("un", 0.05, 0.15, 0.95),
            tok("believ", 0.15, 0.30, 0.42),
            tok("able", 0.30, 0.45, 0.88),
            tok(" ", 0.45, 0.50, 0.20),
            tok("result", 0.50, 0.80, 0.85),
            tok(" ", 0.80, 0.85, 0.15),
        ];
        let words = tokens_to_words(&tokens, &[], 0.9);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "unbelievable");
        assert!((words[0].start_ms - 50.0).abs() < 1e-6);
        assert!((words[0].end_ms - 450.0).abs() < 1e-6);
        assert!((words[0].prob - 0.42).abs() < 1e-6);

        assert_eq!(words[1].text, " result");
        assert!((words[1].start_ms - 500.0).abs() < 1e-6);
        assert!((words[1].end_ms - 800.0).abs() < 1e-6);
        assert!((words[1].prob - 0.85).abs() < 1e-6);
    }

    #[test]
    fn tokens_to_words_supports_sentencepiece_u2581_markers() {
        let tokens = vec![
            tok("▁", 0.0, 0.05, 0.10),
            tok("▁наш", 0.10, 0.25, 0.95),
            tok("▁fore", 0.30, 0.45, 0.90),
            tok("cast", 0.45, 0.65, 0.80),
            tok("▁", 0.65, 0.70, 0.20),
            tok("new▁york", 0.70, 1.00, 0.75),
        ];
        let words = tokens_to_words(&tokens, &[], 0.9);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "наш");
        assert!((words[0].start_ms - 100.0).abs() < 1e-6);
        assert!((words[0].end_ms - 250.0).abs() < 1e-6);
        assert!((words[0].prob - 0.95).abs() < 1e-6);

        assert_eq!(words[1].text, " forecast");
        assert!((words[1].start_ms - 300.0).abs() < 1e-6);
        assert!((words[1].end_ms - 650.0).abs() < 1e-6);
        assert!((words[1].prob - 0.80).abs() < 1e-6);

        assert_eq!(words[2].text, " new york");
        assert!((words[2].start_ms - 700.0).abs() < 1e-6);
        assert!((words[2].end_ms - 1000.0).abs() < 1e-6);
        assert!((words[2].prob - 0.75).abs() < 1e-6);
    }

    #[test]
    fn tokens_to_words_handles_empty_and_fallback_words() {
        let empty = tokens_to_words(&[], &[], 0.85);
        assert!(empty.is_empty());

        let fallback = vec![
            FluidWord {
                w: "hello".to_string(),
                start: 1.0,
                end: 1.5,
            },
            FluidWord {
                w: "world".to_string(),
                start: 1.6,
                end: 2.0,
            },
        ];
        let words = tokens_to_words(&[], &fallback, 0.87);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert!((words[0].start_ms - 1000.0).abs() < 1e-6);
        assert!((words[0].end_ms - 1500.0).abs() < 1e-6);
        assert!((words[0].prob - 0.87).abs() < 1e-6);

        assert_eq!(words[1].text, " world");
        assert!((words[1].start_ms - 1600.0).abs() < 1e-6);
        assert!((words[1].end_ms - 2000.0).abs() < 1e-6);
        assert!((words[1].prob - 0.87).abs() < 1e-6);
    }

    #[test]
    fn job_bookkeeping_counts_same_model_jobs_and_refuses_different_model() {
        let mut jobs = FluidJobBookkeeping::new();
        assert_eq!(jobs.active_jobs(), 0);
        assert_eq!(jobs.active_model(), None);

        // First job with V3 succeeds
        jobs.begin(FluidModel::V3).unwrap();
        assert_eq!(jobs.active_jobs(), 1);
        assert_eq!(jobs.active_model(), Some(FluidModel::V3));

        // Second concurrent job with the same model (V3) succeeds
        jobs.begin(FluidModel::V3).unwrap();
        assert_eq!(jobs.active_jobs(), 2);
        assert_eq!(jobs.active_model(), Some(FluidModel::V3));

        // Concurrent job requesting Ultra while V3 is active is refused
        let err = jobs.begin(FluidModel::Ultra).unwrap_err().to_string();
        assert_eq!(
            err,
            "FluidAudio is busy with fluid-parakeet-v3 in another job; try again when it finishes"
        );
        assert_eq!(jobs.active_jobs(), 2);
        assert_eq!(jobs.active_model(), Some(FluidModel::V3));

        // Finishing one V3 job still leaves one active; Ultra is still refused
        jobs.finish();
        assert_eq!(jobs.active_jobs(), 1);
        assert_eq!(jobs.active_model(), Some(FluidModel::V3));
        assert!(jobs.begin(FluidModel::Ultra).is_err());

        // Finishing the second V3 job returns count to 0 and clears active_model
        jobs.finish();
        assert_eq!(jobs.active_jobs(), 0);
        assert_eq!(jobs.active_model(), None);

        // Now starting an Ultra job succeeds, and V3 is refused while Ultra is active
        jobs.begin(FluidModel::Ultra).unwrap();
        assert_eq!(jobs.active_jobs(), 1);
        assert_eq!(jobs.active_model(), Some(FluidModel::Ultra));
        let err_v3 = jobs.begin(FluidModel::V3).unwrap_err().to_string();
        assert_eq!(
            err_v3,
            "FluidAudio is busy with fluid-parakeet-ultra in another job; try again when it finishes"
        );

        jobs.finish();
        assert_eq!(jobs.active_jobs(), 0);
        assert_eq!(jobs.active_model(), None);
    }

    #[test]
    fn fluid_job_guard_decrements_active_count_on_drop() {
        let engine = Arc::new(FluidAudioEngine::new());
        {
            let mut jobs = engine.inner.jobs.lock().unwrap();
            jobs.begin(FluidModel::V3).unwrap();
            jobs.begin(FluidModel::V3).unwrap();
        }
        assert_eq!(engine.active_job_count(), 2);
        assert!(engine.is_model_in_use(FluidModel::V3));
        assert!(!engine.is_model_in_use(FluidModel::Ultra));

        let guard1 = FluidJobGuard {
            engine: Arc::clone(&engine),
        };
        let guard2 = FluidJobGuard {
            engine: Arc::clone(&engine),
        };

        drop(guard1);
        assert_eq!(engine.active_job_count(), 1);
        assert!(engine.is_model_in_use(FluidModel::V3));

        drop(guard2);
        assert_eq!(engine.active_job_count(), 0);
        assert!(!engine.is_model_in_use(FluidModel::V3));
    }

    #[tokio::test]
    async fn delete_model_files_refuses_when_model_is_in_use() {
        let engine = FluidAudioEngine::new();
        {
            let mut jobs = engine.inner.jobs.lock().unwrap();
            jobs.begin(FluidModel::V3).unwrap();
        }

        let mut verify_called = false;
        let err = engine
            .delete_model_files(FluidModel::V3, Path::new("/tmp"), |_| {
                verify_called = true;
                Ok(())
            })
            .await
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("Cannot delete 'fluid-parakeet-v3' while it is in use"),
            "Unexpected error message: {}",
            err
        );
        assert!(
            !verify_called,
            "verify callback must not run when model is in use"
        );
    }

    fn parse_pcm16_mono_16k_wav(bytes: &[u8]) -> Result<Vec<f32>> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(anyhow!("Invalid RIFF/WAVE header"));
        }
        let mut pos = 12usize;
        while pos + 8 <= bytes.len() {
            let chunk_id = &bytes[pos..pos + 4];
            let chunk_size =
                u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let data_start = pos + 8;
            let data_end = (data_start + chunk_size).min(bytes.len());

            if chunk_id == b"fmt " && chunk_size >= 16 {
                let audio_format = u16::from_le_bytes([bytes[data_start], bytes[data_start + 1]]);
                let channels = u16::from_le_bytes([bytes[data_start + 2], bytes[data_start + 3]]);
                let sample_rate = u32::from_le_bytes([
                    bytes[data_start + 4],
                    bytes[data_start + 5],
                    bytes[data_start + 6],
                    bytes[data_start + 7],
                ]);
                let bits_per_sample =
                    u16::from_le_bytes([bytes[data_start + 14], bytes[data_start + 15]]);
                if audio_format != 1 || channels != 1 || sample_rate != 16000 || bits_per_sample != 16
                {
                    return Err(anyhow!(
                        "Expected 16kHz mono 16-bit PCM WAV, got format={}, channels={}, rate={}, bits={}",
                        audio_format,
                        channels,
                        sample_rate,
                        bits_per_sample
                    ));
                }
            } else if chunk_id == b"data" {
                let pcm = &bytes[data_start..data_end];
                let samples = pcm
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                return Ok(samples);
            }

            pos = data_start + chunk_size + (chunk_size % 2);
        }
        Err(anyhow!("No data chunk found in WAV file"))
    }

    #[tokio::test]
    #[ignore]
    async fn integration_transcribes_wav_with_fluid_helper() {
        let Some(wav_path) = std::env::var("FLUID_TEST_WAV")
            .ok()
            .filter(|s| !s.is_empty())
        else {
            eprintln!("Skipping integration test: FLUID_TEST_WAV not set");
            return;
        };

        let wav_bytes = std::fs::read(&wav_path).expect("Failed to read FLUID_TEST_WAV");
        let samples = parse_pcm16_mono_16k_wav(&wav_bytes).expect("Failed to parse FLUID_TEST_WAV");

        let models_dir = std::env::var("FLUID_TEST_MODELS_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var("HOME").expect("HOME must be set");
                PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
                    .join("com.meetily.ai")
                    .join("models")
                    .join("fluidaudio")
            });

        let engine = FluidAudioEngine::new();
        engine
            .load_model(FluidModel::V3, &models_dir, |_| {})
            .await
            .expect("Failed to load FluidModel::V3");
        assert!(engine.is_model_loaded().await);
        assert_eq!(
            engine.get_current_model().await.as_deref(),
            Some("fluid-parakeet-v3")
        );

        let transcript = engine
            .transcribe(&samples)
            .await
            .expect("Failed to transcribe test WAV");

        assert!(!transcript.text.trim().is_empty(), "Transcript text must not be empty");
        assert!(!transcript.words.is_empty(), "Transcript words must not be empty");
        let char_count = transcript.text.chars().count();
        assert!(
            transcript.words.len() >= (char_count / 12).max(1)
                && transcript.words.len() * 12 >= char_count,
            "Expected at least 1 word span per 12 characters of text (got {} words for {} chars)",
            transcript.words.len(),
            char_count
        );
        assert_eq!(
            transcript.words.len(),
            transcript.text.split_whitespace().count(),
            "Word span count must equal whitespace-split word count of result text"
        );
        assert!(
            transcript
                .words
                .windows(2)
                .all(|pair| pair[1].start_ms >= pair[0].start_ms),
            "Word start_ms must be monotonically non-decreasing: {:?}",
            transcript.words
        );

        let concat: String = transcript.words.iter().map(|w| w.text.as_str()).collect();
        let normalize_ws = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(
            normalize_ws(concat.trim()),
            normalize_ws(transcript.text.trim())
        );

        engine.unload_model().await;
        assert!(!engine.is_model_loaded().await);
    }
}
