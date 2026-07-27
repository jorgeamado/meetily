// Speaker diarization: downloads/caches sherpa-onnx pyannote segmentation + speaker
// embedding models and runs the `diarize-helper` sidecar binary against a WAV file.

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager, Runtime};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::Command;
use uuid::Uuid;

const DIARIZE_PROGRESS_IDLE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HelperProgress {
    Percent(u32),
    Heartbeat,
}

fn parse_progress_line(line: &str) -> Option<HelperProgress> {
    let value: i32 = line.strip_prefix("PROGRESS ")?.trim().parse().ok()?;
    match value {
        -1 => Some(HelperProgress::Heartbeat),
        0..=100 => Some(HelperProgress::Percent(value as u32)),
        _ => None,
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    let total_secs = elapsed.as_secs();
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    if mins > 0 {
        format!("{}m {}s", mins, secs)
    } else {
        format!("{}s", secs)
    }
}

const SEGMENTATION_MODEL_URL: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx";

const SEGMENTATION_MODEL_FILENAME: &str = "segmentation-3.0.onnx";

// Expected sizes: ~5.99MB and ~39.6MB/~28.3MB. Use a conservative floor to detect partial downloads.
const SEGMENTATION_MODEL_MIN_BYTES: u64 = 5_000_000;
const EMBEDDING_MODEL_MIN_BYTES: u64 = 25_000_000;

pub const DEFAULT_EMBEDDING_MODEL: &str = "campplus";

struct EmbeddingModelSpec {
    key: &'static str,
    url: &'static str,
    filename: &'static str,
}

const EMBEDDING_MODELS: &[EmbeddingModelSpec] = &[
    EmbeddingModelSpec {
        key: "campplus",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        filename: "campplus-zh-en-advanced-16k.onnx",
    },
    EmbeddingModelSpec {
        key: "eres2net",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        filename: "eres2net-base-sv-3dspeaker-16k.onnx",
    },
];

fn resolve_embedding_model(key: Option<&str>) -> &'static EmbeddingModelSpec {
    let key = key.unwrap_or(DEFAULT_EMBEDDING_MODEL);
    EMBEDDING_MODELS.iter().find(|m| m.key == key).unwrap_or_else(|| {
        warn!("Unknown speaker embedding model key '{}', defaulting to '{}'", key, DEFAULT_EMBEDDING_MODEL);
        EMBEDDING_MODELS
            .iter()
            .find(|m| m.key == DEFAULT_EMBEDDING_MODEL)
            .expect("default embedding model must exist in catalog")
    })
}

const DIARIZE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiarizeOptions {
    pub enabled: bool,
    pub num_speakers: Option<u32>,
    pub threshold: Option<f32>,
    pub embedding_model: Option<String>,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            num_speakers: None,
            threshold: None,
            embedding_model: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizedSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: usize,
}

struct ModelPaths {
    segmentation: PathBuf,
    embedding: PathBuf,
}

#[derive(Debug, Deserialize)]
struct HelperSegment {
    start: f32,
    end: f32,
    speaker: i32,
}

#[derive(Debug, Deserialize)]
struct HelperOutput {
    segments: Vec<HelperSegment>,
    num_speakers: i32,
}

/// Run speaker diarization on 16kHz mono samples. Returns diarized segments and the
/// number of speakers detected. Reports 0-100 progress with a human-readable message
/// via `on_progress` as models are downloaded and the helper runs.
pub async fn diarize<R: Runtime>(
    samples_16k_mono: &[f32],
    opts: &DiarizeOptions,
    app: &AppHandle<R>,
    mut on_progress: impl FnMut(u32, &str) + Send,
) -> Result<(Vec<DiarizedSegment>, u32)> {
    on_progress(0, "Preparing speaker diarization models...");
    let models = ensure_models(app, opts.embedding_model.as_deref(), |pct, msg| on_progress(pct / 2, msg)).await?;

    on_progress(50, "Identifying speakers...");

    let wav_bytes = build_wav_bytes(samples_16k_mono);
    let temp_path = std::env::temp_dir().join(format!("meetily-diarize-{}.wav", Uuid::new_v4()));
    tokio::fs::write(&temp_path, &wav_bytes)
        .await
        .with_context(|| format!("Failed to write temp diarization wav: {}", temp_path.display()))?;

    let result = run_diarize_helper(&temp_path, &models, opts, |pct, msg| {
        on_progress(50 + pct / 2, msg);
    })
    .await;
    let _ = tokio::fs::remove_file(&temp_path).await;

    let output = result?;
    on_progress(100, &format!("Identified {} speaker(s)", output.num_speakers));

    let segments = output
        .segments
        .into_iter()
        .map(|s| DiarizedSegment {
            start: s.start,
            end: s.end,
            speaker: s.speaker.max(0) as usize,
        })
        .collect();

    Ok((segments, output.num_speakers.max(0) as u32))
}

/// Given a transcript segment's [start, end] range (seconds), pick the diarized speaker
/// with the greatest time overlap. Returns None if there is zero overlap with every
/// diarized segment.
pub fn assign_speaker_label(
    seg_start_sec: f64,
    seg_end_sec: f64,
    diarized: &[DiarizedSegment],
) -> Option<usize> {
    diarized
        .iter()
        .map(|d| {
            (
                overlap_seconds(seg_start_sec, seg_end_sec, d.start as f64, d.end as f64),
                d.speaker,
            )
        })
        .filter(|(overlap, _)| *overlap > 0.0)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, speaker)| speaker)
}

fn overlap_seconds(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

/// 1-based display label, e.g. speaker index 0 -> "Speaker 1".
pub fn speaker_label(speaker_index: usize) -> String {
    format!("Speaker {}", speaker_index + 1)
}

fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| anyhow!("Failed to get app data dir: {}", e))?;
    Ok(base.join("models").join("diarization"))
}

async fn ensure_models<R: Runtime>(
    app: &AppHandle<R>,
    embedding_model_key: Option<&str>,
    mut on_progress: impl FnMut(u32, &str) + Send,
) -> Result<ModelPaths> {
    let dir = models_dir(app)?;
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("Failed to create diarization models dir: {}", dir.display()))?;

    let embedding_model = resolve_embedding_model(embedding_model_key);
    let segmentation = dir.join(SEGMENTATION_MODEL_FILENAME);
    let embedding = dir.join(embedding_model.filename);

    if !is_valid_file(&segmentation, SEGMENTATION_MODEL_MIN_BYTES).await {
        download_file(SEGMENTATION_MODEL_URL, &segmentation, |pct| {
            on_progress(pct / 2, "Downloading speaker segmentation model...")
        })
        .await?;
    }

    if !is_valid_file(&embedding, EMBEDDING_MODEL_MIN_BYTES).await {
        download_file(embedding_model.url, &embedding, |pct| {
            on_progress(50 + pct / 2, "Downloading speaker embedding model...")
        })
        .await?;
    }

    on_progress(100, "Speaker diarization models ready");

    Ok(ModelPaths {
        segmentation,
        embedding,
    })
}

async fn is_valid_file(path: &Path, min_bytes: u64) -> bool {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata.len() >= min_bytes,
        Err(_) => false,
    }
}

async fn download_file(url: &str, dest: &Path, mut on_progress: impl FnMut(u32) + Send) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 3;
    let mut last_err = None;

    for attempt in 1..=MAX_ATTEMPTS {
        match download_file_once(url, dest, &mut on_progress).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                warn!(
                    "Diarization model download attempt {}/{} failed for {}: {}",
                    attempt, MAX_ATTEMPTS, url, e
                );
                last_err = Some(e);
                if attempt < MAX_ATTEMPTS {
                    tokio::time::sleep(Duration::from_secs(2 * attempt as u64)).await;
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("Download failed for {}", url)))
}

async fn download_file_once(url: &str, dest: &Path, on_progress: &mut impl FnMut(u32)) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1800))
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Request failed: {}", url))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Download failed with status {}: {}",
            response.status(),
            url
        ));
    }

    let total = response.content_length().unwrap_or(0);
    let temp_dest = dest.with_extension("part");
    let file = tokio::fs::File::create(&temp_dest)
        .await
        .with_context(|| format!("Failed to create {}", temp_dest.display()))?;
    let mut writer = BufWriter::new(file);
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Stream error downloading {}", url))?;
        writer
            .write_all(&chunk)
            .await
            .with_context(|| format!("Failed to write to {}", temp_dest.display()))?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            on_progress(((downloaded as f64 / total as f64) * 100.0).min(100.0) as u32);
        }
    }

    writer.flush().await.context("Failed to flush download")?;
    drop(writer);
    tokio::fs::rename(&temp_dest, dest)
        .await
        .with_context(|| format!("Failed to finalize download to {}", dest.display()))?;

    Ok(())
}

fn target_triple() -> String {
    std::env::var("TARGET").unwrap_or_else(|_| {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            "x86_64-unknown-linux-gnu".to_string()
        }
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            "aarch64-unknown-linux-gnu".to_string()
        }
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        {
            "x86_64-apple-darwin".to_string()
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            "aarch64-apple-darwin".to_string()
        }
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            "x86_64-pc-windows-msvc".to_string()
        }
        #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
        {
            "aarch64-pc-windows-msvc".to_string()
        }
        #[cfg(not(any(
            all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")),
            all(target_os = "macos", any(target_arch = "x86_64", target_arch = "aarch64")),
            all(target_os = "windows", any(target_arch = "x86_64", target_arch = "aarch64"))
        )))]
        {
            "unknown".to_string()
        }
    })
}

fn resolve_diarize_helper_binary() -> Result<PathBuf> {
    if let Ok(env_path) = std::env::var("MEETILY_DIARIZE_HELPER") {
        if !env_path.is_empty() {
            let path = PathBuf::from(env_path);
            if path.exists() {
                log::info!("Using diarize-helper from MEETILY_DIARIZE_HELPER: {}", path.display());
                return Ok(path);
            }
        }
    }

    let target_triple = target_triple();
    let binary_name = if cfg!(windows) {
        format!("diarize-helper-{}.exe", target_triple)
    } else {
        format!("diarize-helper-{}", target_triple)
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
                        if name.starts_with("diarize-helper") && !name.ends_with(".d") {
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
                    if name.starts_with("diarize-helper") && !name.ends_with(".d") {
                        return Ok(path);
                    }
                }
            }
        }
    }

    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let project_root = PathBuf::from(&manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| anyhow!("Failed to determine project root"))?
            .to_path_buf();

        let candidates = vec![
            project_root.join("target/release/diarize-helper"),
            project_root.join("target/debug/diarize-helper"),
            project_root.join("target/release/diarize-helper.exe"),
            project_root.join("target/debug/diarize-helper.exe"),
        ];

        for candidate in candidates {
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    Err(anyhow!(
        "diarize-helper binary not found. Build with 'cargo build -p diarize-helper --release' or set MEETILY_DIARIZE_HELPER env var."
    ))
}

async fn run_diarize_helper(
    wav_path: &Path,
    models: &ModelPaths,
    opts: &DiarizeOptions,
    mut on_progress: impl FnMut(u32, &str) + Send,
) -> Result<HelperOutput> {
    let binary = resolve_diarize_helper_binary()?;

    let mut command = Command::new(&binary);
    command
        .arg("--audio")
        .arg(wav_path)
        .arg("--segmentation-model")
        .arg(&models.segmentation)
        .arg("--embedding-model")
        .arg(&models.embedding)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(num_speakers) = opts.num_speakers {
        command.arg("--num-speakers").arg(num_speakers.to_string());
    }
    if let Some(threshold) = opts.threshold {
        command.arg("--threshold").arg(threshold.to_string());
    }

    log::info!("Spawning diarize-helper: {}", binary.display());

    let mut child = command
        .spawn()
        .with_context(|| format!("Failed to spawn diarize-helper at {}", binary.display()))?;

    let mut stdout = child.stdout.take().ok_or_else(|| anyhow!("Failed to capture diarize-helper stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| anyhow!("Failed to capture diarize-helper stderr"))?;

    let stdout_task = tokio::spawn(async move {
        let mut buf = String::new();
        stdout.read_to_string(&mut buf).await.map(|_| buf)
    });

    let progress_queue: Arc<Mutex<VecDeque<HelperProgress>>> = Arc::new(Mutex::new(VecDeque::new()));
    let progress_queue_for_task = progress_queue.clone();
    let stderr_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        let mut buf = String::new();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => match parse_progress_line(&line) {
                    Some(event) => progress_queue_for_task.lock().unwrap().push_back(event),
                    None => {
                        buf.push_str(&line);
                        buf.push('\n');
                    }
                },
                Ok(None) => break,
                Err(_) => break,
            }
        }
        buf
    });

    let start = Instant::now();
    let mut last_known_pct: u32 = 0;
    let mut last_progress_at = Instant::now();
    let status = loop {
        if start.elapsed() > DIARIZE_TIMEOUT {
            let _ = child.kill().await;
            return Err(anyhow!("diarize-helper timed out after {:?}", DIARIZE_TIMEOUT));
        }
        if super::retranscription::is_cancellation_requested() {
            let _ = child.kill().await;
            return Err(anyhow!("Retranscription cancelled"));
        }

        {
            let mut queue = progress_queue.lock().unwrap();
            while let Some(event) = queue.pop_front() {
                last_progress_at = Instant::now();
                match event {
                    HelperProgress::Percent(pct) => {
                        last_known_pct = pct;
                        on_progress(pct, &format!("Identifying speakers... ({}%)", pct));
                    }
                    HelperProgress::Heartbeat => {
                        on_progress(last_known_pct, "Identifying speakers...");
                    }
                }
            }
        }

        if last_progress_at.elapsed() > DIARIZE_PROGRESS_IDLE_TIMEOUT {
            on_progress(
                last_known_pct,
                &format!("Identifying speakers... ({})", format_elapsed(start.elapsed())),
            );
            last_progress_at = Instant::now();
        }

        match tokio::time::timeout(Duration::from_millis(300), child.wait()).await {
            Ok(status_result) => break status_result.context("Failed to wait for diarize-helper")?,
            Err(_) => continue,
        }
    };

    let stdout_output = stdout_task
        .await
        .context("diarize-helper stdout reader task panicked")?
        .context("Failed reading diarize-helper stdout")?;
    let stderr_output = stderr_task
        .await
        .context("diarize-helper stderr reader task panicked")?;

    if !stderr_output.trim().is_empty() {
        debug!("diarize-helper stderr: {}", stderr_output.trim());
    }

    if !status.success() {
        let err_msg = serde_json::from_str::<serde_json::Value>(stdout_output.trim())
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(|s| s.to_string()))
            .unwrap_or_else(|| format!("diarize-helper exited with status {}", status));
        return Err(anyhow!(err_msg));
    }

    serde_json::from_str(stdout_output.trim())
        .with_context(|| format!("Failed to parse diarize-helper output: {}", stdout_output.trim()))
}

fn build_wav_bytes(samples: &[f32]) -> Vec<u8> {
    let sample_rate: u32 = 16000;
    let channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * block_align as u32;

    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        pcm.extend_from_slice(&value.to_le_bytes());
    }

    let data_size = pcm.len() as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(&pcm);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, speaker: usize) -> DiarizedSegment {
        DiarizedSegment { start, end, speaker }
    }

    #[test]
    fn parses_percent_progress_line() {
        assert_eq!(parse_progress_line("PROGRESS 0"), Some(HelperProgress::Percent(0)));
        assert_eq!(parse_progress_line("PROGRESS 42"), Some(HelperProgress::Percent(42)));
        assert_eq!(parse_progress_line("PROGRESS 100"), Some(HelperProgress::Percent(100)));
    }

    #[test]
    fn parses_heartbeat_progress_line() {
        assert_eq!(parse_progress_line("PROGRESS -1"), Some(HelperProgress::Heartbeat));
    }

    #[test]
    fn rejects_out_of_range_and_malformed_lines() {
        assert_eq!(parse_progress_line("PROGRESS 101"), None);
        assert_eq!(parse_progress_line("PROGRESS -2"), None);
        assert_eq!(parse_progress_line("PROGRESS abc"), None);
        assert_eq!(parse_progress_line("using 8 threads"), None);
        assert_eq!(parse_progress_line(""), None);
    }

    #[test]
    fn formats_elapsed_time() {
        assert_eq!(format_elapsed(Duration::from_secs(5)), "5s");
        assert_eq!(format_elapsed(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_elapsed(Duration::from_secs(135)), "2m 15s");
    }

    #[test]
    fn full_overlap_picks_the_containing_speaker() {
        let diarized = vec![seg(0.0, 5.0, 0), seg(5.0, 10.0, 1)];
        assert_eq!(assign_speaker_label(1.0, 2.0, &diarized), Some(0));
    }

    #[test]
    fn partial_overlap_picks_speaker_with_more_overlap() {
        // Whisper segment [4, 8] overlaps speaker 0 by 1s ([4,5]) and speaker 1 by 3s ([5,8])
        let diarized = vec![seg(0.0, 5.0, 0), seg(5.0, 10.0, 1)];
        assert_eq!(assign_speaker_label(4.0, 8.0, &diarized), Some(1));
    }

    #[test]
    fn zero_overlap_returns_none() {
        let diarized = vec![seg(0.0, 5.0, 0), seg(10.0, 15.0, 1)];
        assert_eq!(assign_speaker_label(6.0, 9.0, &diarized), None);
    }

    #[test]
    fn touching_boundary_has_zero_overlap() {
        let diarized = vec![seg(0.0, 5.0, 0)];
        assert_eq!(assign_speaker_label(5.0, 10.0, &diarized), None);
    }

    #[test]
    fn empty_diarized_segments_returns_none() {
        assert_eq!(assign_speaker_label(0.0, 10.0, &[]), None);
    }

    #[test]
    fn speaker_label_is_one_based() {
        assert_eq!(speaker_label(0), "Speaker 1");
        assert_eq!(speaker_label(1), "Speaker 2");
    }

    #[test]
    fn resolves_known_and_unknown_embedding_model_keys() {
        assert_eq!(resolve_embedding_model(Some("campplus")).filename, "campplus-zh-en-advanced-16k.onnx");
        assert_eq!(resolve_embedding_model(Some("eres2net")).filename, "eres2net-base-sv-3dspeaker-16k.onnx");
        assert_eq!(resolve_embedding_model(Some("bogus")).filename, "campplus-zh-en-advanced-16k.onnx");
        assert_eq!(resolve_embedding_model(None).filename, "campplus-zh-en-advanced-16k.onnx");
    }

    #[test]
    fn wav_bytes_have_valid_riff_header() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let wav = build_wav_bytes(&samples);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + samples.len() * 2);
    }
}
