// Retranscription module - allows re-processing stored audio with different settings

use crate::audio::decoder::decode_audio_file;
use crate::audio::boundary_refine;
use crate::audio::diarization::{self, DiarizeOptions};
use crate::audio::glossary;
use crate::audio::stereo;
use crate::audio::transcript_repair;
use crate::audio::vad::get_speech_chunks_with_progress;
use super::common::{create_transcript_segments, split_segment_at_silence, write_transcripts_json};
use super::constants::AUDIO_EXTENSIONS;
use crate::config::{DEFAULT_WHISPER_MODEL, DEFAULT_PARAKEET_MODEL};
use crate::parakeet_engine::ParakeetEngine;
use crate::state::AppState;
use crate::whisper_engine::WhisperEngine;
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, Runtime};

/// Global flag to track if retranscription is in progress
static RETRANSCRIPTION_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Global flag to signal cancellation
static RETRANSCRIPTION_CANCELLED: AtomicBool = AtomicBool::new(false);

/// RAII guard for RETRANSCRIPTION_IN_PROGRESS flag
/// Ensures flag is cleared even if retranscription panics or returns early
struct RetranscriptionGuard;

impl RetranscriptionGuard {
    /// Create guard and set flag atomically
    fn acquire() -> Result<Self, String> {
        if RETRANSCRIPTION_IN_PROGRESS
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err("Retranscription already in progress".to_string());
        }
        Ok(RetranscriptionGuard)
    }
}

impl Drop for RetranscriptionGuard {
    fn drop(&mut self) {
        RETRANSCRIPTION_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

/// VAD redemption time in milliseconds - bridges natural pauses in speech
/// Batch processing needs longer redemption (2000ms) than live pipeline (400ms)
/// because the entire file is processed at once by VAD, and 400ms fragments
/// speech at every natural sentence/topic pause (500ms-2s)
const VAD_REDEMPTION_TIME_MS: u32 = 2000;

/// Progress update emitted during retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionProgress {
    pub meeting_id: String,
    pub stage: String, // "decoding", "transcribing", "saving"
    pub progress_percentage: u32,
    pub message: String,
}

/// Result of retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionResult {
    pub meeting_id: String,
    pub segments_count: usize,
    pub duration_seconds: f64,
    pub language: Option<String>,
}

/// Error during retranscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionError {
    pub meeting_id: String,
    pub error: String,
}

/// Check if retranscription is currently in progress
pub fn is_retranscription_in_progress() -> bool {
    RETRANSCRIPTION_IN_PROGRESS.load(Ordering::SeqCst)
}

/// Cancel ongoing retranscription
pub fn cancel_retranscription() {
    RETRANSCRIPTION_CANCELLED.store(true, Ordering::SeqCst);
}

/// Check whether cancellation has been requested (used by the diarization sidecar
/// to know when to kill the running helper process)
pub(crate) fn is_cancellation_requested() -> bool {
    RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst)
}

/// Start retranscription of a meeting's audio
pub async fn start_retranscription<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    diarize: Option<DiarizeOptions>,
    repair: Option<bool>,
) -> Result<RetranscriptionResult> {
    // Acquire guard - ensures flag is cleared even on panic/early return
    let _guard = RetranscriptionGuard::acquire().map_err(|e| anyhow!(e))?;

    // Reset cancellation flag
    RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);

    let use_parakeet = provider.as_deref() == Some("parakeet");
    let result = run_retranscription(app.clone(), meeting_id.clone(), meeting_folder_path, language, model, provider, diarize, repair).await;

    // The glossary prompt must never outlive the batch run: live
    // transcription shares the engine singleton
    {
        use crate::whisper_engine::commands::WHISPER_ENGINE;
        let engine = {
            let guard = WHISPER_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().cloned()
        };
        if let Some(e) = engine {
            e.set_initial_prompt(None).await;
        }
    }

    // Unload the engine after the batch job (success, failure, or cancellation)
    super::common::unload_engine_after_batch(use_parakeet).await;

    // Guard will automatically clear flag on drop
    // No need for manual: RETRANSCRIPTION_IN_PROGRESS.store(false, Ordering::SeqCst);

    match &result {
        Ok(res) => {
            let _ = app.emit(
                "retranscription-complete",
                serde_json::json!({
                    "meeting_id": res.meeting_id,
                    "segments_count": res.segments_count,
                    "duration_seconds": res.duration_seconds,
                    "language": res.language
                }),
            );
        }
        Err(e) => {
            let _ = app.emit(
                "retranscription-error",
                RetranscriptionError {
                    meeting_id: meeting_id.clone(),
                    error: e.to_string(),
                },
            );
        }
    }

    result
}

/// Find audio file in meeting folder
/// Tries common names first, then scans for any file with an audio extension
fn find_audio_file(folder: &Path) -> Result<PathBuf> {
    let candidates = [
        "audio.mp4", "audio.m4a", "audio.wav", "audio.mp3",
        "audio.flac", "audio.ogg", "recording.mp4",
        "audio.mkv", "audio.webm", "audio.wma",
    ];

    for name in candidates {
        let path = folder.join(name);
        if path.exists() {
            return Ok(path);
        }
    }

    // Fallback: scan folder for any file with an audio extension
    if let Ok(entries) = std::fs::read_dir(folder) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if AUDIO_EXTENSIONS.contains(&ext.as_str()) {
                    return Ok(path);
                }
            }
        }
    }

    Err(anyhow!("No audio file found in: {}", folder.display()))
}

/// Internal function to run retranscription
async fn run_retranscription<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    diarize: Option<DiarizeOptions>,
    repair: Option<bool>,
) -> Result<RetranscriptionResult> {
    let folder_path = PathBuf::from(&meeting_folder_path);
    let audio_path = find_audio_file(&folder_path)?;

    // Determine which provider to use (default to whisper)
    let use_parakeet = provider.as_deref() == Some("parakeet");
    let diarize_opts = diarize.unwrap_or_default();

    info!(
        "Starting retranscription for meeting {} with language {:?}, model {:?}, provider {:?}, diarize {:?}",
        meeting_id, language, model, provider, diarize_opts
    );

    // Emit progress: decoding
    emit_progress(&app, &meeting_id, "decoding", 5, "Decoding audio file...");

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }

    // Decode the audio file (CPU-intensive, run in blocking task)
    let path_for_decode = audio_path.clone();
    let decoded = tokio::task::spawn_blocking(move || {
        decode_audio_file(&path_for_decode)
    })
    .await
    .map_err(|e| anyhow!("Decode task panicked: {}", e))??;
    let duration_seconds = decoded.duration_seconds;

    info!(
        "Decoded audio: {:.2}s, {}Hz, {} channels",
        duration_seconds, decoded.sample_rate, decoded.channels
    );

    emit_progress(&app, &meeting_id, "decoding", 15, "Converting audio format...");

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }

    // Stereo recordings made by this app carry channel identity: left = local
    // mic, right = system audio. When present (and diarization is on), the
    // system channel alone is diarized and the mic channel becomes a
    // ground-truth local-speaker track overlaid on top.
    let stereo_identity = decoded.channels == 2
        && stereo::channel_layout(&folder_path).as_deref() == Some(stereo::MIC_SYSTEM_LAYOUT);
    if stereo_identity {
        info!("Stereo channel identity detected (mic-left/system-right)");
    }

    // Convert to 16kHz mono format (CPU-intensive, run in blocking task)
    let split_for_identity = stereo_identity && diarize_opts.enabled;
    let (audio_samples, stereo_16k) = tokio::task::spawn_blocking(move || {
        let mono = decoded.to_whisper_format();
        let split = if split_for_identity {
            stereo::split_channels_16k(&decoded)
        } else {
            None
        };
        (mono, split)
    })
    .await
    .map_err(|e| anyhow!("Resample task panicked: {}", e))?;
    info!("Converted to 16kHz mono format: {} samples", audio_samples.len());
    // Shared with the VAD task and, later, boundary refinement's acoustic gate
    let audio_samples = std::sync::Arc::new(audio_samples);

    // Mic-channel speech intervals (local user) and the system channel that
    // the diarizer should see instead of the mixed mono
    let (mic_intervals, system_16k) = match stereo_16k {
        Some((mic, sys)) => {
            let intervals = stereo::mic_activity_intervals(&mic, &sys, 16000);
            info!(
                "Mic channel: {} local-speech interval(s), {:.1}s total",
                intervals.len(),
                intervals.iter().map(|(s, e)| e - s).sum::<f64>()
            );
            (Some(intervals), Some(sys))
        }
        None => (None, None),
    };

    // Keep a copy of the full audio for diarization, which runs independently of VAD/whisper.
    // With channel identity, the diarizer only sees the system channel: the local
    // user's voice never enters clustering, so it can't be confused with a remote one.
    let audio_samples_for_diarization = if diarize_opts.enabled {
        Some(system_16k.unwrap_or_else(|| (*audio_samples).clone()))
    } else {
        None
    };

    emit_progress(&app, &meeting_id, "vad", 20, "Detecting speech segments...");

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("Retranscription cancelled"));
    }

    // Use VAD to find natural speech boundaries (same approach as live transcription)
    // IMPORTANT: Run VAD in a blocking task to avoid blocking the async runtime
    // For large files (35+ minutes), VAD processing can take several minutes
    let app_for_vad = app.clone();
    let meeting_id_for_vad = meeting_id.clone();
    let audio_for_vad = audio_samples.clone();

    let speech_segments = tokio::task::spawn_blocking(move || {
        get_speech_chunks_with_progress(
            &audio_for_vad,
            VAD_REDEMPTION_TIME_MS,
            |vad_progress, segments_found| {
                // Map VAD progress (0-100) to overall progress (20-25)
                let overall_progress = 20 + (vad_progress as f32 * 0.05) as u32;
                emit_progress(
                    &app_for_vad,
                    &meeting_id_for_vad,
                    "vad",
                    overall_progress,
                    &format!("Detecting speech segments... {}% ({} found)", vad_progress, segments_found),
                );

                // Return false to cancel if cancellation requested
                !RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst)
            },
        )
    })
    .await
    .map_err(|e| anyhow!("VAD task panicked: {}", e))?
    .map_err(|e| anyhow!("VAD processing failed: {}", e))?;

    let total_segments = speech_segments.len();
    info!("VAD detected {} speech segments (redemption_time={}ms)", total_segments, VAD_REDEMPTION_TIME_MS);

    // Diagnostic: log segment duration distribution
    if !speech_segments.is_empty() {
        let durations_ms: Vec<f64> = speech_segments.iter()
            .map(|s| s.end_timestamp_ms - s.start_timestamp_ms)
            .collect();
        let total_speech_ms: f64 = durations_ms.iter().sum();
        let avg_duration = total_speech_ms / durations_ms.len() as f64;
        let min_duration = durations_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_duration = durations_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        info!(
            "VAD segment stats: avg={:.0}ms, min={:.0}ms, max={:.0}ms, total_speech={:.1}s/{:.1}s ({:.0}%)",
            avg_duration, min_duration, max_duration,
            total_speech_ms / 1000.0, duration_seconds,
            (total_speech_ms / 1000.0 / duration_seconds) * 100.0
        );
        // Log first 10 segments for detailed inspection
        for (i, seg) in speech_segments.iter().take(10).enumerate() {
            let dur = seg.end_timestamp_ms - seg.start_timestamp_ms;
            debug!("  Segment {}: {:.0}ms-{:.0}ms ({:.0}ms, {} samples)",
                i, seg.start_timestamp_ms, seg.end_timestamp_ms, dur, seg.samples.len());
        }
        if total_segments > 10 {
            debug!("  ... and {} more segments", total_segments - 10);
        }
    }

    if total_segments == 0 {
        warn!("No speech detected in audio");
        return Err(anyhow!("No speech detected in audio file"));
    }

    // Run speaker diarization (if enabled) concurrently with transcription below. Failures
    // here must not fail the whole retranscription - we just fall back to no speaker labels.
    let mut diarize_handle: Option<tauri::async_runtime::JoinHandle<Vec<diarization::DiarizedSegment>>> =
        if let Some(samples) = audio_samples_for_diarization {
            if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
                return Err(anyhow!("Retranscription cancelled"));
            }

            let app_for_diarize = app.clone();
            let meeting_id_for_diarize = meeting_id.clone();
            let diarize_opts_for_task = diarize_opts.clone();
            let mic_intervals_for_task = mic_intervals.clone();

            Some(tauri::async_runtime::spawn(async move {
                let diarize_result = diarization::diarize(
                    &samples,
                    &diarize_opts_for_task,
                    &app_for_diarize,
                    |pct, msg| {
                        emit_progress(&app_for_diarize, &meeting_id_for_diarize, "diarizing", pct, msg);
                    },
                )
                .await;

                match diarize_result {
                    Ok((segments, num_speakers)) => {
                        info!(
                            "Speaker diarization found {} speaker(s) across {} segments",
                            num_speakers,
                            segments.len()
                        );
                        // Channel identity: overlay mic-channel speech as the
                        // local speaker on top of the remote-only diarization
                        if let Some(intervals) = &mic_intervals_for_task {
                            let overlaid = stereo::overlay_local_speaker(&segments, intervals);
                            info!(
                                "Overlaid {} local-speaker interval(s): {} -> {} segments",
                                intervals.len(),
                                segments.len(),
                                overlaid.len()
                            );
                            return overlaid;
                        }
                        segments
                    }
                    Err(e) => {
                        warn!("Speaker diarization failed, continuing without speaker labels: {}", e);
                        emit_progress(
                            &app_for_diarize,
                            &meeting_id_for_diarize,
                            "diarizing",
                            100,
                            "Speaker identification failed, continuing without speaker labels",
                        );
                        Vec::new()
                    }
                }
            }))
        } else {
            None
        };

    emit_progress(&app, &meeting_id, "transcribing", 0, "Loading transcription engine...");

    // Initialize the appropriate engine once (not per-segment)
    let whisper_engine = if !use_parakeet {
        Some(get_or_init_whisper(&app, model.as_deref()).await?)
    } else {
        None
    };
    let parakeet_engine = if use_parakeet {
        Some(get_or_init_parakeet(&app, model.as_deref()).await?)
    } else {
        None
    };

    // Glossary biasing uses APPROVED terms only and is gated on the
    // Settings toggle (initial_prompt returns None otherwise). Auto-learned
    // suggestions never reach whisper: unvetted injection hallucinated
    // prompt continuations at unclear segment starts ("Geroen Gürtel-",
    // 2026-07-28).
    if let Some(engine) = whisper_engine.as_ref() {
        if let Some(prompt) = glossary::initial_prompt(&app) {
            info!("Applying approved glossary as whisper initial_prompt: {}", prompt);
            engine.set_initial_prompt(Some(prompt)).await;
        }
    }

    // Split very long segments at silence boundaries for better transcription quality.
    // Hard cuts at arbitrary sample positions lose words at boundaries. Instead, scan
    // for the lowest-energy window near the target split point and cut there.
    const MAX_SEGMENT_SAMPLES: usize = 25 * 16000; // 25 seconds at 16kHz

    let mut processable_segments: Vec<crate::audio::vad::SpeechSegment> = Vec::new();
    for segment in &speech_segments {
        if segment.samples.len() > MAX_SEGMENT_SAMPLES {
            debug!(
                "Splitting large segment ({:.0}ms, {} samples) at silence boundaries",
                segment.end_timestamp_ms - segment.start_timestamp_ms,
                segment.samples.len()
            );

            let sub_segments = split_segment_at_silence(segment, MAX_SEGMENT_SAMPLES);
            debug!("Split into {} sub-segments", sub_segments.len());
            processable_segments.extend(sub_segments);
        } else {
            processable_segments.push(segment.clone());
        }
    }

    let processable_count = processable_segments.len();
    info!("Processing {} segments (after splitting)", processable_count);

    // First use of a freshly installed CoreML encoder blocks inside whisper
    // while macOS compiles it for the Neural Engine (~15 min for large
    // models, cached per app afterwards). Without feedback that reads as a
    // hang at "Transcribing 0%" — tick elapsed time until the first
    // segment comes back, then write the marker so this never shows again.
    struct AbortOnDrop(Option<tauri::async_runtime::JoinHandle<()>>);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                h.abort();
            }
        }
    }
    #[cfg(target_os = "macos")]
    let mut ane_marker: Option<PathBuf> = match whisper_engine.as_ref() {
        Some(engine) => {
            let name = model.as_deref().unwrap_or(DEFAULT_WHISPER_MODEL);
            match engine.model_path(name).await {
                Some(p) => crate::whisper_engine::coreml::first_use_pending(&p),
                None => None,
            }
        }
        None => None,
    };
    #[cfg(not(target_os = "macos"))]
    let mut ane_marker: Option<PathBuf> = None;
    let mut ane_ticker = AbortOnDrop(ane_marker.as_ref().map(|_| {
        let app_for_tick = app.clone();
        let meeting_for_tick = meeting_id.clone();
        let started = std::time::Instant::now();
        tauri::async_runtime::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                let secs = started.elapsed().as_secs();
                emit_progress(
                    &app_for_tick,
                    &meeting_for_tick,
                    "transcribing",
                    0,
                    &format!(
                        "Optimizing model for the Neural Engine — one-time, up to ~15 min ({}m {:02}s)...",
                        secs / 60,
                        secs % 60
                    ),
                );
            }
        })
    }));

    // Process each speech segment with progress updates
    // (text, start_ms, end_ms, token spans in absolute ms for speaker splitting)
    let mut all_transcripts: Vec<(String, f64, f64, Vec<diarization::WordSpan>)> = Vec::new();
    let mut total_confidence = 0.0f32;

    for (i, segment) in processable_segments.iter().enumerate() {
        // Check for cancellation before each segment
        if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
            if let Some(handle) = diarize_handle.take() {
                let _ = handle.await;
            }
            return Err(anyhow!("Retranscription cancelled"));
        }

        // Calculate progress (0-100 range, own to the "transcribing" stage)
        let progress = ((i as u64 * 100) / processable_count as u64) as u32;
        let segment_duration_sec = (segment.end_timestamp_ms - segment.start_timestamp_ms) / 1000.0;
        emit_progress(
            &app,
            &meeting_id,
            "transcribing",
            progress,
            &format!(
                "Transcribing segment {} of {} ({:.1}s)...",
                i + 1,
                processable_count,
                segment_duration_sec
            ),
        );

        // Skip very short segments (< 100ms of audio = 1600 samples at 16kHz)
        if segment.samples.len() < 1600 {
            debug!("Skipping short segment {} with {} samples", i, segment.samples.len());
            continue;
        }

        // Transcribe this segment
        let (text, conf, words) = if use_parakeet {
            let engine = parakeet_engine.as_ref().unwrap();
            let text = engine
                .transcribe_audio(segment.samples.clone())
                .await
                .map_err(|e| anyhow!("Parakeet transcription failed on segment {}: {}", i, e))?;
            (text, 0.9f32, Vec::new())
        } else {
            let engine = whisper_engine.as_ref().unwrap();
            let (text, conf, words) = engine
                .transcribe_audio_with_words(segment.samples.clone(), language.clone())
                .await
                .map_err(|e| anyhow!("Whisper transcription failed on segment {}: {}", i, e))?;
            // Token times are chunk-relative; shift onto the recording timeline
            let words = words
                .into_iter()
                .map(|w| diarization::WordSpan {
                    text: w.text,
                    start_ms: w.start_ms + segment.start_timestamp_ms,
                    end_ms: w.end_ms + segment.start_timestamp_ms,
                    prob: w.prob,
                })
                .collect();
            (text, conf, words)
        };

        // First transcription returned: the ANE compile (if any) is done —
        // stop the ticker and remember so future runs skip the warning
        if let Some(marker) = ane_marker.take() {
            ane_ticker = AbortOnDrop(None);
            let _ = std::fs::write(&marker, "ok");
            info!("CoreML encoder ready (ANE compile complete), marker at {}", marker.display());
        }

        // Skip empty transcripts
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            debug!(
                "Segment {}/{}: {:.1}s, conf={:.2}, text='{}'",
                i + 1, processable_count, segment_duration_sec, conf,
                if trimmed.len() > 80 { let mut end = 80; while !trimmed.is_char_boundary(end) { end -= 1; } &trimmed[..end] } else { trimmed }
            );
            // Stream the raw block to the UI so the transcript is readable
            // while later segments are still transcribing; speaker labels
            // arrive with the final save
            let _ = app.emit(
                "retranscription-partial",
                serde_json::json!({
                    "meeting_id": meeting_id,
                    "text": text.trim(),
                    "audio_start_time": segment.start_timestamp_ms / 1000.0,
                    "audio_end_time": segment.end_timestamp_ms / 1000.0,
                }),
            );
            all_transcripts.push((text, segment.start_timestamp_ms, segment.end_timestamp_ms, words));
            total_confidence += conf;
        } else {
            debug!("Segment {}/{}: {:.1}s — empty transcription", i + 1, processable_count, segment_duration_sec);
        }
    }

    let transcribed_count = all_transcripts.len();
    let avg_confidence = if transcribed_count > 0 {
        total_confidence / transcribed_count as f32
    } else {
        0.0
    };

    info!(
        "Transcription complete: {} segments transcribed out of {}, avg confidence: {:.2}",
        transcribed_count, processable_count, avg_confidence
    );

    // Check for cancellation
    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        if let Some(handle) = diarize_handle.take() {
            let _ = handle.await;
        }
        return Err(anyhow!("Retranscription cancelled"));
    }

    // Await diarization (runs concurrently with the transcription above) before
    // moving past the dual-progress phase; a failed or panicked task just means
    // no speaker labels. The spawned task keeps emitting "diarizing" progress
    // while we wait, so the UI stays live until it finishes.
    let diarized_segments: Vec<diarization::DiarizedSegment> = if let Some(handle) = diarize_handle {
        match handle.await {
            Ok(segments) => segments,
            Err(join_err) => {
                warn!("Speaker diarization task panicked: {}", join_err);
                emit_progress(
                    &app,
                    &meeting_id,
                    "diarizing",
                    100,
                    "Speaker identification failed, continuing without speaker labels",
                );
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    emit_progress(&app, &meeting_id, "saving", 80, "Saving transcripts...");

    // Split blocks that span speaker changes into per-turn rows using token
    // timestamps, so one row never mixes two speakers' words. Blocks without
    // token data (Parakeet) keep whole-block max-overlap labeling.
    let mut turns: Vec<diarization::SpeakerTurn> = all_transcripts
        .iter()
        .flat_map(|(text, start_ms, end_ms, words)| {
            if diarized_segments.is_empty() {
                vec![diarization::SpeakerTurn {
                    text: text.clone(),
                    start_ms: *start_ms,
                    end_ms: *end_ms,
                    speaker: None,
                    words: words.clone(),
                }]
            } else {
                diarization::split_block_into_turns(
                    text,
                    *start_ms,
                    *end_ms,
                    words,
                    &diarized_segments,
                )
            }
        })
        .collect();

    // Second opinion on tight speaker handovers: the acoustic diarizer often
    // hands the next speaker's first word to the previous one. A local LLM
    // picks the natural cut from numbered candidates; text can only be
    // re-partitioned between rows, never altered.
    if diarize_opts.llm_refine.unwrap_or(true) && !diarized_segments.is_empty() {
        emit_progress(&app, &meeting_id, "saving", 80, "Refining speaker boundaries...");
        let app_for_refine = app.clone();
        let meeting_id_for_refine = meeting_id.clone();
        let stats = boundary_refine::refine_turns(
            &app,
            &mut turns,
            Some(&audio_samples[..]),
            || RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst),
            move |done, total| {
                emit_progress(
                    &app_for_refine,
                    &meeting_id_for_refine,
                    "saving",
                    80 + (5 * done / total.max(1)) as u32,
                    &format!("Refining speaker boundaries ({}/{})...", done, total),
                );
            },
        )
        .await;
        info!(
            "Boundary refinement: {} sandwiches ({} merged), {} tight boundaries, {} queried, {} moved ({} acoustic, {} confirmed), {} failures",
            stats.sandwiches, stats.merged, stats.boundaries, stats.queried, stats.moved,
            stats.acoustic_moved, stats.acoustic_confirmed, stats.failures
        );
    }

    // Confidence-gated wording repair: sentences whose tokens decoded with
    // low probability get one constrained LLM patch each (at most 2 words
    // may change; anything bigger is rejected). Timings are untouched.
    if repair.unwrap_or(true) && !use_parakeet {
        emit_progress(&app, &meeting_id, "saving", 86, "Checking low-confidence wording...");
        let app_for_repair = app.clone();
        let meeting_id_for_repair = meeting_id.clone();
        let stats = transcript_repair::repair_turns(
            &app,
            &mut turns,
            || RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst),
            move |done, total| {
                emit_progress(
                    &app_for_repair,
                    &meeting_id_for_repair,
                    "saving",
                    86 + (4 * done / total.max(1)) as u32,
                    &format!("Checking low-confidence wording ({}/{})...", done, total),
                );
            },
        )
        .await;
        info!(
            "Transcript repair: {} flagged, {} queried, {} repaired, {} failures",
            stats.flagged, stats.queried, stats.repaired, stats.failures
        );
    }

    let labeled_blocks: Vec<(String, f64, f64, Option<usize>)> = diarization::turns_to_rows(&turns);

    // Create transcript segments with proper timestamps from VAD
    let segments = save_transcript_rows(&app, &meeting_id, &labeled_blocks).await?;

    // Persist the refinement inputs so the AI passes can be re-run later
    // without redoing transcription (standalone "AI fix-up")
    if let Err(e) = write_refine_data(&folder_path, &turns, &diarized_segments) {
        warn!("Failed to write refine-data.json: {}", e);
    }

    // Learn vocabulary from this meeting in the background (names, terms)
    // so the NEXT retranscription can bias whisper toward it
    if !use_parakeet {
        let row_texts: Vec<String> = labeled_blocks.iter().map(|(t, _, _, _)| t.clone()).collect();
        let sample = glossary::sample_rows(&row_texts);
        tauri::async_runtime::spawn(glossary::update_from_transcript(app.clone(), sample));
    }

    // Write updated transcripts.json and metadata.json to the meeting folder
    emit_progress(&app, &meeting_id, "saving", 90, "Writing transcript files...");

    if let Err(e) = write_transcripts_json(&folder_path, &segments) {
        warn!("Failed to write transcripts.json: {}", e);
    }

    // Find audio filename for metadata
    let audio_filename = audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.mp4")
        .to_string();

    if let Err(e) = write_retranscription_metadata(
        &folder_path,
        &meeting_id,
        duration_seconds,
        &audio_filename,
    ) {
        warn!("Failed to update metadata.json: {}", e);
    }

    emit_progress(&app, &meeting_id, "complete", 100, "Retranscription complete");

    Ok(RetranscriptionResult {
        meeting_id,
        segments_count: segments.len(),
        duration_seconds,
        language,
    })
}

/// Emit progress event
fn emit_progress<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    stage: &str,
    progress: u32,
    message: &str,
) {
    let _ = app.emit(
        "retranscription-progress",
        RetranscriptionProgress {
            meeting_id: meeting_id.to_string(),
            stage: stage.to_string(),
            progress_percentage: progress,
            message: message.to_string(),
        },
    );
}

/// Get or initialize the Whisper engine, auto-loading the model if needed
/// If `requested_model` is provided, ensures that specific model is loaded
/// Replace a meeting's transcript rows in one transaction; returns the
/// written segments (used for transcripts.json). Shared by retranscription
/// and the standalone AI fix-up.
async fn save_transcript_rows<R: Runtime>(
    app: &AppHandle<R>,
    meeting_id: &str,
    labeled_blocks: &[(String, f64, f64, Option<usize>)],
) -> Result<Vec<crate::api::TranscriptSegment>> {
    let plain_blocks: Vec<(String, f64, f64)> = labeled_blocks
        .iter()
        .map(|(t, s, e, _)| (t.clone(), *s, *e))
        .collect();
    let segments = create_transcript_segments(&plain_blocks);

    let app_state = app
        .try_state::<AppState>()
        .ok_or_else(|| anyhow!("App state not available"))?;

    // Wrap delete+insert in a transaction to prevent data loss
    let pool = app_state.db_manager.pool();
    let mut conn = pool.acquire().await.map_err(|e| anyhow!("DB error: {}", e))?;
    let mut tx = sqlx::Connection::begin(&mut *conn)
        .await
        .map_err(|e| anyhow!("Failed to start transaction: {}", e))?;

    sqlx::query("DELETE FROM transcripts WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| anyhow!("Failed to delete existing transcripts: {}", e))?;

    for (segment, (_, _, _, speaker_idx)) in segments.iter().zip(labeled_blocks.iter()) {
        let speaker = speaker_idx.map(diarization::speaker_label);

        sqlx::query(
            "INSERT INTO transcripts (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration, speaker)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&segment.id)
        .bind(meeting_id)
        .bind(&segment.text)
        .bind(&segment.timestamp)
        .bind(segment.audio_start_time)
        .bind(segment.audio_end_time)
        .bind(segment.duration)
        .bind(speaker)
        .execute(&mut *tx)
        .await
        .map_err(|e| anyhow!("Failed to insert transcript: {}", e))?;
    }

    tx.commit().await
        .map_err(|e| anyhow!("Failed to commit transaction: {}", e))?;

    info!(
        "Updated {} transcripts for meeting {} in transaction",
        segments.len(),
        meeting_id
    );
    Ok(segments)
}

/// Inputs the AI passes need, persisted per meeting so they can be re-run
/// without redoing transcription.
#[derive(Serialize, Deserialize)]
struct RefineData {
    turns: Vec<diarization::SpeakerTurn>,
    diarized: Vec<diarization::DiarizedSegment>,
}

const REFINE_DATA_FILENAME: &str = "refine-data.json";

fn write_refine_data(
    folder: &Path,
    turns: &[diarization::SpeakerTurn],
    diarized: &[diarization::DiarizedSegment],
) -> Result<()> {
    let data = RefineData { turns: turns.to_vec(), diarized: diarized.to_vec() };
    let json = serde_json::to_string(&data)?;
    std::fs::write(folder.join(REFINE_DATA_FILENAME), json)?;
    Ok(())
}

fn load_refine_data(folder: &Path) -> Result<RefineData> {
    let path = folder.join(REFINE_DATA_FILENAME);
    let json = std::fs::read_to_string(&path).map_err(|_| {
        anyhow!("No refinement data for this meeting yet — run Enhance (retranscribe) once first")
    })?;
    Ok(serde_json::from_str(&json)?)
}

/// Standalone AI fix-up: re-run the boundary and wording passes on the
/// saved transcript without redoing transcription. Reuses the
/// retranscription progress/complete/error events so the live transcript
/// banner covers this flow too.
async fn run_transcript_refinement<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
) -> Result<()> {
    let started = std::time::Instant::now();
    let folder_path = PathBuf::from(&meeting_folder_path);
    let mut data = load_refine_data(&folder_path)?;
    info!(
        "AI fix-up for meeting {}: {} turns, {} diarized segments",
        meeting_id,
        data.turns.len(),
        data.diarized.len()
    );

    emit_progress(&app, &meeting_id, "refining", 5, "Refining speaker boundaries...");
    if !data.diarized.is_empty() {
        // Decode the recording so the acoustic gate can weigh in; a fix-up
        // without audio (file gone) still works, LLM-only.
        let audio_16k = tokio::task::spawn_blocking({
            let folder = folder_path.clone();
            move || -> Option<Vec<f32>> {
                let audio_path = find_audio_file(&folder).ok()?;
                let decoded = decode_audio_file(&audio_path).ok()?;
                Some(decoded.to_whisper_format())
            }
        })
        .await
        .ok()
        .flatten();
        if audio_16k.is_none() {
            info!("AI fix-up: no decodable audio, boundary refine runs without acoustic gate");
        }
        let app_for_refine = app.clone();
        let meeting_id_for_refine = meeting_id.clone();
        let stats = boundary_refine::refine_turns(
            &app,
            &mut data.turns,
            audio_16k.as_deref(),
            || RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst),
            move |done, total| {
                emit_progress(
                    &app_for_refine,
                    &meeting_id_for_refine,
                    "refining",
                    5 + (45 * done / total.max(1)) as u32,
                    &format!("Refining speaker boundaries ({}/{})...", done, total),
                );
            },
        )
        .await;
        info!(
            "AI fix-up boundaries: {} sandwiches ({} merged), {} boundaries, {} queried, {} moved ({} acoustic, {} confirmed), {} failures",
            stats.sandwiches, stats.merged, stats.boundaries, stats.queried, stats.moved,
            stats.acoustic_moved, stats.acoustic_confirmed, stats.failures
        );
    }

    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("AI fix-up cancelled"));
    }

    emit_progress(&app, &meeting_id, "refining", 50, "Checking low-confidence wording...");
    {
        let app_for_repair = app.clone();
        let meeting_id_for_repair = meeting_id.clone();
        let stats = transcript_repair::repair_turns(
            &app,
            &mut data.turns,
            || RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst),
            move |done, total| {
                emit_progress(
                    &app_for_repair,
                    &meeting_id_for_repair,
                    "refining",
                    50 + (40 * done / total.max(1)) as u32,
                    &format!("Checking low-confidence wording ({}/{})...", done, total),
                );
            },
        )
        .await;
        info!(
            "AI fix-up wording: {} flagged, {} queried, {} repaired, {} failures",
            stats.flagged, stats.queried, stats.repaired, stats.failures
        );
    }

    if RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst) {
        return Err(anyhow!("AI fix-up cancelled"));
    }

    emit_progress(&app, &meeting_id, "refining", 92, "Saving transcripts...");
    let labeled_blocks = diarization::turns_to_rows(&data.turns);
    let segments = save_transcript_rows(&app, &meeting_id, &labeled_blocks).await?;
    if let Err(e) = write_refine_data(&folder_path, &data.turns, &data.diarized) {
        warn!("Failed to update refine-data.json: {}", e);
    }
    if let Err(e) = write_transcripts_json(&folder_path, &segments) {
        warn!("Failed to write transcripts.json: {}", e);
    }

    let _ = app.emit(
        "retranscription-complete",
        serde_json::json!({
            "meeting_id": meeting_id,
            "segments_count": segments.len(),
            "duration_seconds": started.elapsed().as_secs(),
            "language": Option::<String>::None,
        }),
    );
    Ok(())
}

#[tauri::command]
pub async fn refine_transcript_command<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
) -> Result<(), String> {
    if RETRANSCRIPTION_IN_PROGRESS.load(Ordering::SeqCst) {
        return Err("Retranscription already in progress".to_string());
    }

    tauri::async_runtime::spawn(async move {
        let _guard = match RetranscriptionGuard::acquire() {
            Ok(g) => g,
            Err(e) => {
                error!("AI fix-up: {}", e);
                return;
            }
        };
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);

        let result =
            run_transcript_refinement(app.clone(), meeting_id.clone(), meeting_folder_path).await;
        if let Err(e) = result {
            error!("AI fix-up failed: {}", e);
            let _ = app.emit(
                "retranscription-error",
                RetranscriptionError { meeting_id, error: e.to_string() },
            );
        }
    });
    Ok(())
}

async fn get_or_init_whisper<R: Runtime>(
    app: &AppHandle<R>,
    requested_model: Option<&str>,
) -> Result<Arc<WhisperEngine>> {
    use crate::whisper_engine::commands::WHISPER_ENGINE;

    let engine = {
        let guard = WHISPER_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().cloned()
    };

    match engine {
        Some(e) => {
            // Determine which model to use
            let target_model = match requested_model {
                Some(model) => model.to_string(),
                None => get_configured_whisper_model(app).await?,
            };

            // Retranscription wants DTW token timestamps for speaker splitting,
            // so reload if the model differs OR the context lacks DTW
            let current_model = e.get_current_model().await;
            let needs_load = match &current_model {
                Some(loaded) => loaded != &target_model || !e.is_dtw_enabled().await,
                None => true,
            };

            if needs_load {
                info!(
                    "Loading Whisper model '{}' with DTW timestamps (current: {:?})",
                    target_model, current_model
                );

                // Discover available models first (populates the internal cache)
                info!("Discovering available Whisper models...");
                if let Err(discover_err) = e.discover_models().await {
                    warn!("Error during model discovery (continuing anyway): {}", discover_err);
                }

                match e.load_model_with_dtw(&target_model, true).await {
                    Ok(_) => {
                        info!("Whisper model '{}' loaded successfully", target_model);
                        Ok(e)
                    }
                    Err(load_err) => {
                        error!("Failed to load Whisper model '{}': {}", target_model, load_err);
                        Err(anyhow!("Failed to load Whisper model '{}': {}", target_model, load_err))
                    }
                }
            } else {
                info!("Whisper model '{}' already loaded", target_model);
                Ok(e)
            }
        }
        None => Err(anyhow!("Whisper engine not initialized")),
    }
}

/// Get the configured Whisper model name from the database
async fn get_configured_whisper_model<R: Runtime>(app: &AppHandle<R>) -> Result<String> {
    debug!("Getting configured Whisper model from database...");

    let app_state = app
        .try_state::<AppState>()
        .ok_or_else(|| {
            error!("App state not available");
            anyhow!("App state not available")
        })?;

    debug!("Querying transcript_settings table...");

    // Query the transcript settings from the database - get both provider and model
    let result: Option<(String, String)> = sqlx::query_as(
        "SELECT provider, model FROM transcript_settings WHERE id = '1'"
    )
    .fetch_optional(app_state.db_manager.pool())
    .await
    .map_err(|e| {
        error!("Failed to query transcript config: {}", e);
        anyhow!("Failed to query transcript config: {}", e)
    })?;

    match result {
        Some((provider, model)) => {
            info!("Found transcript config: provider={}, model={}", provider, model);

            // Check if provider is Whisper-based
            if provider == "localWhisper" || provider == "whisper" {
                Ok(model)
            } else {
                error!("Retranscription requires Whisper provider, but configured provider is: {}", provider);
                Err(anyhow!("Retranscription requires Whisper. Current provider '{}' does not support retranscription with language selection.", provider))
            }
        },
        None => {
            // Default to configured Whisper model if no config exists
            warn!("No transcript config found, using default model '{}'", DEFAULT_WHISPER_MODEL);
            Ok(DEFAULT_WHISPER_MODEL.to_string())
        }
    }
}

/// Get or initialize the Parakeet engine, auto-loading the model if needed
async fn get_or_init_parakeet<R: Runtime>(
    app: &AppHandle<R>,
    requested_model: Option<&str>,
) -> Result<Arc<ParakeetEngine>> {
    use crate::parakeet_engine::commands::PARAKEET_ENGINE;

    let engine = {
        let guard = PARAKEET_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().cloned()
    };

    match engine {
        Some(e) => {
            // Determine which model to use
            let target_model = match requested_model {
                Some(model) => model.to_string(),
                None => get_configured_parakeet_model(app).await?,
            };

            // Check if the correct model is already loaded
            let current_model = e.get_current_model().await;
            let needs_load = match &current_model {
                Some(loaded) => loaded != &target_model,
                None => true,
            };

            if needs_load {
                info!(
                    "Loading Parakeet model '{}' (current: {:?})",
                    target_model, current_model
                );

                // Discover available models first
                info!("Discovering available Parakeet models...");
                if let Err(discover_err) = e.discover_models().await {
                    warn!("Error during Parakeet model discovery (continuing anyway): {}", discover_err);
                }

                match e.load_model(&target_model).await {
                    Ok(_) => {
                        info!("Parakeet model '{}' loaded successfully", target_model);
                        Ok(e)
                    }
                    Err(load_err) => {
                        error!("Failed to load Parakeet model '{}': {}", target_model, load_err);
                        Err(anyhow!("Failed to load Parakeet model '{}': {}", target_model, load_err))
                    }
                }
            } else {
                info!("Parakeet model '{}' already loaded", target_model);
                Ok(e)
            }
        }
        None => Err(anyhow!("Parakeet engine not initialized")),
    }
}

/// Get the configured Parakeet model name from the database
async fn get_configured_parakeet_model<R: Runtime>(app: &AppHandle<R>) -> Result<String> {
    debug!("Getting configured Parakeet model from database...");

    let app_state = app
        .try_state::<AppState>()
        .ok_or_else(|| {
            error!("App state not available");
            anyhow!("App state not available")
        })?;

    // Query the transcript settings from the database
    let result: Option<(String, String)> = sqlx::query_as(
        "SELECT provider, model FROM transcript_settings WHERE id = '1'"
    )
    .fetch_optional(app_state.db_manager.pool())
    .await
    .map_err(|e| {
        error!("Failed to query transcript config: {}", e);
        anyhow!("Failed to query transcript config: {}", e)
    })?;

    match result {
        Some((provider, model)) => {
            info!("Found transcript config: provider={}, model={}", provider, model);

            if provider == "parakeet" {
                Ok(model)
            } else {
                // Default to configured Parakeet model
                warn!("Configured provider is not Parakeet, using default model");
                Ok(DEFAULT_PARAKEET_MODEL.to_string())
            }
        },
        None => {
            // Default to configured Parakeet model if no config exists
            warn!("No transcript config found, using default Parakeet model");
            Ok(DEFAULT_PARAKEET_MODEL.to_string())
        }
    }
}

/// Write or update metadata.json for retranscription (preserves existing fields, adds retranscribed_at)
fn write_retranscription_metadata(
    folder: &Path,
    meeting_id: &str,
    duration_seconds: f64,
    audio_filename: &str,
) -> Result<()> {
    let metadata_path = folder.join("metadata.json");
    let temp_path = folder.join(".metadata.json.tmp");
    let now = chrono::Utc::now().to_rfc3339();

    // Try to read existing metadata and update it
    let json = if metadata_path.exists() {
        let existing = std::fs::read_to_string(&metadata_path)?;
        let mut value: serde_json::Value = serde_json::from_str(&existing)?;
        if let Some(obj) = value.as_object_mut() {
            obj.insert("retranscribed_at".to_string(), serde_json::json!(now));
            obj.insert("status".to_string(), serde_json::json!("completed"));
            obj.insert("transcript_file".to_string(), serde_json::json!("transcripts.json"));
            obj.remove("detected_summary_language");
        }
        value
    } else {
        serde_json::json!({
            "version": "1.0",
            "meeting_id": meeting_id,
            "created_at": now,
            "completed_at": now,
            "retranscribed_at": now,
            "duration_seconds": duration_seconds,
            "audio_file": audio_filename,
            "transcript_file": "transcripts.json",
            "status": "completed",
            "source": "retranscription"
        })
    };

    let json_string = serde_json::to_string_pretty(&json)?;
    std::fs::write(&temp_path, &json_string)?;
    std::fs::rename(&temp_path, &metadata_path)?;

    info!("Wrote metadata.json to {}", metadata_path.display());
    Ok(())
}

// Tauri commands

/// Response when retranscription is started
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetranscriptionStarted {
    pub meeting_id: String,
    pub message: String,
}

// Start retranscription (Beta gated using configContext.betaFeatures)
#[tauri::command]
pub async fn start_retranscription_command<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    meeting_folder_path: String,
    language: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    diarize: Option<DiarizeOptions>,
    repair: Option<bool>,
) -> Result<RetranscriptionStarted, String> {

    // Check if retranscription is already in progress (guard will be acquired in start_retranscription)
    if RETRANSCRIPTION_IN_PROGRESS.load(Ordering::SeqCst) {
        return Err("Retranscription already in progress".to_string());
    }

    // Clone values for the spawned task
    let meeting_id_clone = meeting_id.clone();

    // Spawn the retranscription in a background task
    tauri::async_runtime::spawn(async move {
        let result = start_retranscription(
            app,
            meeting_id_clone,
            meeting_folder_path,
            language,
            model,
            provider,
            diarize,
            repair,
        )
        .await;

        // Errors are already emitted as events in start_retranscription
        // so we just log here for debugging
        if let Err(e) = result {
            error!("Retranscription failed: {}", e);
        }
    });

    Ok(RetranscriptionStarted {
        meeting_id,
        message: "Retranscription started".to_string(),
    })
}

#[tauri::command]
pub async fn cancel_retranscription_command() -> Result<(), String> {
    if !is_retranscription_in_progress() {
        return Err("No retranscription in progress".to_string());
    }
    cancel_retranscription();
    Ok(())
}

#[tauri::command]
pub async fn is_retranscription_in_progress_command() -> bool {
    is_retranscription_in_progress()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_transcript_segments_empty() {
        let transcripts: Vec<(String, f64, f64)> = vec![];
        let segments = create_transcript_segments(&transcripts);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_create_transcript_segments_single() {
        let transcripts = vec![
            ("Hello world".to_string(), 0.0, 1500.0), // 0-1.5 seconds
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello world");
        assert_eq!(segments[0].audio_start_time, Some(0.0));
        assert_eq!(segments[0].audio_end_time, Some(1.5));
        assert_eq!(segments[0].duration, Some(1.5));
    }

    #[test]
    fn test_create_transcript_segments_multiple() {
        let transcripts = vec![
            ("First segment".to_string(), 0.0, 2000.0),      // 0-2 seconds
            ("Second segment".to_string(), 3000.0, 5000.0),  // 3-5 seconds
            ("Third segment".to_string(), 6500.0, 8000.0),   // 6.5-8 seconds
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 3);

        // First segment
        assert_eq!(segments[0].text, "First segment");
        assert_eq!(segments[0].audio_start_time, Some(0.0));
        assert_eq!(segments[0].audio_end_time, Some(2.0));
        assert_eq!(segments[0].duration, Some(2.0));

        // Second segment
        assert_eq!(segments[1].text, "Second segment");
        assert_eq!(segments[1].audio_start_time, Some(3.0));
        assert_eq!(segments[1].audio_end_time, Some(5.0));
        assert_eq!(segments[1].duration, Some(2.0));

        // Third segment
        assert_eq!(segments[2].text, "Third segment");
        assert_eq!(segments[2].audio_start_time, Some(6.5));
        assert_eq!(segments[2].audio_end_time, Some(8.0));
        assert_eq!(segments[2].duration, Some(1.5));
    }

    #[test]
    fn test_create_transcript_segments_trims_whitespace() {
        let transcripts = vec![
            ("  Hello with spaces  ".to_string(), 0.0, 1000.0),
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello with spaces");
    }

    #[test]
    fn test_create_transcript_segments_generates_unique_ids() {
        let transcripts = vec![
            ("Segment one".to_string(), 0.0, 1000.0),
            ("Segment two".to_string(), 1000.0, 2000.0),
        ];
        let segments = create_transcript_segments(&transcripts);

        assert_eq!(segments.len(), 2);
        assert_ne!(segments[0].id, segments[1].id);
        assert!(segments[0].id.starts_with("transcript-"));
        assert!(segments[1].id.starts_with("transcript-"));
    }

    #[test]
    fn test_cancellation_flag() {
        // Reset flag to known state
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
        RETRANSCRIPTION_IN_PROGRESS.store(false, Ordering::SeqCst);

        assert!(!is_retranscription_in_progress());

        // Test cancellation
        cancel_retranscription();
        assert!(RETRANSCRIPTION_CANCELLED.load(Ordering::SeqCst));

        // Reset for other tests
        RETRANSCRIPTION_CANCELLED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn test_vad_redemption_time_constant() {
        // Batch processing uses 2000ms to bridge natural pauses in full-file VAD
        assert_eq!(VAD_REDEMPTION_TIME_MS, 2000);
    }

    #[test]
    fn test_find_audio_file_common_candidates() {
        let dir = tempfile::tempdir().unwrap();

        // No audio file → error
        assert!(find_audio_file(dir.path()).is_err());

        // Create audio.mp4 — should be found first
        std::fs::write(dir.path().join("audio.mp4"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.mp4");
    }

    #[test]
    fn test_find_audio_file_non_mp4_extensions() {
        let dir = tempfile::tempdir().unwrap();

        // Create audio.wav (imported as .wav, not .mp4)
        std::fs::write(dir.path().join("audio.wav"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.wav");
    }

    #[test]
    fn test_find_audio_file_fallback_scan() {
        let dir = tempfile::tempdir().unwrap();

        // Create a file with an audio extension but non-standard name
        std::fs::write(dir.path().join("my_recording.flac"), b"fake").unwrap();
        // Also add a non-audio file that should be ignored
        std::fs::write(dir.path().join("notes.txt"), b"text").unwrap();

        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "my_recording.flac");
    }

    #[test]
    fn test_find_audio_file_priority_order() {
        let dir = tempfile::tempdir().unwrap();

        // Create both audio.m4a and audio.mp4 — mp4 should win (listed first in candidates)
        std::fs::write(dir.path().join("audio.m4a"), b"fake").unwrap();
        std::fs::write(dir.path().join("audio.mp4"), b"fake").unwrap();
        let found = find_audio_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "audio.mp4");
    }

    #[test]
    fn test_find_audio_file_empty_folder() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_audio_file(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No audio file found"));
    }

    #[test]
    fn test_find_audio_file_nonexistent_folder() {
        let result = find_audio_file(Path::new("/nonexistent/path/12345"));
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_extensions_constant() {
        // Verify all expected formats are covered
        assert!(AUDIO_EXTENSIONS.contains(&"mp4"));
        assert!(AUDIO_EXTENSIONS.contains(&"m4a"));
        assert!(AUDIO_EXTENSIONS.contains(&"wav"));
        assert!(AUDIO_EXTENSIONS.contains(&"mp3"));
        assert!(AUDIO_EXTENSIONS.contains(&"flac"));
        assert!(AUDIO_EXTENSIONS.contains(&"ogg"));
        assert!(AUDIO_EXTENSIONS.contains(&"aac"));
        // FFmpeg-backed formats
        assert!(AUDIO_EXTENSIONS.contains(&"mkv"));
        assert!(AUDIO_EXTENSIONS.contains(&"webm"));
        assert!(AUDIO_EXTENSIONS.contains(&"wma"));
        // Non-audio formats
        assert!(!AUDIO_EXTENSIONS.contains(&"txt"));
        assert!(!AUDIO_EXTENSIONS.contains(&"pdf"));
    }
}
