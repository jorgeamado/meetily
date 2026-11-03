// End-to-end batch retranscription benchmark binary for Whisper vs FluidAudio Parakeet.
//
// Mirrors `run_retranscription` in `frontend/src-tauri/src/audio/retranscription.rs`
// stage by stage (decode -> vad -> load -> transcribe -> split) using the app's
// own library code, timing each stage with `Instant`.
//
// Usage:
//   batchbench --audio <file> --engine whisper|fluid --diarization <segments.json> \
//              --out <result.json> [--whisper-model large-v3-turbo] \
//              [--fluid-model fluid-parakeet-v3] [--models-dir <dir>]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use app_lib::audio::common::split_segment_at_silence;
use app_lib::audio::decoder::decode_audio_file;
use app_lib::audio::diarization::{
    self, DiarizedSegment, SpeakerTurn, WordSpan,
};
use app_lib::audio::retranscription::VAD_REDEMPTION_TIME_MS;
use app_lib::audio::vad::{get_speech_chunks_with_progress, SpeechSegment};
use app_lib::fluid_engine::{FluidJobGuard, FluidModel, FLUID_ENGINE};
use app_lib::whisper_engine::WhisperEngine;
use serde::{Deserialize, Serialize};

const DEFAULT_WHISPER_MODEL: &str = "large-v3-turbo";
const DEFAULT_FLUID_MODEL: &str = "fluid-parakeet-v3";
const DEFAULT_MODELS_SUBPATH: &str = "Library/Application Support/com.meetily.ai/models";
const MAX_SEGMENT_SAMPLES: usize = 25 * 16000; // 25 seconds at 16kHz
const MIN_SEGMENT_SAMPLES: usize = 1600; // 100ms at 16kHz

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineChoice {
    Whisper,
    Fluid,
}

impl EngineChoice {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "whisper" => Some(Self::Whisper),
            "fluid" => Some(Self::Fluid),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Fluid => "fluid",
        }
    }
}

enum LoadedBenchEngine {
    Whisper(WhisperEngine),
    Fluid(FluidJobGuard),
}

#[derive(Debug, Deserialize)]
struct HelperSegment {
    start: f32,
    end: f32,
    speaker: i32,
    #[serde(default)]
    voice: bool,
}

#[derive(Debug, Deserialize)]
struct HelperOutput {
    segments: Vec<HelperSegment>,
    #[serde(default)]
    #[allow(dead_code)]
    num_speakers: i32,
}

#[derive(Debug, Serialize)]
struct TimingsSecs {
    decode: f64,
    vad: f64,
    load: f64,
    transcribe: f64,
    split: f64,
    total: f64,
}

#[derive(Debug, Serialize)]
struct BlockOut {
    start_ms: f64,
    end_ms: f64,
    text: String,
    n_words: usize,
}

#[derive(Debug, Serialize)]
struct TurnOut {
    speaker: Option<String>,
    start_ms: f64,
    end_ms: f64,
    text: String,
}

#[derive(Debug, Serialize)]
struct BenchOutput {
    engine: String,
    model: String,
    audio_secs: f64,
    timings_secs: TimingsSecs,
    segments: usize,
    blocks: Vec<BlockOut>,
    turns: Vec<TurnOut>,
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if path == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}

fn default_models_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME environment variable must be set");
    PathBuf::from(home).join(DEFAULT_MODELS_SUBPATH)
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: batchbench --audio <file> --engine whisper|fluid \
         --diarization <segments.json> --out <result.json> \
         [--whisper-model large-v3-turbo] [--fluid-model fluid-parakeet-v3] \
         [--models-dir <dir>]"
    );
    std::process::exit(2);
}

fn ensure_fluid_helper_discoverable() {
    if std::env::var_os("MEETILY_FLUID_HELPER").is_none()
        && std::env::var_os("RESOURCE_DIR").is_none()
    {
        if let Some(repo_root) = Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
            let binaries_dir = repo_root.join("frontend/src-tauri/binaries");
            if binaries_dir.is_dir() {
                std::env::set_var("RESOURCE_DIR", &binaries_dir);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut audio_path: Option<PathBuf> = None;
    let mut engine_choice: Option<EngineChoice> = None;
    let mut diarization_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut whisper_model = String::from(DEFAULT_WHISPER_MODEL);
    let mut fluid_model = String::from(DEFAULT_FLUID_MODEL);
    let mut models_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--audio" => {
                let Some(val) = args.get(i + 1) else { usage_and_exit() };
                audio_path = Some(expand_tilde(val));
                i += 2;
            }
            "--engine" => {
                let Some(val) = args.get(i + 1) else { usage_and_exit() };
                engine_choice = Some(EngineChoice::parse(val).unwrap_or_else(|| {
                    eprintln!("unknown --engine '{}': expected 'whisper' or 'fluid'", val);
                    usage_and_exit();
                }));
                i += 2;
            }
            "--diarization" => {
                let Some(val) = args.get(i + 1) else { usage_and_exit() };
                diarization_path = Some(expand_tilde(val));
                i += 2;
            }
            "--out" => {
                let Some(val) = args.get(i + 1) else { usage_and_exit() };
                out_path = Some(expand_tilde(val));
                i += 2;
            }
            "--whisper-model" => {
                let Some(val) = args.get(i + 1) else { usage_and_exit() };
                whisper_model = val.clone();
                i += 2;
            }
            "--fluid-model" => {
                let Some(val) = args.get(i + 1) else { usage_and_exit() };
                fluid_model = val.clone();
                i += 2;
            }
            "--models-dir" => {
                let Some(val) = args.get(i + 1) else { usage_and_exit() };
                models_dir = Some(expand_tilde(val));
                i += 2;
            }
            "-h" | "--help" => usage_and_exit(),
            other => {
                eprintln!("unknown argument: {}", other);
                usage_and_exit();
            }
        }
    }

    let Some(audio_path) = audio_path else { usage_and_exit() };
    let Some(engine_choice) = engine_choice else { usage_and_exit() };
    let Some(diarization_path) = diarization_path else { usage_and_exit() };
    let Some(out_path) = out_path else { usage_and_exit() };
    let models_dir = models_dir.unwrap_or_else(default_models_dir);

    ensure_fluid_helper_discoverable();

    app_lib::async_runtime::block_on(async move {
        let total_start = Instant::now();

        // 1. decode: load audio to 16 kHz mono f32 and record audio_secs
        let t_decode = Instant::now();
        let decoded = decode_audio_file(&audio_path).expect("decode audio file");
        let audio_secs = decoded.duration_seconds;
        let audio_samples = decoded.to_whisper_format();
        let decode_secs = t_decode.elapsed().as_secs_f64();

        // 2. vad: detect speech segments with VAD_REDEMPTION_TIME_MS (2000ms),
        // split segments > 25s at silence boundaries, and skip segments < 1600 samples
        let t_vad = Instant::now();
        let speech_segments = get_speech_chunks_with_progress(
            &audio_samples,
            VAD_REDEMPTION_TIME_MS,
            |_, _| true,
        )
        .expect("VAD processing failed");

        let mut processable_segments: Vec<SpeechSegment> = Vec::new();
        for segment in &speech_segments {
            if segment.samples.len() > MAX_SEGMENT_SAMPLES {
                let sub_segments = split_segment_at_silence(segment, MAX_SEGMENT_SAMPLES);
                processable_segments.extend(sub_segments);
            } else {
                processable_segments.push(segment.clone());
            }
        }
        processable_segments.retain(|seg| seg.samples.len() >= MIN_SEGMENT_SAMPLES);
        let segments_count = processable_segments.len();
        let vad_secs = t_vad.elapsed().as_secs_f64();

        // 3. load: initialize and load the selected engine
        let t_load = Instant::now();
        let (loaded_engine, model_name) = match engine_choice {
            EngineChoice::Whisper => {
                let engine = WhisperEngine::new_with_models_dir(Some(models_dir.clone()))
                    .expect("initialize WhisperEngine");
                engine
                    .discover_models()
                    .await
                    .expect("discover Whisper models");
                engine
                    .load_model_with_dtw(&whisper_model, true)
                    .await
                    .expect("load Whisper model with DTW");
                (LoadedBenchEngine::Whisper(engine), whisper_model.clone())
            }
            EngineChoice::Fluid => {
                let parsed_model = FluidModel::from_model_id(&fluid_model)
                    .unwrap_or_else(|| panic!("unknown FluidAudio model '{}'", fluid_model));
                let fluid_models_dir = models_dir.join("fluidaudio");
                let guard = FLUID_ENGINE
                    .begin_job(parsed_model, &fluid_models_dir, |_| {})
                    .await
                    .expect("load FluidAudio model");
                (
                    LoadedBenchEngine::Fluid(guard),
                    parsed_model.model_id().to_string(),
                )
            }
        };
        let load_secs = t_load.elapsed().as_secs_f64();

        // 4. transcribe: transcribe each segment in order and shift word timestamps
        // onto the recording timeline by segment.start_timestamp_ms
        let t_transcribe = Instant::now();
        let mut all_transcripts: Vec<(String, f64, f64, Vec<WordSpan>)> = Vec::new();

        for (idx, segment) in processable_segments.iter().enumerate() {
            if segment.samples.len() < MIN_SEGMENT_SAMPLES {
                continue;
            }

            let (text, _conf, words) = match &loaded_engine {
                LoadedBenchEngine::Whisper(engine) => {
                    let (text, conf, words) = engine
                        .transcribe_audio_with_words(segment.samples.clone(), None)
                        .await
                        .unwrap_or_else(|e| {
                            panic!("Whisper transcription failed on segment {}: {}", idx, e)
                        });
                    let words = words
                        .into_iter()
                        .map(|w| WordSpan {
                            text: w.text,
                            start_ms: w.start_ms + segment.start_timestamp_ms,
                            end_ms: w.end_ms + segment.start_timestamp_ms,
                            prob: w.prob,
                        })
                        .collect();
                    (text, conf, words)
                }
                LoadedBenchEngine::Fluid(guard) => {
                    let transcript = guard.transcribe(&segment.samples).await.unwrap_or_else(|e| {
                        panic!("FluidAudio transcription failed on segment {}: {}", idx, e)
                    });
                    let words = transcript
                        .words
                        .into_iter()
                        .map(|w| WordSpan {
                            text: w.text,
                            start_ms: w.start_ms + segment.start_timestamp_ms,
                            end_ms: w.end_ms + segment.start_timestamp_ms,
                            prob: w.prob,
                        })
                        .collect();
                    (transcript.text, transcript.confidence, words)
                }
            };

            let trimmed = text.trim();
            if !trimmed.is_empty() {
                all_transcripts.push((
                    text,
                    segment.start_timestamp_ms,
                    segment.end_timestamp_ms,
                    words,
                ));
            }
        }
        let transcribe_secs = t_transcribe.elapsed().as_secs_f64();

        // 5. split: load diarization JSON (diarize-helper stdout format) and split blocks
        // into speaker turns using word timestamps.
        // Note: stereo local-speaker overlay, LLM boundary refinement, transcript repair,
        // and glossary biasing/learning are intentionally skipped here because those stages
        // are identical across engines and orthogonal to raw engine speed/turn splitting.
        let t_split = Instant::now();
        let diarization_raw =
            fs::read_to_string(&diarization_path).expect("read diarization JSON file");
        let helper_out: HelperOutput =
            serde_json::from_str(&diarization_raw).expect("parse diarization JSON");
        let diarized_segments: Vec<DiarizedSegment> = helper_out
            .segments
            .into_iter()
            .map(|s| DiarizedSegment {
                start: s.start,
                end: s.end,
                speaker: s.speaker.max(0) as usize,
                voice: s.voice,
            })
            .collect();

        let turns: Vec<SpeakerTurn> = all_transcripts
            .iter()
            .flat_map(|(text, start_ms, end_ms, words)| {
                if diarized_segments.is_empty() {
                    vec![SpeakerTurn {
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

        let labeled_rows = diarization::turns_to_rows(&turns);
        let turns_out: Vec<TurnOut> = labeled_rows
            .into_iter()
            .map(|(text, start_ms, end_ms, speaker_idx)| TurnOut {
                speaker: speaker_idx.map(diarization::speaker_label),
                start_ms,
                end_ms,
                text,
            })
            .collect();
        let split_secs = t_split.elapsed().as_secs_f64();
        let total_secs = total_start.elapsed().as_secs_f64();

        // Unload engine after benchmark completes
        match loaded_engine {
            LoadedBenchEngine::Whisper(engine) => {
                engine.unload_model().await;
            }
            LoadedBenchEngine::Fluid(guard) => {
                drop(guard);
                FLUID_ENGINE.unload_if_idle().await;
            }
        }

        // 6. Write --out JSON and print one-line summary
        let blocks_out: Vec<BlockOut> = all_transcripts
            .into_iter()
            .map(|(text, start_ms, end_ms, words)| BlockOut {
                start_ms,
                end_ms,
                text: text.trim().to_string(),
                n_words: words.len(),
            })
            .collect();

        let output = BenchOutput {
            engine: engine_choice.as_str().to_string(),
            model: model_name.clone(),
            audio_secs,
            timings_secs: TimingsSecs {
                decode: decode_secs,
                vad: vad_secs,
                load: load_secs,
                transcribe: transcribe_secs,
                split: split_secs,
                total: total_secs,
            },
            segments: segments_count,
            blocks: blocks_out,
            turns: turns_out,
        };

        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).expect("create output directory");
            }
        }
        let json_str = serde_json::to_string_pretty(&output).expect("serialize output JSON");
        fs::write(&out_path, json_str).expect("write output JSON");

        let audio_mins = audio_secs / 60.0;
        let rtf = if audio_secs > 0.0 {
            transcribe_secs / audio_secs
        } else {
            0.0
        };
        let speedup = if transcribe_secs > 0.0 {
            audio_secs / transcribe_secs
        } else {
            0.0
        };
        println!(
            "engine={} model={} audio_min={:.2} transcribe_secs={:.2} rtf={:.4} ({:.1}x realtime)",
            engine_choice.as_str(),
            model_name,
            audio_mins,
            transcribe_secs,
            rtf,
            speedup
        );
    });
}
