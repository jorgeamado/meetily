use std::ffi::CString;
use std::os::raw::{c_float, c_void};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use sherpa_onnx_sys as sys;

extern "C" {
    fn SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
        sd: *const sys::OfflineSpeakerDiarization,
        samples: *const c_float,
        n: i32,
        callback: extern "C" fn(i32, i32, *mut c_void) -> i32,
        arg: *mut c_void,
    ) -> *const sys::OfflineSpeakerDiarizationResult;
}

struct ProgressState {
    last_pct: i32,
}

extern "C" fn diarization_progress_callback(
    num_processed_chunks: i32,
    num_total_chunks: i32,
    arg: *mut c_void,
) -> i32 {
    if num_total_chunks <= 0 {
        return 0;
    }
    let state = unsafe { &mut *(arg as *mut ProgressState) };
    let pct = ((num_processed_chunks as i64 * 100) / num_total_chunks as i64).clamp(0, 100) as i32;
    if pct > state.last_pct {
        state.last_pct = pct;
        eprintln!("PROGRESS {}", pct);
    }
    0
}

struct Diarizer(*const sys::OfflineSpeakerDiarization);

impl Diarizer {
    fn create(config: &sys::OfflineSpeakerDiarizationConfig) -> Option<Self> {
        let ptr = unsafe { sys::SherpaOnnxCreateOfflineSpeakerDiarization(config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self(ptr))
        }
    }

    fn sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(self.0) }
    }

    fn process_with_progress(&self, samples: &[f32]) -> Option<DiarizationResult> {
        let mut state = ProgressState { last_pct: -1 };
        let ptr = unsafe {
            SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                self.0,
                samples.as_ptr(),
                samples.len() as i32,
                diarization_progress_callback,
                &mut state as *mut ProgressState as *mut c_void,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(DiarizationResult(ptr))
        }
    }
}

impl Drop for Diarizer {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                sys::SherpaOnnxDestroyOfflineSpeakerDiarization(self.0);
            }
        }
    }
}

struct DiarizationResult(*const sys::OfflineSpeakerDiarizationResult);

impl DiarizationResult {
    fn sort_by_start_time(&self) -> Vec<sys::OfflineSpeakerDiarizationSegment> {
        let n = unsafe { sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(self.0) };
        if n <= 0 {
            return Vec::new();
        }
        unsafe {
            let p = sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(self.0);
            if p.is_null() {
                return Vec::new();
            }
            let segments = std::slice::from_raw_parts(p, n as usize).to_vec();
            sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(p);
            segments
        }
    }
}

impl Drop for DiarizationResult {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(self.0);
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "diarize-helper")]
struct Args {
    #[arg(long)]
    audio: PathBuf,

    #[arg(long = "segmentation-model")]
    segmentation_model: PathBuf,

    #[arg(long = "embedding-model")]
    embedding_model: PathBuf,

    #[arg(long = "num-speakers")]
    num_speakers: Option<i32>,

    #[arg(long, default_value_t = 0.9)]
    threshold: f32,

    #[arg(long = "min-duration-on", default_value_t = 0.3)]
    min_duration_on: f32,

    #[arg(long = "min-duration-off", default_value_t = 0.5)]
    min_duration_off: f32,
}

#[derive(Serialize)]
struct Segment {
    start: f32,
    end: f32,
    speaker: i32,
}

#[derive(Serialize)]
struct Output {
    segments: Vec<Segment>,
    num_speakers: i32,
}

fn read_wav_as_f32_mono(path: &PathBuf) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav file: {}", path.display()))?;
    let spec = reader.spec();

    eprintln!(
        "audio: {} Hz, {} channel(s), {} bits, format {:?}",
        spec.sample_rate, spec.channels, spec.bits_per_sample, spec.sample_format
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read f32 samples from wav")?,
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read s16 samples from wav")?,
            32 => reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read s32 samples from wav")?,
            bits => bail!("unsupported wav bit depth: {}", bits),
        },
    };

    let channels = spec.channels as usize;
    let mono = if channels <= 1 {
        samples
    } else {
        eprintln!("downmixing {} channels to mono", channels);
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, spec.sample_rate))
}

fn run(args: Args) -> Result<Output> {
    let (samples, sample_rate) = read_wav_as_f32_mono(&args.audio)?;

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4) as i32;
    eprintln!("using {} threads", num_threads);

    let seg_model_str = args
        .segmentation_model
        .to_str()
        .context("segmentation-model path is not valid UTF-8")?;
    let emb_model_str = args
        .embedding_model
        .to_str()
        .context("embedding-model path is not valid UTF-8")?;
    let seg_model = CString::new(seg_model_str).context("segmentation-model path contains a NUL byte")?;
    let emb_model = CString::new(emb_model_str).context("embedding-model path contains a NUL byte")?;
    let cpu_provider = CString::new("cpu").expect("static string has no NUL byte");

    let config = sys::OfflineSpeakerDiarizationConfig {
        segmentation: sys::OfflineSpeakerSegmentationModelConfig {
            pyannote: sys::OfflineSpeakerSegmentationPyannoteModelConfig {
                model: seg_model.as_ptr(),
            },
            num_threads,
            debug: 0,
            provider: cpu_provider.as_ptr(),
        },
        embedding: sys::SpeakerEmbeddingExtractorConfig {
            model: emb_model.as_ptr(),
            num_threads,
            debug: 0,
            provider: cpu_provider.as_ptr(),
        },
        clustering: sys::FastClusteringConfig {
            num_clusters: args.num_speakers.unwrap_or(-1),
            threshold: args.threshold,
        },
        min_duration_on: args.min_duration_on,
        min_duration_off: args.min_duration_off,
    };

    eprintln!("creating offline speaker diarizer");
    let diarizer = Diarizer::create(&config)
        .context("failed to create offline speaker diarization pipeline")?;

    let expected_rate = diarizer.sample_rate();
    if expected_rate as u32 != sample_rate {
        eprintln!(
            "warning: model expects {} Hz audio but input is {} Hz; no resampling is performed",
            expected_rate, sample_rate
        );
    }

    eprintln!(
        "processing {} samples ({:.2}s of audio)",
        samples.len(),
        samples.len() as f32 / sample_rate as f32
    );

    let result = diarizer
        .process_with_progress(&samples)
        .context("diarization processing failed")?;

    let raw_segments = result.sort_by_start_time();

    let mut speaker_ids: Vec<i32> = Vec::new();
    for s in &raw_segments {
        if !speaker_ids.contains(&s.speaker) {
            speaker_ids.push(s.speaker);
        }
    }

    let segments = raw_segments
        .into_iter()
        .map(|s| Segment {
            start: s.start,
            end: s.end,
            speaker: speaker_ids
                .iter()
                .position(|&id| id == s.speaker)
                .expect("speaker id was just collected above") as i32,
        })
        .collect();

    let num_speakers = speaker_ids.len() as i32;
    eprintln!("done: {} speaker(s) detected", num_speakers);

    Ok(Output {
        segments,
        num_speakers,
    })
}

fn main() {
    let args = Args::parse();

    match run(args) {
        Ok(output) => {
            let json = serde_json::to_string(&output).expect("failed to serialize output");
            println!("{}", json);
        }
        Err(err) => {
            let error_json = serde_json::json!({ "error": err.to_string() });
            eprintln!("{}", error_json);
            std::process::exit(1);
        }
    }
}
