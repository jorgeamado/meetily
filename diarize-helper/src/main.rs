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

/// Post-clustering over-split repair. FastClustering routinely shatters one
/// voice into several clusters (observed: 8 clusters on a 3-4 person call).
/// Verified on clean speech the split halves score cos >= 0.93 while genuinely
/// different voices stay <= 0.68, so a 0.8 merge bar has wide margin.
const CLEAN_MIN_SECS: f32 = 1.5;
const CLEAN_CAP_SECS: f32 = 12.0;
const MAJOR_MIN_CLEAN_SECS: f32 = 10.0;
const DEBRIS_MIN_SECS: f32 = 0.25;
const DEBRIS_VOICE_FLOOR: f32 = 0.60;

struct EmbeddingExtractor {
    ptr: *const sys::SpeakerEmbeddingExtractor,
    dim: usize,
}

impl EmbeddingExtractor {
    fn create(config: &sys::SpeakerEmbeddingExtractorConfig) -> Option<Self> {
        let ptr = unsafe { sys::SherpaOnnxCreateSpeakerEmbeddingExtractor(config) };
        if ptr.is_null() {
            None
        } else {
            let dim = unsafe { sys::SherpaOnnxSpeakerEmbeddingExtractorDim(ptr) } as usize;
            Some(Self { ptr, dim })
        }
    }

    /// L2-normalized speaker embedding of a span of samples, or None if the
    /// span is too short for the model to produce one.
    fn embed(&self, samples: &[f32], sample_rate: i32) -> Option<Vec<f32>> {
        if samples.is_empty() {
            return None;
        }
        unsafe {
            let stream = sys::SherpaOnnxSpeakerEmbeddingExtractorCreateStream(self.ptr);
            if stream.is_null() {
                return None;
            }
            sys::SherpaOnnxOnlineStreamAcceptWaveform(
                stream,
                sample_rate,
                samples.as_ptr(),
                samples.len() as i32,
            );
            sys::SherpaOnnxOnlineStreamInputFinished(stream);
            let out = if sys::SherpaOnnxSpeakerEmbeddingExtractorIsReady(self.ptr, stream) != 0 {
                let v = sys::SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(self.ptr, stream);
                if v.is_null() {
                    None
                } else {
                    let e = std::slice::from_raw_parts(v, self.dim).to_vec();
                    sys::SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(v);
                    normalized(e)
                }
            } else {
                None
            };
            sys::SherpaOnnxDestroyOnlineStream(stream);
            out
        }
    }
}

impl Drop for EmbeddingExtractor {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                sys::SherpaOnnxDestroySpeakerEmbeddingExtractor(self.ptr);
            }
        }
    }
}

fn normalized(mut v: Vec<f32>) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-6 {
        return None;
    }
    for x in &mut v {
        *x /= norm;
    }
    Some(v)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Embed [start, end] seconds of audio, center-cropped to CLEAN_CAP_SECS.
fn embed_span(
    extractor: &EmbeddingExtractor,
    samples: &[f32],
    sample_rate: i32,
    start: f32,
    end: f32,
) -> Option<Vec<f32>> {
    let (mut s, mut e) = (start, end);
    if e - s > CLEAN_CAP_SECS {
        let mid = (s + e) / 2.0;
        s = mid - CLEAN_CAP_SECS / 2.0;
        e = mid + CLEAN_CAP_SECS / 2.0;
    }
    let sr = sample_rate as f32;
    let lo = ((s * sr) as usize).min(samples.len());
    let hi = ((e * sr) as usize).min(samples.len());
    extractor.embed(&samples[lo..hi], sample_rate)
}

/// Merge same-voice clusters and fold tiny "debris" clusters into the real
/// speakers. Only clean spans (>= CLEAN_MIN_SECS, no time overlap with any
/// other segment) vote on cluster identity — overlapped spans carry mixed
/// voices and previously made split halves of one voice look distinct.
/// Segments are expected to be sorted by start time.
fn repair_oversplit(
    segments: &mut [sys::OfflineSpeakerDiarizationSegment],
    samples: &[f32],
    sample_rate: i32,
    extractor: &EmbeddingExtractor,
    merge_threshold: f32,
) {
    use std::collections::HashMap;

    let n = segments.len();
    if n < 2 {
        return;
    }

    let mut clean = vec![false; n];
    for i in 0..n {
        let a = &segments[i];
        if a.end - a.start < CLEAN_MIN_SECS {
            continue;
        }
        let mut ok = true;
        for j in (0..i).rev() {
            if segments[j].end > a.start {
                ok = false;
                break;
            }
            if a.start - segments[j].end > 15.0 {
                break;
            }
        }
        if ok && i + 1 < n && segments[i + 1].start < a.end {
            ok = false;
        }
        clean[i] = ok;
    }

    // Duration-weighted centroid of clean spans per cluster.
    let mut sums: HashMap<i32, (Vec<f32>, f32)> = HashMap::new();
    for i in 0..n {
        if !clean[i] {
            continue;
        }
        let seg = &segments[i];
        let Some(e) = embed_span(extractor, samples, sample_rate, seg.start, seg.end) else {
            continue;
        };
        let d = seg.end - seg.start;
        let entry = sums.entry(seg.speaker).or_insert_with(|| (vec![0.0; e.len()], 0.0));
        for (acc, x) in entry.0.iter_mut().zip(&e) {
            *acc += x * d;
        }
        entry.1 += d;
    }

    let mut centroids: HashMap<i32, Vec<f32>> = HashMap::new();
    let mut clean_time: HashMap<i32, f32> = HashMap::new();
    for (k, (sum, w)) in sums {
        if let Some(c) = normalized(sum) {
            centroids.insert(k, c);
            clean_time.insert(k, w);
        }
    }

    let mut majors: Vec<i32> = centroids
        .keys()
        .copied()
        .filter(|k| clean_time[k] >= MAJOR_MIN_CLEAN_SECS)
        .collect();
    majors.sort_unstable();
    if majors.is_empty() {
        eprintln!("oversplit: no cluster has enough clean speech; leaving clusters unchanged");
        return;
    }

    // Iteratively merge the closest major pair while it clears the bar.
    let mut remap: HashMap<i32, i32> = HashMap::new();
    let mut weight: HashMap<i32, f32> = clean_time.clone();
    loop {
        let mut best = (f32::MIN, 0usize, 0usize);
        for x in 0..majors.len() {
            for y in (x + 1)..majors.len() {
                let s = cosine(&centroids[&majors[x]], &centroids[&majors[y]]);
                if s > best.0 {
                    best = (s, x, y);
                }
            }
        }
        if majors.len() < 2 || best.0 < merge_threshold {
            if majors.len() >= 2 {
                eprintln!(
                    "oversplit: closest major pair {} & {} at cos {:.3} (< {:.2}); majors: {:?}",
                    majors[best.1], majors[best.2], best.0, merge_threshold, majors
                );
            }
            break;
        }
        let (keep, gone) = (majors[best.1], majors[best.2]);
        eprintln!(
            "oversplit: merged speaker cluster {} into {} (cos {:.3})",
            gone, keep, best.0
        );
        let (wk, wg) = (weight[&keep], weight[&gone]);
        let merged: Vec<f32> = centroids[&keep]
            .iter()
            .zip(&centroids[&gone])
            .map(|(a, b)| a * wk + b * wg)
            .collect();
        if let Some(c) = normalized(merged) {
            centroids.insert(keep, c);
        }
        weight.insert(keep, wk + wg);
        centroids.remove(&gone);
        majors.remove(best.2);
        for v in remap.values_mut() {
            if *v == gone {
                *v = keep;
            }
        }
        remap.insert(gone, keep);
    }

    // Relabel major segments first so debris has final labels to fall back on.
    let mut major_mids: Vec<(f32, i32)> = Vec::new();
    for seg in segments.iter_mut() {
        let mapped = remap.get(&seg.speaker).copied().unwrap_or(seg.speaker);
        if majors.contains(&mapped) {
            seg.speaker = mapped;
            major_mids.push(((seg.start + seg.end) / 2.0, mapped));
        }
    }

    // Debris: confident voice match wins; otherwise fold into the temporally
    // nearest real speaker so stray blips join the surrounding conversation
    // instead of surfacing as phantom speakers.
    let (mut by_voice, mut by_time, mut debris_secs) = (0usize, 0usize, 0f32);
    for i in 0..n {
        let spk = segments[i].speaker;
        let mapped = remap.get(&spk).copied().unwrap_or(spk);
        if majors.contains(&mapped) {
            continue;
        }
        let (s0, e0) = (segments[i].start, segments[i].end);
        debris_secs += e0 - s0;
        let mut assigned = None;
        if e0 - s0 >= DEBRIS_MIN_SECS {
            if let Some(v) = embed_span(extractor, samples, sample_rate, s0, e0) {
                let bm = majors
                    .iter()
                    .map(|m| (cosine(&v, &centroids[m]), *m))
                    .fold((f32::MIN, -1), |acc, c| if c.0 > acc.0 { c } else { acc });
                if bm.0 >= DEBRIS_VOICE_FLOOR {
                    assigned = Some(bm.1);
                    by_voice += 1;
                }
            }
        }
        let label = assigned.unwrap_or_else(|| {
            by_time += 1;
            let mid = (s0 + e0) / 2.0;
            major_mids
                .iter()
                .min_by(|a, b| {
                    (a.0 - mid).abs().partial_cmp(&(b.0 - mid).abs()).expect("finite times")
                })
                .map(|&(_, lab)| lab)
                .expect("majors is non-empty")
        });
        segments[i].speaker = label;
    }
    if by_voice + by_time > 0 {
        eprintln!(
            "oversplit: folded {} debris segment(s) ({:.1}s) into real speakers ({} by voice, {} by adjacency)",
            by_voice + by_time,
            debris_secs,
            by_voice,
            by_time
        );
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

    #[arg(long, default_value_t = 1.1)]
    threshold: f32,

    #[arg(long = "min-duration-on", default_value_t = 0.3)]
    min_duration_on: f32,

    #[arg(long = "min-duration-off", default_value_t = 0.5)]
    min_duration_off: f32,

    /// Override the ONNX intra-op thread count (default: min(cores, 8))
    #[arg(long = "num-threads")]
    num_threads: Option<i32>,

    /// Merge clusters whose clean-speech centroids exceed this cosine
    /// similarity and fold tiny clusters into real speakers (<= 0 disables)
    #[arg(long = "merge-threshold", default_value_t = 0.8)]
    merge_threshold: f32,
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

/// Small ONNX ops in this pipeline synchronize heavily across the thread
/// pool, and efficiency cores amplify the contention: on an M1, 4 P-core
/// threads measured 2x faster than 8 mixed threads. Default to the
/// performance-core count on Apple Silicon.
fn default_num_threads() -> i32 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
        {
            if let Ok(p_cores) = String::from_utf8_lossy(&out.stdout).trim().parse::<i32>() {
                return p_cores.clamp(1, 8);
            }
        }
    }
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as i32;
    (logical / 2).clamp(1, 4)
}

fn run(args: Args) -> Result<Output> {
    let (samples, sample_rate) = read_wav_as_f32_mono(&args.audio)?;

    let num_threads = args.num_threads.unwrap_or_else(default_num_threads);
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

    let mut raw_segments = result.sort_by_start_time();

    if args.merge_threshold > 0.0 && !raw_segments.is_empty() {
        let extractor_config = sys::SpeakerEmbeddingExtractorConfig {
            model: emb_model.as_ptr(),
            num_threads,
            debug: 0,
            provider: cpu_provider.as_ptr(),
        };
        match EmbeddingExtractor::create(&extractor_config) {
            Some(extractor) => repair_oversplit(
                &mut raw_segments,
                &samples,
                sample_rate as i32,
                &extractor,
                args.merge_threshold,
            ),
            None => eprintln!("oversplit: failed to create embedding extractor; skipping repair"),
        }
    }

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
