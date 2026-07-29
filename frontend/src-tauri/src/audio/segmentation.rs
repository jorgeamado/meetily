//! Frame-level voice-change evidence from the pyannote segmentation model.
//!
//! The diarization sidecar runs pyannote segmentation-3.0 internally but
//! only returns clustered segments — the model's frame-level output is
//! discarded. That output is a powerset classification (silence, three
//! single-speaker classes, three two-speaker overlap classes) every ~17ms,
//! and within one 10s window its speaker labels are locally consistent.
//!
//! Measured on the labeled boundary cases (2026-07-29): at a misattributed
//! handover the frame labels showed the disputed words as the SAME local
//! voice as the previous turn, with the true change-point at the next
//! pause — exactly the ground truth the embedding clustering destroyed.
//! Overlap classes carried no signal at these boundaries (≤0.04 mass);
//! the change-points and same/different-voice runs carried all of it.
//!
//! The evidence is used asymmetrically: it may reposition a cut onto a
//! pause between two different sustained voices, or confirm a cut already
//! there — it must never delete a boundary, because the same measurement
//! showed the model sometimes gives two real speakers one local label
//! (two of twelve windows).
//!
//! Runs in-process through `ort` (already linked for Parakeet) on the
//! same segmentation-3.0.onnx the diarization sidecar downloads.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

pub const SAMPLE_RATE: usize = 16_000;
/// The model consumes fixed 10s windows.
pub const WINDOW_SECS: f64 = 10.0;
const WINDOW_SAMPLES: usize = 160_000;

/// A local voice run must last this long to count as a real turn. A measured
/// spurious speaker split sat on a 0.3s blip; real turns in the same windows
/// ran 0.7s and longer.
const MIN_SUSTAINED_SECS: f64 = 0.5;
/// The pause between two different sustained voices must be at least this
/// long for a cut to be attributed to it (the tightest measured true
/// handover pause was 0.25s).
const MIN_GAP_SECS: f64 = 0.2;
/// A candidate instant must sit this far inside the pause to count as "on"
/// the voice change — word timestamps carry some slop.
const GAP_MARGIN_SECS: f64 = 0.05;

/// One run of consecutive frames sharing an argmax class, in absolute
/// seconds on the recording timeline. Classes: 0 silence, 1-3 single
/// speakers (labels only meaningful within one window), 4-6 overlaps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceRun {
    pub start: f64,
    pub end: f64,
    pub class: usize,
}

pub struct SegmentationModel {
    session: Session,
}

impl SegmentationModel {
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load segmentation model {}", model_path.display()))?;
        Ok(Self { session })
    }

    /// Run one 10s window starting at `window_start_s` (zero-padded past the
    /// end of `audio_16k`) and return the collapsed argmax runs.
    pub fn voice_runs(&mut self, audio_16k: &[f32], window_start_s: f64) -> Result<Vec<VoiceRun>> {
        let start = (window_start_s * SAMPLE_RATE as f64) as usize;
        let mut chunk: Vec<f32> = audio_16k
            .get(start.min(audio_16k.len())..)
            .unwrap_or(&[])
            .iter()
            .take(WINDOW_SAMPLES)
            .copied()
            .collect();
        chunk.resize(WINDOW_SAMPLES, 0.0);

        let input = Array3::from_shape_vec((1, 1, WINDOW_SAMPLES), chunk)?;
        let outputs = self
            .session
            .run(ort::inputs!["x" => TensorRef::from_array_view(input.view())?])?;
        let logits = outputs
            .get("y")
            .context("segmentation model has no output named 'y'")?
            .try_extract_array::<f32>()?;

        let shape = logits.shape();
        let (frames, classes) = (shape[1], shape[2]);
        let frame_secs = WINDOW_SECS / frames as f64;
        let mut runs: Vec<VoiceRun> = Vec::new();
        for i in 0..frames {
            let class = (0..classes)
                .max_by(|&a, &b| {
                    logits[[0, i, a]]
                        .partial_cmp(&logits[[0, i, b]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let t0 = window_start_s + i as f64 * frame_secs;
            let t1 = t0 + frame_secs;
            match runs.last_mut() {
                Some(last) if last.class == class => last.end = t1,
                _ => runs.push(VoiceRun { start: t0, end: t1, class }),
            }
        }
        Ok(runs)
    }
}

/// Speech runs (any non-silence class) long enough to be real turns.
pub fn sustained_runs(runs: &[VoiceRun]) -> Vec<VoiceRun> {
    runs.iter()
        .filter(|r| r.class != 0 && r.end - r.start >= MIN_SUSTAINED_SECS)
        .copied()
        .collect()
}

/// Speech runs that may bound a change gap: sustained runs, plus short runs
/// whose class is sustained elsewhere in the window. A short interjection
/// ("Yeah.") of an established voice still marks where that voice took
/// over; an isolated blip of a class heard nowhere else is noise.
fn bounding_runs(runs: &[VoiceRun]) -> Vec<VoiceRun> {
    let sustained_classes: Vec<usize> =
        sustained_runs(runs).iter().map(|r| r.class).collect();
    runs.iter()
        .filter(|r| {
            r.class != 0
                && (r.end - r.start >= MIN_SUSTAINED_SECS
                    || sustained_classes.contains(&r.class))
        })
        .copied()
        .collect()
}

/// The pause between two consecutive voices of DIFFERENT single-speaker
/// classes nearest to `t`. Same-voice pauses and pauses shorter than
/// MIN_GAP_SECS yield nothing — silence alone is not evidence of a speaker
/// change. Overlap classes (4-6) never bound a gap: who owns the words
/// around simultaneous speech is exactly what a pause cannot attribute
/// (observed live: an overlap-bounded gap produced the only questionable
/// cut move in two real-meeting dry-runs).
pub fn change_gap_near(runs: &[VoiceRun], t: f64) -> Option<(f64, f64)> {
    // The gap must be a real pause: dropped blips and overlap runs still
    // count as speech when they sit INSIDE the interval (observed live: an
    // overlap run dropped from bounding left its interval looking like a
    // pause, and a cut was confirmed in the middle of simultaneous speech).
    let speech_inside = |a: f64, b: f64| -> f64 {
        runs.iter()
            .filter(|r| r.class != 0)
            .map(|r| (r.end.min(b) - r.start.max(a)).max(0.0))
            .sum()
    };
    let speech = bounding_runs(runs);
    speech
        .windows(2)
        .filter(|pair| {
            (1..=3).contains(&pair[0].class)
                && (1..=3).contains(&pair[1].class)
                && pair[0].class != pair[1].class
        })
        .map(|pair| (pair[0].end, pair[1].start))
        .filter(|&(a, b)| b - a >= MIN_GAP_SECS && speech_inside(a, b) < 0.05)
        .min_by(|x, y| {
            let dx = ((x.0 + x.1) / 2.0 - t).abs();
            let dy = ((y.0 + y.1) / 2.0 - t).abs();
            dx.partial_cmp(&dy).unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Whether an instant falls firmly inside the pause.
pub fn instant_in_gap(t: f64, gap: (f64, f64)) -> bool {
    t >= gap.0 + GAP_MARGIN_SECS && t <= gap.1 - GAP_MARGIN_SECS
}

/// Word timestamps can be off by this much (whisper pads a turn's last word
/// toward the next speech): a current cut this close to the pause may
/// already be correct, so repositioning would claim precision the
/// timestamps don't have.
const CURRENT_CUT_SLOP_SECS: f64 = 0.25;

/// Of all candidate cut instants, the single one falling inside the voice
/// change — None when none or several do (several = the pause is too wide
/// to attribute the words between them by acoustics alone). A move (a
/// non-zero winning shift) additionally requires the current cut (shift 0)
/// to sit clearly away from the pause; observed live: an inflated last-word
/// end pushed a correct cut 0.04s past the gap edge while the mid-idiom
/// candidate before it landed inside.
pub fn unique_candidate_in_gap(instants: &[(i32, f64)], gap: (f64, f64)) -> Option<i32> {
    let mut hits = instants.iter().filter(|&&(_, t)| instant_in_gap(t, gap));
    let shift = match (hits.next(), hits.next()) {
        (Some(&(shift, _)), None) => shift,
        _ => return None,
    };
    if shift != 0 {
        let current = instants.iter().find(|&&(s, _)| s == 0)?.1;
        let distance = (gap.0 - current).max(current - gap.1).max(0.0);
        if distance < CURRENT_CUT_SLOP_SECS {
            return None;
        }
    }
    Some(shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(start: f64, end: f64, class: usize) -> VoiceRun {
        VoiceRun { start, end, class }
    }

    #[test]
    fn sustained_drops_blips_and_silence() {
        let runs = vec![
            run(0.0, 4.7, 1),
            run(4.7, 5.8, 0),
            run(5.8, 6.1, 2), // 0.3s blip — the measured false-split signature
            run(6.1, 6.5, 1), // 0.4s — still below threshold
            run(6.5, 7.8, 0),
            run(7.8, 9.8, 2),
        ];
        let s = sustained_runs(&runs);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].class, s[1].class), (1, 2));
    }

    #[test]
    fn change_gap_found_between_different_voices_only() {
        // voice 2 ... pause ... voice 2 (same) ... pause ... voice 1
        let s = vec![run(0.0, 4.4, 2), run(5.2, 5.9, 2), run(6.2, 9.0, 1)];
        // The 4.4-5.2 pause separates SAME voice — not a change gap
        let gap = change_gap_near(&s, 4.8).expect("gap");
        assert_eq!(gap, (5.9, 6.2));
    }

    #[test]
    fn short_interjection_of_established_voice_bounds_the_gap() {
        // The measured "environment. | Yeah. | I thought..." shape: voice 2
        // ends, a 0.4s "Yeah." blip of voice 3 follows, then sustained
        // voice 3. The change gap is the FIRST pause; the pause after the
        // blip is same-voice and must not qualify.
        let runs = vec![
            run(156.1, 158.8, 2),
            run(158.8, 160.5, 0),
            run(160.5, 160.9, 3),
            run(160.9, 162.0, 0),
            run(162.0, 165.2, 3),
        ];
        assert_eq!(change_gap_near(&runs, 160.2), Some((158.8, 160.5)));
    }

    #[test]
    fn overlap_speech_inside_the_interval_disqualifies_the_gap() {
        // The live regression: voice 2, a 0.4s overlap run (dropped from
        // bounding), voice 3 — the interval between the bounding runs is
        // full of simultaneous speech, not a pause.
        let runs = vec![
            run(1640.8, 1644.16, 2),
            run(1644.16, 1644.55, 6),
            run(1644.55, 1647.65, 3),
        ];
        assert_eq!(change_gap_near(&runs, 1644.25), None);
    }

    #[test]
    fn isolated_noise_blip_does_not_fake_a_change() {
        // A 0.3s blip of a class heard nowhere else sits between two runs
        // of one voice: without it there is no change, and it must not
        // manufacture one.
        let runs = vec![
            run(0.0, 4.0, 1),
            run(4.0, 4.6, 0),
            run(4.6, 4.9, 2),
            run(4.9, 5.5, 0),
            run(5.5, 9.0, 1),
        ];
        assert_eq!(change_gap_near(&runs, 4.7), None);
    }

    #[test]
    fn abutting_change_yields_no_gap() {
        let s = vec![run(0.0, 5.0, 1), run(5.1, 9.0, 2)];
        assert!(change_gap_near(&s, 5.0).is_none());
    }

    #[test]
    fn nearest_gap_wins() {
        let s = vec![run(0.0, 2.0, 1), run(2.5, 4.0, 2), run(4.4, 6.0, 1)];
        assert_eq!(change_gap_near(&s, 2.2), Some((2.0, 2.5)));
        assert_eq!(change_gap_near(&s, 4.5), Some((4.0, 4.4)));
    }

    #[test]
    fn unique_hit_decides_ties_do_not() {
        let gap = (95.93, 96.18);
        // dbea case 8: current cut mid-speech, the +N "after George." cut on the pause
        assert_eq!(
            unique_candidate_in_gap(&[(0, 94.8), (3, 96.05)], gap),
            Some(3)
        );
        // both candidates inside one wide pause -> not decidable
        assert_eq!(
            unique_candidate_in_gap(&[(0, 96.0), (1, 96.1)], gap),
            None
        );
        // margin: an instant right at the gap edge does not count
        assert_eq!(unique_candidate_in_gap(&[(0, 95.94)], gap), None);
    }

    #[test]
    fn near_gap_current_cut_suppresses_moves() {
        // The live near-miss: pause at 1766.09-1767.40, current cut 0.04s
        // past its edge (inflated word end), the mid-idiom -1 candidate
        // inside — must NOT move.
        let gap = (1766.09, 1767.40);
        assert_eq!(
            unique_candidate_in_gap(&[(-1, 1766.15), (0, 1767.44)], gap),
            None
        );
        // A current cut clearly away from the pause still allows the move
        assert_eq!(
            unique_candidate_in_gap(&[(-1, 1766.15), (0, 1768.0)], gap),
            Some(-1)
        );
    }

    #[test]
    fn real_model_runs_a_window() {
        let model_path = dirs::data_dir()
            .unwrap_or_default()
            .join("com.meetily.ai/models/diarization/segmentation-3.0.onnx");
        if !model_path.exists() {
            eprintln!("segmentation model not downloaded; skipping");
            return;
        }
        let mut model = SegmentationModel::load(&model_path).expect("load");
        let audio = vec![0.0f32; WINDOW_SAMPLES];
        let runs = model.voice_runs(&audio, 0.0).expect("run");
        assert!(!runs.is_empty());
        // Silence in, silence out
        assert!(runs.iter().all(|r| r.class == 0));
    }
}
