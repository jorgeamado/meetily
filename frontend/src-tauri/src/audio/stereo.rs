//! Channel-identity support for stereo recordings.
//!
//! New recordings are saved as stereo with the local microphone on the left
//! channel and system audio (all remote parties) on the right. That makes the
//! left channel a ground-truth "the local user is speaking" signal: no
//! embedding clustering can confuse the local user with a remote voice.
//!
//! Retranscription uses this module to
//! 1. split the decoded stereo file into two 16 kHz mono streams,
//! 2. find intervals where the mic channel carries speech (RMS with a
//!    dominance check against the system channel to reject speaker bleed),
//! 3. overlay those intervals as a dedicated LOCAL_SPEAKER onto the diarized
//!    segments computed from the system channel alone.
//!
//! Whisper still transcribes the mono downmix; only speaker attribution
//! changes.

use super::decoder::DecodedAudio;
use super::diarization::{DiarizedSegment, LOCAL_SPEAKER};

/// metadata.json marker for stereo recordings with channel identity.
pub const MIC_SYSTEM_LAYOUT: &str = "mic-left-system-right";

/// Analysis frame length. 50ms matches the recording mixer's window, so the
/// mask has the same granularity the ducking decisions had at record time.
const FRAME_MS: usize = 50;
/// Speech gate for the mic channel — same RMS threshold the live mixer uses.
const MIC_RMS_THRESHOLD: f32 = 0.01;
/// The mic frame must also reach this fraction of the system frame's RMS.
/// A mic that only picks up loudspeaker bleed sits well below its source.
const DOMINANCE_RATIO: f32 = 0.6;
/// Gaps shorter than this between active frames are bridged (natural
/// intra-sentence pauses shouldn't fragment the local speaker's turns).
const MERGE_GAP_MS: f64 = 400.0;
/// Active islands shorter than this are dropped as noise (breath, key click).
const MIN_ON_MS: f64 = 200.0;

/// Read `channel_layout` from a meeting folder's metadata.json.
pub fn channel_layout(meeting_folder: &std::path::Path) -> Option<String> {
    let raw = std::fs::read_to_string(meeting_folder.join("metadata.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("channel_layout")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Split decoded stereo audio into (mic, system) 16 kHz mono streams.
/// Returns None for non-stereo input.
pub fn split_channels_16k(decoded: &DecodedAudio) -> Option<(Vec<f32>, Vec<f32>)> {
    if decoded.channels != 2 {
        return None;
    }
    let n = decoded.samples.len() / 2;
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for pair in decoded.samples.chunks_exact(2) {
        left.push(pair[0]);
        right.push(pair[1]);
    }
    let to_16k = |samples: Vec<f32>| {
        DecodedAudio {
            samples,
            sample_rate: decoded.sample_rate,
            channels: 1,
            duration_seconds: decoded.duration_seconds,
        }
        .to_whisper_format()
    };
    Some((to_16k(left), to_16k(right)))
}

/// Intervals (seconds) where the mic channel carries the local user's speech.
///
/// A frame is active when its RMS clears the speech gate AND dominates the
/// system channel's RMS in the same frame — loudspeaker bleed into the mic is
/// a strongly attenuated copy of the system signal, so it fails the ratio.
pub fn mic_activity_intervals(mic_16k: &[f32], sys_16k: &[f32], sample_rate: u32) -> Vec<(f64, f64)> {
    let frame_len = (sample_rate as usize * FRAME_MS / 1000).max(1);
    let frame_secs = frame_len as f64 / sample_rate as f64;

    let mut intervals: Vec<(f64, f64)> = Vec::new();
    let mut frame_idx = 0usize;
    for mic_frame in mic_16k.chunks(frame_len) {
        let start = frame_idx * frame_len;
        let sys_frame = &sys_16k[start.min(sys_16k.len())..(start + mic_frame.len()).min(sys_16k.len())];
        let mic_rms = rms(mic_frame);
        let sys_rms = rms(sys_frame);
        let active = mic_rms > MIC_RMS_THRESHOLD && mic_rms > DOMINANCE_RATIO * sys_rms;

        if active {
            let t0 = frame_idx as f64 * frame_secs;
            let t1 = t0 + mic_frame.len() as f64 / sample_rate as f64;
            match intervals.last_mut() {
                Some(last) if (t0 - last.1) * 1000.0 <= MERGE_GAP_MS => last.1 = t1,
                _ => intervals.push((t0, t1)),
            }
        }
        frame_idx += 1;
    }

    intervals
        .into_iter()
        .filter(|(s, e)| (e - s) * 1000.0 >= MIN_ON_MS)
        .collect()
}

/// Overlay mic-channel intervals onto system-channel diarization: mic-active
/// time belongs to LOCAL_SPEAKER; remote segments are clipped around it.
pub fn overlay_local_speaker(
    remote: &[DiarizedSegment],
    mic_intervals: &[(f64, f64)],
) -> Vec<DiarizedSegment> {
    const MIN_PIECE_SECS: f32 = 0.05;
    let mut out: Vec<DiarizedSegment> = Vec::new();

    for seg in remote {
        // Subtract every mic interval from this remote segment
        let mut pieces: Vec<(f32, f32)> = vec![(seg.start, seg.end)];
        for &(ms, me) in mic_intervals {
            let (ms, me) = (ms as f32, me as f32);
            let mut next: Vec<(f32, f32)> = Vec::with_capacity(pieces.len() + 1);
            for (ps, pe) in pieces {
                if me <= ps || ms >= pe {
                    next.push((ps, pe)); // no overlap
                } else {
                    if ms > ps {
                        next.push((ps, ms));
                    }
                    if me < pe {
                        next.push((me, pe));
                    }
                }
            }
            pieces = next;
        }
        for (ps, pe) in pieces {
            if pe - ps >= MIN_PIECE_SECS {
                out.push(DiarizedSegment { start: ps, end: pe, speaker: seg.speaker });
            }
        }
    }

    for &(ms, me) in mic_intervals {
        out.push(DiarizedSegment { start: ms as f32, end: me as f32, speaker: LOCAL_SPEAKER });
    }

    out.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_deinterleaves_channels() {
        let decoded = DecodedAudio {
            samples: vec![0.1, 0.5, 0.2, 0.6, 0.3, 0.7],
            sample_rate: 16000,
            channels: 2,
            duration_seconds: 3.0 / 16000.0,
        };
        let (mic, sys) = split_channels_16k(&decoded).unwrap();
        assert_eq!(mic, vec![0.1, 0.2, 0.3]);
        assert_eq!(sys, vec![0.5, 0.6, 0.7]);
    }

    #[test]
    fn split_rejects_mono() {
        let decoded = DecodedAudio {
            samples: vec![0.1, 0.2],
            sample_rate: 16000,
            channels: 1,
            duration_seconds: 2.0 / 16000.0,
        };
        assert!(split_channels_16k(&decoded).is_none());
    }

    fn tone(len: usize, amp: f32) -> Vec<f32> {
        (0..len).map(|i| if i % 2 == 0 { amp } else { -amp }).collect()
    }

    #[test]
    fn mic_speech_detected_and_silence_ignored() {
        let sr = 16000u32;
        // 1s loud mic speech, then 1s silence
        let mut mic = tone(sr as usize, 0.3);
        mic.extend(vec![0.0; sr as usize]);
        let sys = vec![0.0; 2 * sr as usize];
        let intervals = mic_activity_intervals(&mic, &sys, sr);
        assert_eq!(intervals.len(), 1);
        assert!(intervals[0].0 < 0.1);
        assert!((intervals[0].1 - 1.0).abs() < 0.1);
    }

    #[test]
    fn loudspeaker_bleed_rejected_by_dominance() {
        let sr = 16000u32;
        // System audio loud; mic only carries a 10% bleed copy — above the
        // absolute gate but far below dominance.
        let sys = tone(sr as usize, 0.5);
        let mic = tone(sr as usize, 0.05);
        let intervals = mic_activity_intervals(&mic, &sys, sr);
        assert!(intervals.is_empty());
    }

    #[test]
    fn short_gaps_merge_short_islands_drop() {
        let sr = 16000u32;
        let frame = sr as usize / 20; // 50ms
        let mut mic = Vec::new();
        // 400ms speech, 200ms gap, 400ms speech -> one merged interval
        mic.extend(tone(8 * frame, 0.3));
        mic.extend(vec![0.0; 4 * frame]);
        mic.extend(tone(8 * frame, 0.3));
        // long silence then a lone 50ms blip -> dropped
        mic.extend(vec![0.0; 20 * frame]);
        mic.extend(tone(frame, 0.3));
        mic.extend(vec![0.0; 4 * frame]);
        let sys = vec![0.0; mic.len()];
        let intervals = mic_activity_intervals(&mic, &sys, sr);
        assert_eq!(intervals.len(), 1);
        assert!((intervals[0].1 - intervals[0].0 - 1.0).abs() < 0.11); // ~1s merged
    }

    #[test]
    fn overlay_clips_remote_and_inserts_local() {
        let remote = vec![
            DiarizedSegment { start: 0.0, end: 10.0, speaker: 0 },
            DiarizedSegment { start: 12.0, end: 14.0, speaker: 1 },
        ];
        let mic = vec![(4.0, 6.0)];
        let out = overlay_local_speaker(&remote, &mic);
        // 0-4 spk0, 4-6 LOCAL, 6-10 spk0, 12-14 spk1
        assert_eq!(out.len(), 4);
        assert_eq!((out[0].start, out[0].end, out[0].speaker), (0.0, 4.0, 0));
        assert_eq!((out[1].start, out[1].end, out[1].speaker), (4.0, 6.0, LOCAL_SPEAKER));
        assert_eq!((out[2].start, out[2].end, out[2].speaker), (6.0, 10.0, 0));
        assert_eq!(out[3].speaker, 1);
    }

    #[test]
    fn overlay_drops_slivers() {
        let remote = vec![DiarizedSegment { start: 0.0, end: 5.0, speaker: 0 }];
        // Mic covers all but 20ms at the edges
        let mic = vec![(0.02, 4.99)];
        let out = overlay_local_speaker(&remote, &mic);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, LOCAL_SPEAKER);
    }
}
