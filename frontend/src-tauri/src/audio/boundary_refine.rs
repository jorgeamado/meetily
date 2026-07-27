// LLM tie-breaker for speaker-turn boundaries.
//
// Acoustic diarization is weakest exactly at fast handovers: the first word
// of the next speaker (often slightly overlapped) gets attributed to the
// previous one ("What do you call them? Are" / "they moving along the
// street?"). Timestamps cannot fix that — the diarizer's speaker map itself
// is wrong for that word. A language model, however, sees immediately that
// the cut belongs after "them?".
//
// This module asks the bundled summary LLM one tiny constrained question per
// suspicious boundary: a numbered list of candidate cut positions (word
// starts near the current cut), answered with `{"cut": N}` in a handful of
// tokens. Free-form rewriting is never requested, so the transcript text can
// only be re-partitioned, never altered.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use log::{info, warn};
use tauri::{AppHandle, Manager, Runtime};

use super::diarization::SpeakerTurn;
use crate::summary::summary_engine::{self, models};

/// Only boundaries with a gap this small between the last word of one turn
/// and the first word of the next are questioned — a handover with over a
/// second of clean silence is reliably attributed by the diarizer. (The
/// misattributed "Are" in the reference meeting sat ~900ms before the next
/// word's start.)
const TIGHT_GAP_MS: f64 = 1200.0;

/// How many words on each side of the current cut may move.
const MAX_SHIFT_WORDS: usize = 2;

/// Words of context shown on each side of a candidate cut.
const CONTEXT_WORDS: usize = 12;

/// Hard caps so refinement never dominates a retranscription run.
const MAX_QUERIES: usize = 24;
const TOTAL_BUDGET: Duration = Duration::from_secs(240);
const PER_QUERY_TIMEOUT: Duration = Duration::from_secs(45);
const MAX_CONSECUTIVE_FAILURES: usize = 2;

const SYSTEM_PROMPT: &str = "You correct speaker-change points in conversation transcripts. \
Answer with JSON only, no explanation.";

#[derive(Debug, Default, Clone, Copy)]
pub struct RefineStats {
    pub boundaries: usize,
    pub queried: usize,
    pub moved: usize,
    pub failures: usize,
}

/// Group a turn's tokens into words: a word starts at token 0 or at a token
/// with leading whitespace. Returns [start, end) token ranges.
fn word_ranges(turn: &SpeakerTurn) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (i, w) in turn.words.iter().enumerate() {
        if i == 0 || w.text.starts_with(char::is_whitespace) {
            ranges.push((i, i + 1));
        } else if let Some(last) = ranges.last_mut() {
            last.1 = i + 1;
        }
    }
    ranges
}

fn words_text(turn: &SpeakerTurn, ranges: &[(usize, usize)]) -> Vec<String> {
    ranges
        .iter()
        .map(|&(a, b)| {
            turn.words[a..b]
                .iter()
                .map(|w| w.text.as_str())
                .collect::<String>()
                .trim()
                .to_string()
        })
        .collect()
}

/// A boundary between turns[i] and turns[i+1] worth asking the LLM about.
#[derive(Debug)]
struct Boundary {
    left_idx: usize,
    gap_ms: f64,
}

fn is_tight_boundary(left: &SpeakerTurn, right: &SpeakerTurn) -> Option<f64> {
    let (l_spk, r_spk) = (left.speaker?, right.speaker?);
    if l_spk == r_spk {
        return None;
    }
    let l_last = left.words.last()?;
    let r_first = right.words.first()?;
    let gap = r_first.start_ms - l_last.end_ms;
    (gap < TIGHT_GAP_MS).then_some(gap.max(0.0))
}

fn find_boundaries(turns: &[SpeakerTurn]) -> Vec<Boundary> {
    let mut out: Vec<Boundary> = turns
        .windows(2)
        .enumerate()
        .filter_map(|(i, pair)| {
            is_tight_boundary(&pair[0], &pair[1]).map(|gap_ms| Boundary { left_idx: i, gap_ms })
        })
        .collect();
    if out.len() > MAX_QUERIES {
        // Keep the tightest handovers — they are the likeliest to be wrong
        out.sort_by(|a, b| a.gap_ms.partial_cmp(&b.gap_ms).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(MAX_QUERIES);
        out.sort_by_key(|b| b.left_idx);
    }
    out
}

/// Candidate cut shifts for one boundary, in whole words. Negative = move
/// that many of the left turn's final words to the right turn; positive =
/// move that many of the right turn's leading words to the left. Each side
/// must keep at least one word.
fn candidate_shifts(left_words: usize, right_words: usize) -> Vec<i32> {
    let take_left = MAX_SHIFT_WORDS.min(left_words.saturating_sub(1)) as i32;
    let take_right = MAX_SHIFT_WORDS.min(right_words.saturating_sub(1)) as i32;
    (-take_left..=take_right).collect()
}

/// Render the numbered options. Returns (prompt, shifts) — shifts[n-1] is the
/// shift encoded by option n.
fn build_prompt(left: &SpeakerTurn, right: &SpeakerTurn) -> Option<(String, Vec<i32>)> {
    let l_ranges = word_ranges(left);
    let r_ranges = word_ranges(right);
    let shifts = candidate_shifts(l_ranges.len(), r_ranges.len());
    if shifts.len() < 2 {
        return None;
    }
    let l_words = words_text(left, &l_ranges);
    let r_words = words_text(right, &r_ranges);

    let mut prompt = String::from(
        "A transcript was split between two speakers A and B, but the split point may be off by a word or two. Pick the option where each speaker's words read as natural, complete utterances.\n\n",
    );
    for (n, &shift) in shifts.iter().enumerate() {
        // Words belonging to A for this option: all of A's words plus the
        // first `shift` of B's (or minus the last `-shift` of A's own)
        let a_end = (l_words.len() as i32 + shift.min(0)) as usize;
        let b_start = shift.max(0) as usize;

        let a_tail_start = a_end.saturating_sub(CONTEXT_WORDS);
        let mut a_text = String::new();
        if a_tail_start > 0 {
            a_text.push_str("... ");
        }
        a_text.push_str(&l_words[a_tail_start..a_end].join(" "));
        if shift > 0 {
            a_text.push(' ');
            a_text.push_str(&r_words[..b_start].join(" "));
        }

        let mut b_text = String::new();
        if shift < 0 {
            b_text.push_str(&l_words[a_end..].join(" "));
            b_text.push(' ');
        }
        let b_head_end = CONTEXT_WORDS.min(r_words.len());
        b_text.push_str(&r_words[b_start..b_head_end.max(b_start)].join(" "));
        if b_head_end < r_words.len() {
            b_text.push_str(" ...");
        }

        prompt.push_str(&format!(
            "Option {}:\nA: \"{}\"\nB: \"{}\"\n\n",
            n + 1,
            a_text.trim(),
            b_text.trim()
        ));
    }
    prompt.push_str(&format!(
        "Reply with only JSON like {{\"cut\": 1}} choosing the best option (1-{}).",
        shifts.len()
    ));
    Some((prompt, shifts))
}

/// Extract the chosen option from the model's reply. Accepts any text that
/// contains "cut" followed by an integer.
fn parse_cut(reply: &str) -> Option<usize> {
    let idx = reply.find("cut")?;
    let digits: String = reply[idx + 3..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Move whole words across the boundary according to `shift` (see
/// [`candidate_shifts`]) and rebuild both turns' text and time bounds.
fn apply_shift(left: &mut SpeakerTurn, right: &mut SpeakerTurn, shift: i32) {
    if shift == 0 {
        return;
    }
    if shift < 0 {
        let l_ranges = word_ranges(left);
        let keep_words = l_ranges.len() - (-shift) as usize;
        let cut_token = l_ranges[keep_words].0;
        let moved: Vec<_> = left.words.drain(cut_token..).collect();
        // Moved tokens become the head of the right turn; ensure the join
        // point keeps a separating space (token already carries one)
        let mut new_words = moved;
        new_words.extend(right.words.drain(..));
        right.words = new_words;
    } else {
        let r_ranges = word_ranges(right);
        let cut_token = r_ranges[shift as usize].0;
        let moved: Vec<_> = right.words.drain(..cut_token).collect();
        left.words.extend(moved);
    }
    left.rebuild_from_words();
    right.rebuild_from_words();
}

/// Pick the best available local model for micro-queries: prefer the smaller
/// Qwen if downloaded, fall back to any downloaded built-in model.
fn pick_model(app_data_dir: &PathBuf) -> Option<models::ModelDef> {
    let preferred = ["qwen3.5:2b", "qwen3.5:4b"];
    let available = models::get_available_models();
    let downloaded = |m: &models::ModelDef| {
        models::get_model_path(app_data_dir, &m.name)
            .map(|p| p.exists())
            .unwrap_or(false)
    };
    for name in preferred {
        if let Some(m) = available.iter().find(|m| m.name == name) {
            if downloaded(m) {
                return Some(m.clone());
            }
        }
    }
    available.into_iter().find(|m| downloaded(m))
}

/// Refine speaker-turn boundaries in place. Never fails the retranscription:
/// any error just leaves the affected boundary as the acoustic pass set it.
pub async fn refine_turns<R: Runtime>(
    app: &AppHandle<R>,
    turns: &mut [SpeakerTurn],
    cancelled: impl Fn() -> bool,
    mut on_progress: impl FnMut(usize, usize),
) -> RefineStats {
    let mut stats = RefineStats::default();

    let boundaries = find_boundaries(turns);
    stats.boundaries = boundaries.len();
    if boundaries.is_empty() {
        return stats;
    }

    let Ok(app_data_dir) = app.path().app_data_dir() else {
        warn!("Boundary refine: no app data dir, skipping");
        return stats;
    };
    let Some(model) = pick_model(&app_data_dir) else {
        info!("Boundary refine: no local LLM downloaded, skipping {} boundaries", boundaries.len());
        return stats;
    };
    info!(
        "Boundary refine: {} tight boundaries, using {}",
        boundaries.len(),
        model.name
    );

    let started = Instant::now();
    let mut consecutive_failures = 0usize;
    let total = boundaries.len();

    for (done, boundary) in boundaries.iter().enumerate() {
        if cancelled() {
            info!("Boundary refine: cancelled after {} queries", stats.queried);
            break;
        }
        if started.elapsed() > TOTAL_BUDGET {
            warn!(
                "Boundary refine: budget exhausted after {} of {} boundaries",
                done, total
            );
            break;
        }
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            warn!("Boundary refine: {} consecutive LLM failures, stopping", consecutive_failures);
            break;
        }
        on_progress(done, total);

        let (left_slice, right_slice) = turns.split_at_mut(boundary.left_idx + 1);
        let left = &mut left_slice[boundary.left_idx];
        let right = &mut right_slice[0];

        let Some((prompt, shifts)) = build_prompt(left, right) else {
            continue;
        };
        let keep_option = shifts.iter().position(|&s| s == 0).map(|p| p + 1);

        stats.queried += 1;
        match summary_engine::generate_micro(
            &app_data_dir,
            &model.name,
            SYSTEM_PROMPT,
            &prompt,
            16,
            PER_QUERY_TIMEOUT,
        )
        .await
        {
            Ok(reply) => {
                consecutive_failures = 0;
                match parse_cut(&reply) {
                    Some(n) if (1..=shifts.len()).contains(&n) => {
                        let shift = shifts[n - 1];
                        if shift != 0 {
                            info!(
                                "Boundary refine: moving {} word(s) {} at {:.1}s ({:?} -> option {})",
                                shift.abs(),
                                if shift < 0 { "right" } else { "left" },
                                right.start_ms / 1000.0,
                                keep_option,
                                n
                            );
                            apply_shift(left, right, shift);
                            stats.moved += 1;
                        }
                    }
                    other => {
                        warn!("Boundary refine: unparseable reply {:?} ({:?})", reply, other);
                        stats.failures += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Boundary refine: LLM query failed: {}", e);
                stats.failures += 1;
                consecutive_failures += 1;
            }
        }
    }
    on_progress(total, total);
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::diarization::WordSpan;

    fn tok(text: &str, start_ms: f64, end_ms: f64) -> WordSpan {
        WordSpan { text: text.to_string(), start_ms, end_ms }
    }

    fn turn(words: Vec<WordSpan>, speaker: usize) -> SpeakerTurn {
        let mut t = SpeakerTurn {
            text: String::new(),
            start_ms: 0.0,
            end_ms: 0.0,
            speaker: Some(speaker),
            words,
        };
        t.rebuild_from_words();
        t
    }

    // The real case from the user's 3-person meeting: "What do you call
    // them? Are" | "they moving along the street?"
    fn are_boundary() -> (SpeakerTurn, SpeakerTurn) {
        let left = turn(
            vec![
                tok("What", 26800.0, 27000.0),
                tok(" do", 27000.0, 27200.0),
                tok(" you", 27200.0, 27400.0),
                tok(" call", 27400.0, 27700.0),
                tok(" them?", 27700.0, 28600.0),
                tok(" Are", 28900.0, 29200.0),
            ],
            1,
        );
        let right = turn(
            vec![
                tok(" they", 30100.0, 30300.0),
                tok(" moving", 30300.0, 30700.0),
                tok(" along", 30700.0, 31000.0),
                tok(" the", 31000.0, 31100.0),
                tok(" street?", 31100.0, 31800.0),
            ],
            0,
        );
        (left, right)
    }

    #[test]
    fn word_ranges_group_multi_token_words() {
        let t = turn(
            vec![tok("hel", 0.0, 100.0), tok("lo", 100.0, 200.0), tok(" world", 200.0, 400.0)],
            0,
        );
        assert_eq!(word_ranges(&t), vec![(0, 2), (2, 3)]);
    }

    #[test]
    fn tight_boundary_detected_and_loose_ignored() {
        let (left, right) = are_boundary();
        // 900ms gap between "Are" (ends 29.2s) and "they" (starts 30.1s)
        assert!(is_tight_boundary(&left, &right).is_some());

        let mut far_right = right.clone();
        for w in far_right.words.iter_mut() {
            w.start_ms += 2000.0;
            w.end_ms += 2000.0;
        }
        far_right.rebuild_from_words();
        assert!(is_tight_boundary(&left, &far_right).is_none());
    }

    #[test]
    fn same_speaker_boundary_not_queried() {
        let (left, mut right) = are_boundary();
        right.speaker = left.speaker;
        assert!(is_tight_boundary(&left, &right).is_none());
    }

    #[test]
    fn candidate_shifts_respect_one_word_minimum() {
        assert_eq!(candidate_shifts(6, 5), vec![-2, -1, 0, 1, 2]);
        assert_eq!(candidate_shifts(1, 5), vec![0, 1, 2]);
        assert_eq!(candidate_shifts(2, 1), vec![-1, 0]);
        assert_eq!(candidate_shifts(1, 1), vec![0]);
    }

    #[test]
    fn prompt_lists_options_and_shift_map_matches() {
        let (left, right) = are_boundary();
        let (prompt, shifts) = build_prompt(&left, &right).expect("prompt");
        assert_eq!(shifts, vec![-2, -1, 0, 1, 2]);
        // Option 2 (shift -1) moves "Are" to B — the correct reading
        assert!(prompt.contains("Option 2:\nA: \"What do you call them?\"\nB: \"Are they moving along the street?\""), "prompt:\n{}", prompt);
        // Option 3 (shift 0) keeps the current split
        assert!(prompt.contains("Option 3:\nA: \"What do you call them? Are\"\nB: \"they moving along the street?\""));
    }

    #[test]
    fn parse_cut_handles_json_and_noise() {
        assert_eq!(parse_cut("{\"cut\": 2}"), Some(2));
        assert_eq!(parse_cut(" {\"cut\":10} "), Some(10));
        assert_eq!(parse_cut("The cut is 3."), Some(3));
        assert_eq!(parse_cut("no answer"), None);
        assert_eq!(parse_cut("{\"cut\": }"), None);
    }

    #[test]
    fn apply_negative_shift_moves_trailing_word_right() {
        let (mut left, mut right) = are_boundary();
        apply_shift(&mut left, &mut right, -1);
        assert_eq!(left.text.trim(), "What do you call them?");
        assert_eq!(right.text.trim(), "Are they moving along the street?");
        assert!((left.end_ms - 28600.0).abs() < 1e-6);
        assert!((right.start_ms - 28900.0).abs() < 1e-6);
    }

    #[test]
    fn apply_positive_shift_moves_leading_word_left() {
        let (mut left, mut right) = are_boundary();
        apply_shift(&mut left, &mut right, 1);
        assert_eq!(left.text.trim(), "What do you call them? Are they");
        assert_eq!(right.text.trim(), "moving along the street?");
        assert!((left.end_ms - 30300.0).abs() < 1e-6);
        assert!((right.start_ms - 30300.0).abs() < 1e-6);
    }

    #[test]
    fn apply_shift_never_splits_multi_token_words() {
        let mut left = turn(
            vec![tok("okay", 0.0, 300.0), tok(" hel", 300.0, 400.0), tok("lo", 400.0, 600.0)],
            0,
        );
        let mut right = turn(vec![tok(" next", 700.0, 900.0)], 1);
        apply_shift(&mut left, &mut right, -1);
        assert_eq!(left.text.trim(), "okay");
        assert_eq!(right.text.trim(), "hello next");
    }
}
