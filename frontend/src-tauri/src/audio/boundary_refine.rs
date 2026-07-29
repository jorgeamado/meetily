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
use super::segmentation::{self, SegmentationModel};
use crate::summary::summary_engine::{self, models};

/// Only boundaries with a gap this small between the last word of one turn
/// and the first word of the next are questioned — a handover with over a
/// second of clean silence is reliably attributed by the diarizer. (The
/// misattributed "Are" in the reference meeting sat ~900ms before the next
/// word's start.)
const TIGHT_GAP_MS: f64 = 1200.0;

/// How many words on each side of the current cut may move.
const MAX_SHIFT_WORDS: usize = 2;

/// Sentence-punctuation cut candidates are offered up to this many words
/// from the current cut. A boundary error like "You think your hangovers
/// are | worse now than when you were younger?" is 7 words off — no ±2
/// option ever looks right, but "cut after 'younger?'" is instantly
/// recognizable. (Widened from 8: a measured real case needed +10.)
const PUNCT_SHIFT_WORDS: usize = 12;

/// A cut is only "trusted" (local jitter only) when the left side ends a
/// sentence of at least this many words. Fragment endings like "Sorry?"
/// are exactly where the diarizer glues the next speaker's words on —
/// measured case: the true cut sat +4 past a trusted "Sorry?".
const TRUSTED_SENTENCE_MIN_WORDS: usize = 3;

/// Cap on options shown per query.
const MAX_OPTIONS: usize = 7;

/// Words of context shown on each side of a candidate cut.
const CONTEXT_WORDS: usize = 12;

/// A short turn attributed to speaker B while both neighbors belong to
/// speaker A is usually diarization noise cutting through the middle of A's
/// sentence ("you should have | proper people | around you"). Turns up to
/// this many words qualify for a merge query. Kept small on purpose: a
/// 5-word turn like "Yeah, you're right, but you" is often a REAL response
/// that merely reads like the surrounding speaker's discourse filler, and a
/// text-only model cannot tell those apart.
const SANDWICH_MAX_WORDS: usize = 3;

/// When the LLM picks the outermost cut option, the true boundary may lie
/// further still — re-ask with the window recentered, up to this many rounds
/// per boundary (reach: MAX_SHIFT_WORDS * MAX_ROUNDS words).
const MAX_ROUNDS_PER_BOUNDARY: usize = 3;

/// Hard caps so refinement never dominates a retranscription run.
const MAX_QUERIES: usize = 32;
const TOTAL_BUDGET: Duration = Duration::from_secs(240);
const PER_QUERY_TIMEOUT: Duration = Duration::from_secs(45);
const MAX_CONSECUTIVE_FAILURES: usize = 2;

const SYSTEM_PROMPT: &str = "You correct speaker-change points in conversation transcripts. \
Answer with JSON only, no explanation.";

#[derive(Debug, Default, Clone, Copy)]
pub struct RefineStats {
    pub boundaries: usize,
    pub sandwiches: usize,
    pub queried: usize,
    pub moved: usize,
    pub merged: usize,
    pub failures: usize,
    /// Cuts decided by frame-level voice-change evidence (no LLM query):
    /// confirmed in place / moved onto the voice change.
    pub acoustic_confirmed: usize,
    pub acoustic_moved: usize,
}

/// Group a turn's tokens into words: a word starts at token 0 or at a token
/// with leading whitespace. Returns [start, end) token ranges.
pub fn word_ranges(turn: &SpeakerTurn) -> Vec<(usize, usize)> {
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

pub fn words_text(turn: &SpeakerTurn, ranges: &[(usize, usize)]) -> Vec<String> {
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

/// True when a word plausibly ends a sentence — a natural cut point.
pub fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', ')', ']', '»'])
        .ends_with(['.', '?', '!', '…'])
}

/// Candidate cut shifts for one boundary, in whole words. Negative = move
/// that many of the left turn's final words to the right turn; positive =
/// move that many of the right turn's leading words to the left. Each side
/// must keep at least one word.
///
/// The base set is every shift within ±MAX_SHIFT_WORDS; further out (to
/// ±PUNCT_SHIFT_WORDS) only cuts falling right after sentence punctuation
/// are offered, so long-range errors stay reachable without exploding the
/// option list.
pub fn candidate_shifts(l_words: &[String], r_words: &[String]) -> Vec<i32> {
    // When the current cut already falls after a COMPLETE sentence,
    // acoustics and syntax agree — only small jitter is plausible. Offering
    // far candidates there makes the model grab the neighbor's whole
    // sentence ("...these platforms. | What do you call them?" reads fine
    // as one voice; only the audio knows it is not). Fragment endings
    // ("Sorry?") do NOT earn that trust: they are where glue errors live.
    let cut_at_sentence_end = l_words.last().map(|w| ends_sentence(w)).unwrap_or(false);
    let trailing_sentence_words = l_words
        .iter()
        .rev()
        .skip(1)
        .take_while(|w| !ends_sentence(w))
        .count()
        + 1;
    let reach = if cut_at_sentence_end && trailing_sentence_words >= TRUSTED_SENTENCE_MIN_WORDS {
        MAX_SHIFT_WORDS
    } else {
        PUNCT_SHIFT_WORDS
    };
    let take_left = reach.min(l_words.len().saturating_sub(1)) as i32;
    let take_right = reach.min(r_words.len().saturating_sub(1)) as i32;
    // Last word on the A side if this shift were applied
    let a_side_last = |s: i32| -> &str {
        if s > 0 {
            &r_words[s as usize - 1]
        } else {
            &l_words[(l_words.len() as i32 + s.min(0)) as usize - 1]
        }
    };
    let mut shifts: Vec<i32> = (-take_left..=take_right)
        .filter(|&s| s.unsigned_abs() as usize <= MAX_SHIFT_WORDS || ends_sentence(a_side_last(s)))
        .collect();
    // Too many options confuse a small model — keep 0, then the nearest
    while shifts.len() > MAX_OPTIONS {
        let farthest = shifts
            .iter()
            .copied()
            .max_by_key(|&s| (s.unsigned_abs(), s != 0))
            .unwrap();
        shifts.retain(|&s| s != farthest);
    }
    shifts
}

/// Render the numbered options. Returns (prompt, shifts) — shifts[n-1] is the
/// shift encoded by option n.
pub fn build_prompt(left: &SpeakerTurn, right: &SpeakerTurn) -> Option<(String, Vec<i32>)> {
    let l_ranges = word_ranges(left);
    let r_ranges = word_ranges(right);
    let l_words = words_text(left, &l_ranges);
    let r_words = words_text(right, &r_ranges);
    let shifts = candidate_shifts(&l_words, &r_words);
    if shifts.len() < 2 {
        return None;
    }

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
    // BOUNDARY_PROMPT_VARIANT (eval-only knob): "blind" hides which option is
    // the current split, "neutral" marks it without the keep instruction.
    let variant = std::env::var("BOUNDARY_PROMPT_VARIANT").unwrap_or_default();
    if variant != "blind" {
        if let Some(current) = shifts.iter().position(|&s| s == 0) {
            if variant == "neutral" {
                prompt.push_str(&format!(
                    "Option {} is the current split, based on voice analysis.\n",
                    current + 1
                ));
            } else {
                prompt.push_str(&format!(
                    "Option {} is the current split, based on voice analysis. Prefer it unless it breaks a sentence mid-thought — a speaker change usually happens at a sentence boundary, not inside one.\n",
                    current + 1
                ));
                if variant == "clause" {
                    prompt.push_str(
                        "But if the words right after the split clearly finish A's sentence, they belong to A — even if that moves the split several words.\n",
                    );
                }
            }
        }
    }
    prompt.push_str(&format!(
        "Reply with only JSON like {{\"cut\": 1}} choosing the best option (1-{}).",
        shifts.len()
    ));
    Some((prompt, shifts))
}

/// Extract the integer following `key` in the model's reply. Accepts any
/// text that contains the key followed by an integer, JSON or not.
pub fn parse_key(reply: &str, key: &str) -> Option<usize> {
    let idx = reply.find(key)?;
    let digits: String = reply[idx + key.len()..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

pub fn parse_cut(reply: &str) -> Option<usize> {
    parse_key(reply, "cut")
}

/// A pick at the edge of the LOCAL window signals the true cut may lie
/// beyond it — recenter and re-ask. A far punctuation jump is final: the
/// model chose a sentence boundary on purpose; re-asking there only
/// produces jitter (observed: a perfect "...were younger?" cut re-asked and
/// nudged one word back).
fn is_extreme_shift(shift: i32, shifts: &[i32]) -> bool {
    shift != 0
        && shift.unsigned_abs() as usize == MAX_SHIFT_WORDS
        && (Some(&shift) == shifts.first() || Some(&shift) == shifts.last())
}

/// True when mid is a short different-speaker turn wedged inside one
/// speaker's speech with tight gaps on both sides.
pub fn is_sandwich(prev: &SpeakerTurn, mid: &SpeakerTurn, next: &SpeakerTurn) -> bool {
    let (Some(p), Some(m), Some(n)) = (prev.speaker, mid.speaker, next.speaker) else {
        return false;
    };
    if p != n || m == p || prev.words.is_empty() || mid.words.is_empty() || next.words.is_empty() {
        return false;
    }
    if word_ranges(mid).len() > SANDWICH_MAX_WORDS {
        return false;
    }
    let tight = |left: &SpeakerTurn, right: &SpeakerTurn| match (left.words.last(), right.words.first()) {
        (Some(a), Some(b)) => b.start_ms - a.end_ms < TIGHT_GAP_MS,
        _ => false,
    };
    tight(prev, mid) && tight(mid, next)
}

pub fn build_sandwich_prompt(prev: &SpeakerTurn, mid: &SpeakerTurn, next: &SpeakerTurn) -> String {
    let tail = |t: &SpeakerTurn| {
        let r = word_ranges(t);
        let w = words_text(t, &r);
        let start = w.len().saturating_sub(CONTEXT_WORDS);
        let mut s = if start > 0 { String::from("... ") } else { String::new() };
        s.push_str(&w[start..].join(" "));
        s
    };
    let head = |t: &SpeakerTurn| {
        let r = word_ranges(t);
        let w = words_text(t, &r);
        let end = CONTEXT_WORDS.min(w.len());
        let mut s = w[..end].join(" ");
        if end < w.len() {
            s.push_str(" ...");
        }
        s
    };
    format!(
        "In this transcript excerpt the middle line was attributed to a second speaker B interrupting speaker A:\n\nA: \"{}\"\nB: \"{}\"\nA: \"{}\"\n\nDoes the B line read as part of A's own continuous sentence (a diarization mistake) rather than another person actually speaking?\nReply with only JSON: {{\"merge\": 1}} if it is A's own sentence, {{\"merge\": 0}} if B is really another person.",
        tail(prev),
        mid.text.trim(),
        head(next)
    )
}

/// Fold turns[mid_idx] and turns[mid_idx + 1] into turns[mid_idx - 1].
fn merge_sandwich(turns: &mut Vec<SpeakerTurn>, mid_idx: usize) {
    let next = turns.remove(mid_idx + 1);
    let mid = turns.remove(mid_idx);
    let prev = &mut turns[mid_idx - 1];
    prev.words.extend(mid.words);
    prev.words.extend(next.words);
    prev.rebuild_from_words();
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

/// The instant (ms) where the cut would fall for a given shift: the middle
/// of the gap between the last A-side word and the first B-side word.
pub fn cut_instant_ms(left: &SpeakerTurn, right: &SpeakerTurn, shift: i32) -> Option<f64> {
    let l_ranges = word_ranges(left);
    let r_ranges = word_ranges(right);
    let word_end = |turn: &SpeakerTurn, r: &(usize, usize)| turn.words[r.1 - 1].end_ms;
    let word_start = |turn: &SpeakerTurn, r: &(usize, usize)| turn.words[r.0].start_ms;
    let (a_end_ms, b_start_ms) = if shift < 0 {
        let keep = l_ranges.len().checked_sub((-shift) as usize)?;
        if keep == 0 {
            return None;
        }
        (
            word_end(left, l_ranges.get(keep - 1)?),
            word_start(left, l_ranges.get(keep)?),
        )
    } else if shift > 0 {
        (
            word_end(right, r_ranges.get(shift as usize - 1)?),
            word_start(right, r_ranges.get(shift as usize)?),
        )
    } else {
        (word_end(left, l_ranges.last()?), word_start(right, r_ranges.first()?))
    };
    Some((a_end_ms + b_start_ms) / 2.0)
}

/// Frame-level acoustic verdict for one boundary: Some(shift) when exactly
/// one candidate cut falls on the pause between two different sustained
/// voices near the current cut (0 = the current cut is confirmed there).
/// None = no usable evidence; fall through to the LLM.
///
/// Deliberately asymmetric: this can reposition or confirm a cut, never
/// remove one — the segmentation model sometimes hears two real speakers
/// as one local voice, so "same voice" must not veto a boundary.
fn acoustic_verdict(
    seg: &mut SegmentationModel,
    audio_16k: &[f32],
    left: &SpeakerTurn,
    right: &SpeakerTurn,
    shifts: &[i32],
) -> Option<i32> {
    let t_cut = cut_instant_ms(left, right, 0)? / 1000.0;
    let duration = audio_16k.len() as f64 / segmentation::SAMPLE_RATE as f64;
    let window_start = (t_cut - segmentation::WINDOW_SECS / 2.0)
        .clamp(0.0, (duration - segmentation::WINDOW_SECS).max(0.0));
    let runs = match seg.voice_runs(audio_16k, window_start) {
        Ok(r) => r,
        Err(e) => {
            warn!("Boundary refine: segmentation window failed: {}", e);
            return None;
        }
    };
    let gap = segmentation::change_gap_near(&runs, t_cut)?;
    let instants: Vec<(i32, f64)> = shifts
        .iter()
        .filter_map(|&s| cut_instant_ms(left, right, s).map(|ms| (s, ms / 1000.0)))
        .collect();
    segmentation::unique_candidate_in_gap(&instants, gap)
}

/// Pick the best available local model for micro-queries: prefer the smaller
/// Qwen if downloaded, fall back to any downloaded built-in model.
pub fn pick_model(app_data_dir: &PathBuf) -> Option<models::ModelDef> {
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

/// Shared budget/failure tracking across both query passes.
struct QueryCtx {
    app_data_dir: PathBuf,
    model_name: String,
    started: Instant,
    consecutive_failures: usize,
}

impl QueryCtx {
    fn exhausted(&self, stats: &RefineStats) -> bool {
        self.started.elapsed() > TOTAL_BUDGET
            || self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES
            || stats.queried >= MAX_QUERIES
    }

    async fn ask(&mut self, prompt: &str, stats: &mut RefineStats) -> Option<String> {
        stats.queried += 1;
        match summary_engine::generate_micro(
            &self.app_data_dir,
            &self.model_name,
            SYSTEM_PROMPT,
            prompt,
            16,
            PER_QUERY_TIMEOUT,
        )
        .await
        {
            Ok(reply) => {
                self.consecutive_failures = 0;
                Some(reply)
            }
            Err(e) => {
                warn!("Boundary refine: LLM query failed: {}", e);
                stats.failures += 1;
                self.consecutive_failures += 1;
                None
            }
        }
    }
}

/// Refine speaker-turn boundaries in place. Never fails the retranscription:
/// any error just leaves the affected boundary as the acoustic pass set it.
///
/// Two passes: (1) phantom-interjection merge — a short turn wedged between
/// two turns of one other speaker is folded back when the LLM reads it as
/// that speaker's own sentence; (2) cut-position refinement — each remaining
/// tight handover gets its cut re-chosen from ±MAX_SHIFT_WORDS candidates,
/// recentering and re-asking when the LLM picks the outermost option.
pub async fn refine_turns<R: Runtime>(
    app: &AppHandle<R>,
    turns: &mut Vec<SpeakerTurn>,
    audio_16k: Option<&[f32]>,
    cancelled: impl Fn() -> bool,
    mut on_progress: impl FnMut(usize, usize),
) -> RefineStats {
    let mut stats = RefineStats::default();
    if turns.len() < 2 {
        return stats;
    }

    let Ok(app_data_dir) = app.path().app_data_dir() else {
        warn!("Boundary refine: no app data dir, skipping");
        return stats;
    };
    let Some(model) = pick_model(&app_data_dir) else {
        info!("Boundary refine: no local LLM downloaded, skipping");
        return stats;
    };

    // Frame-level acoustic evidence (needs the audio and the downloaded
    // segmentation model). Unavailable is fine — the LLM path stands alone.
    let mut seg_model = audio_16k.and_then(|_| {
        let path = super::diarization::segmentation_model_path(app).ok()?;
        if !path.exists() {
            return None;
        }
        match SegmentationModel::load(&path) {
            Ok(m) => Some(m),
            Err(e) => {
                warn!("Boundary refine: segmentation model load failed: {}", e);
                None
            }
        }
    });
    let mut ctx = QueryCtx {
        app_data_dir,
        model_name: model.name.clone(),
        started: Instant::now(),
        consecutive_failures: 0,
    };

    // Pass 1: phantom interjections. Capped at half the query budget so a
    // sandwich-heavy meeting cannot starve pass 2 (observed: a 57-min call
    // produced 32 sandwiches and consumed every query before any cut was
    // refined).
    let mut i = 1;
    while i + 1 < turns.len() {
        if cancelled() {
            break;
        }
        if is_sandwich(&turns[i - 1], &turns[i], &turns[i + 1]) {
            stats.sandwiches += 1;
            // Once the pass-1 query share is spent, keep scanning without
            // querying so `sandwiches` reports how many actually exist.
            if !ctx.exhausted(&stats) && stats.queried < MAX_QUERIES / 2 {
                let prompt = build_sandwich_prompt(&turns[i - 1], &turns[i], &turns[i + 1]);
                if let Some(reply) = ctx.ask(&prompt, &mut stats).await {
                    match parse_key(&reply, "merge") {
                        Some(1) => {
                            info!(
                                "Boundary refine: merging phantom interjection {:?} at {:.1}s",
                                turns[i].text.trim(),
                                turns[i].start_ms / 1000.0
                            );
                            merge_sandwich(turns, i);
                            stats.merged += 1;
                            // Same index now holds the following turn; re-check
                            // without advancing
                            continue;
                        }
                        Some(0) => {
                            info!(
                                "Boundary refine: keeping interjection {:?} at {:.1}s",
                                turns[i].text.trim(),
                                turns[i].start_ms / 1000.0
                            );
                        }
                        other => {
                            warn!(
                                "Boundary refine: unparseable merge reply {:?} ({:?})",
                                reply, other
                            );
                            stats.failures += 1;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    let pass1_queried = stats.queried;

    // Pass 2: cut positions on the (possibly merged) turn list
    let boundaries = find_boundaries(turns);
    stats.boundaries = boundaries.len();
    info!(
        "Boundary refine: {} sandwiches ({} queried, {} merged), {} tight boundaries, model {}",
        stats.sandwiches, pass1_queried, stats.merged, stats.boundaries, ctx.model_name
    );
    let total = boundaries.len();

    let mut llm_stopped = false;
    for (done, boundary) in boundaries.iter().enumerate() {
        if cancelled() {
            warn!(
                "Boundary refine: cancelled after {} of {} boundaries ({} queries)",
                done, total, stats.queried
            );
            break;
        }
        on_progress(done, total);

        // Acoustic gate: when frame-level segmentation places exactly one
        // candidate cut on the voice change, the decision is made without
        // (and above) the LLM — measured stronger than both the local 2B
        // and a frontier model on the labeled handover cases.
        if let (Some(seg), Some(audio)) = (seg_model.as_mut(), audio_16k) {
            let (left_slice, right_slice) = turns.split_at_mut(boundary.left_idx + 1);
            let left = &mut left_slice[boundary.left_idx];
            let right = &mut right_slice[0];
            let l_words = words_text(left, &word_ranges(left));
            let r_words = words_text(right, &word_ranges(right));
            let shifts = candidate_shifts(&l_words, &r_words);
            match acoustic_verdict(seg, audio, left, right, &shifts) {
                Some(0) => {
                    info!(
                        "Boundary refine: acoustic voice change confirms cut at {:.1}s",
                        right.start_ms / 1000.0
                    );
                    stats.acoustic_confirmed += 1;
                    continue;
                }
                Some(shift) => {
                    info!(
                        "Boundary refine: acoustic voice change moves {} word(s) {} at {:.1}s",
                        shift.abs(),
                        if shift < 0 { "right" } else { "left" },
                        right.start_ms / 1000.0
                    );
                    apply_shift(left, right, shift);
                    stats.moved += 1;
                    stats.acoustic_moved += 1;
                    continue;
                }
                None => {}
            }
        }

        // The acoustic gate above is free; only the LLM fallback needs
        // budget. Keep walking boundaries after exhaustion so the gate
        // still decides what it can (observed: a 57-min meeting's sandwich
        // pass drained every query and the gate never ran).
        if ctx.exhausted(&stats) {
            if !llm_stopped {
                warn!(
                    "Boundary refine: LLM budget exhausted at {} of {} boundaries ({} queries); acoustic gate continues",
                    done, total, stats.queried
                );
                llm_stopped = true;
            }
            continue;
        }

        let mut rounds = 0;
        loop {
            rounds += 1;
            let (left_slice, right_slice) = turns.split_at_mut(boundary.left_idx + 1);
            let left = &mut left_slice[boundary.left_idx];
            let right = &mut right_slice[0];

            let Some((prompt, shifts)) = build_prompt(left, right) else {
                break;
            };
            let Some(reply) = ctx.ask(&prompt, &mut stats).await else {
                break;
            };
            match parse_cut(&reply) {
                Some(n) if (1..=shifts.len()).contains(&n) => {
                    let shift = shifts[n - 1];
                    if shift == 0 {
                        info!(
                            "Boundary refine: keeping cut at {:.1}s (round {})",
                            right.start_ms / 1000.0,
                            rounds
                        );
                        break;
                    }
                    info!(
                        "Boundary refine: moving {} word(s) {} at {:.1}s (round {})",
                        shift.abs(),
                        if shift < 0 { "right" } else { "left" },
                        right.start_ms / 1000.0,
                        rounds
                    );
                    apply_shift(left, right, shift);
                    stats.moved += 1;
                    // Outermost pick: the true cut may lie further out —
                    // recenter and ask again
                    if !is_extreme_shift(shift, &shifts)
                        || rounds >= MAX_ROUNDS_PER_BOUNDARY
                        || ctx.exhausted(&stats)
                    {
                        break;
                    }
                }
                other => {
                    warn!("Boundary refine: unparseable cut reply {:?} ({:?})", reply, other);
                    stats.failures += 1;
                    break;
                }
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
        WordSpan { text: text.to_string(), start_ms, end_ms, prob: 1.0 }
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

    /// Dry-run the acoustic gate against a real meeting folder (no LLM, no
    /// writes): prints each tight boundary's verdict for inspection.
    ///
    ///   BOUNDARY_DRYRUN_FOLDER="~/Movies/meetily-recordings/<meeting>" \
    ///     cargo test --lib boundary_dryrun -- --ignored --nocapture
    #[test]
    #[ignore]
    fn boundary_dryrun_on_real_meeting() {
        let Ok(folder) = std::env::var("BOUNDARY_DRYRUN_FOLDER") else {
            eprintln!("BOUNDARY_DRYRUN_FOLDER not set");
            return;
        };
        let folder = std::path::PathBuf::from(folder);
        #[derive(serde::Deserialize)]
        struct Data {
            turns: Vec<SpeakerTurn>,
        }
        let raw = std::fs::read_to_string(folder.join("refine-data.json")).expect("refine-data.json");
        let data: Data = serde_json::from_str(&raw).expect("parse refine-data");
        let decoded = crate::audio::decoder::decode_audio_file(&folder.join("audio.mp4")).expect("decode");
        let audio = decoded.to_whisper_format();
        let model_path = dirs::data_dir()
            .expect("data dir")
            .join("com.meetily.ai/models/diarization/segmentation-3.0.onnx");
        let mut seg = SegmentationModel::load(&model_path).expect("segmentation model");

        let turns = data.turns;
        let boundaries = find_boundaries(&turns);
        println!("{} turns, {} tight boundaries", turns.len(), boundaries.len());
        for b in boundaries {
            let left = &turns[b.left_idx];
            let right = &turns[b.left_idx + 1];
            let l_words = words_text(left, &word_ranges(left));
            let r_words = words_text(right, &word_ranges(right));
            let shifts = candidate_shifts(&l_words, &r_words);
            let verdict = acoustic_verdict(&mut seg, &audio, left, right, &shifts);
            let t = cut_instant_ms(left, right, 0).unwrap_or(0.0) / 1000.0;
            let tail: Vec<_> = l_words.iter().rev().take(5).rev().cloned().collect();
            let head: Vec<_> = r_words.iter().take(5).cloned().collect();
            println!(
                "t={:7.2}s verdict={:>8} shifts={:?} | ...{} ‖ {}...",
                t,
                verdict.map(|s| s.to_string()).unwrap_or_else(|| "none".into()),
                shifts,
                tail.join(" "),
                head.join(" ")
            );
            if std::env::var("BOUNDARY_DRYRUN_VERBOSE").is_ok() {
                let ws = (t - segmentation::WINDOW_SECS / 2.0)
                    .clamp(0.0, (audio.len() as f64 / 16000.0 - segmentation::WINDOW_SECS).max(0.0));
                let runs = seg.voice_runs(&audio, ws).unwrap();
                let speech: Vec<_> = runs
                    .iter()
                    .filter(|r| r.class != 0)
                    .map(|r| format!("{:.2}-{:.2} c{}", r.start, r.end, r.class))
                    .collect();
                println!("  speech runs: {:?}", speech);
                println!("  gap near {:.2}: {:?}", t, segmentation::change_gap_near(&runs, t));
                println!(
                    "  instants: {:?}",
                    shifts
                        .iter()
                        .map(|&s| (s, cut_instant_ms(left, right, s).map(|m| (m / 1000.0 * 100.0).round() / 100.0)))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn cut_instants_track_word_gaps() {
        let (left, right) = are_boundary();
        // Current cut: between "Are" (ends 29200) and "they" (starts 30100)
        assert_eq!(cut_instant_ms(&left, &right, 0), Some(29650.0));
        // Shift -1: between "them?" (ends 28600) and "Are" (starts 28900)
        assert_eq!(cut_instant_ms(&left, &right, -1), Some(28750.0));
        // Shift +1: between "they" (ends 30300) and "moving" (starts 30300)
        assert_eq!(cut_instant_ms(&left, &right, 1), Some(30300.0));
        // Shifts that would empty a side are rejected
        assert_eq!(cut_instant_ms(&left, &right, -6), None);
        assert_eq!(cut_instant_ms(&left, &right, 5), None);
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

    fn plain_words(n: usize) -> Vec<String> {
        (0..n).map(|k| format!("w{}", k)).collect()
    }

    #[test]
    fn candidate_shifts_respect_one_word_minimum() {
        assert_eq!(candidate_shifts(&plain_words(6), &plain_words(5)), vec![-2, -1, 0, 1, 2]);
        assert_eq!(candidate_shifts(&plain_words(1), &plain_words(5)), vec![0, 1, 2]);
        assert_eq!(candidate_shifts(&plain_words(2), &plain_words(1)), vec![-1, 0]);
        assert_eq!(candidate_shifts(&plain_words(1), &plain_words(1)), vec![0]);
    }

    #[test]
    fn punctuation_cut_candidates_reach_past_the_local_window() {
        // The hangovers case: correct cut is 7 words into the right turn,
        // right after "younger?"
        let l_words: Vec<String> =
            ["You", "think", "your", "hangovers", "are"].map(String::from).to_vec();
        let r_words: Vec<String> = [
            "worse", "now", "than", "when", "you", "were", "younger?", "Like,", "the", "main",
            "problem", "is,",
        ]
        .map(String::from)
        .to_vec();
        let shifts = candidate_shifts(&l_words, &r_words);
        assert!(shifts.contains(&7), "punct candidate after 'younger?' missing: {:?}", shifts);
        assert!(shifts.len() <= MAX_OPTIONS);
        // No punctuation on the left side within reach, so nothing beyond -2
        assert_eq!(shifts.iter().copied().min(), Some(-2));
    }

    #[test]
    fn hangovers_prompt_offers_the_full_question_option() {
        let left = turn(
            vec![
                tok("You", 121000.0, 121200.0),
                tok(" think", 121200.0, 121500.0),
                tok(" your", 121500.0, 121700.0),
                tok(" hangovers", 121700.0, 122300.0),
                tok(" are", 122300.0, 122500.0),
            ],
            1,
        );
        let right = turn(
            vec![
                tok(" worse", 123000.0, 123300.0),
                tok(" now", 123300.0, 123500.0),
                tok(" than", 123500.0, 123700.0),
                tok(" when", 123700.0, 123900.0),
                tok(" you", 123900.0, 124000.0),
                tok(" were", 124000.0, 124200.0),
                tok(" younger?", 124200.0, 124800.0),
                tok(" Like,", 125200.0, 125500.0),
                tok(" the", 125500.0, 125600.0),
                tok(" main", 125600.0, 125900.0),
                tok(" problem", 125900.0, 126400.0),
            ],
            2,
        );
        let (prompt, shifts) = build_prompt(&left, &right).expect("prompt");
        let full_q = shifts.iter().position(|&s| s == 7).expect("shift 7 offered") + 1;
        assert!(prompt.contains(&format!(
            "Option {}:\nA: \"You think your hangovers are worse now than when you were younger?\"\nB: \"Like, the main problem\"",
            full_q
        )), "prompt:\n{}", prompt);
    }

    #[test]
    fn five_word_response_phrase_is_not_a_sandwich() {
        // "Yeah, you're right, but you" is a REAL response — merging it was
        // the regression in the 2nd meeting run
        let mut t = sandwich_turns();
        t[1] = turn(
            vec![
                tok(" Yeah,", 1600.0, 1700.0),
                tok(" you're", 1700.0, 1800.0),
                tok(" right,", 1800.0, 2000.0),
                tok(" but", 2000.0, 2100.0),
                tok(" you", 2100.0, 2300.0),
            ],
            1,
        );
        assert!(!is_sandwich(&t[0], &t[1], &t[2]));
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

    // The real phantom-interjection case: "Yeah, you're right, but you
    // should have | proper people | around you."
    fn sandwich_turns() -> Vec<SpeakerTurn> {
        vec![
            turn(
                vec![
                    tok("Yeah,", 0.0, 300.0),
                    tok(" you're", 300.0, 500.0),
                    tok(" right,", 500.0, 800.0),
                    tok(" but", 800.0, 900.0),
                    tok(" you", 900.0, 1000.0),
                    tok(" should", 1000.0, 1300.0),
                    tok(" have", 1300.0, 1500.0),
                ],
                2,
            ),
            turn(vec![tok(" proper", 1600.0, 1900.0), tok(" people", 1900.0, 2300.0)], 1),
            turn(
                vec![
                    tok(" around", 2400.0, 2700.0),
                    tok(" you.", 2700.0, 3000.0),
                    tok(" I", 3100.0, 3200.0),
                    tok(" find", 3200.0, 3500.0),
                ],
                2,
            ),
        ]
    }

    #[test]
    fn sandwich_detected_only_between_same_speaker_neighbors() {
        let t = sandwich_turns();
        assert!(is_sandwich(&t[0], &t[1], &t[2]));

        let mut different = t.clone();
        different[2].speaker = Some(0);
        assert!(!is_sandwich(&different[0], &different[1], &different[2]));

        let mut long_mid = t.clone();
        long_mid[1] = turn(
            (0..6).map(|k| tok(&format!(" w{}", k), 1600.0 + k as f64 * 100.0, 1700.0 + k as f64 * 100.0)).collect(),
            1,
        );
        assert!(!is_sandwich(&long_mid[0], &long_mid[1], &long_mid[2]));
    }

    #[test]
    fn sandwich_prompt_shows_three_lines() {
        let t = sandwich_turns();
        let prompt = build_sandwich_prompt(&t[0], &t[1], &t[2]);
        assert!(prompt.contains("A: \"Yeah, you're right, but you should have\""));
        assert!(prompt.contains("B: \"proper people\""));
        assert!(prompt.contains("A: \"around you. I find\""));
        assert!(prompt.contains("{\"merge\": 1}"));
    }

    #[test]
    fn merge_sandwich_folds_three_turns_into_one() {
        let mut t = sandwich_turns();
        merge_sandwich(&mut t, 1);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].text.trim(), "Yeah, you're right, but you should have proper people around you. I find");
        assert_eq!(t[0].speaker, Some(2));
        assert!((t[0].end_ms - 3500.0).abs() < 1e-6);
    }

    #[test]
    fn parse_key_reads_merge_replies() {
        assert_eq!(parse_key("{\"merge\": 1}", "merge"), Some(1));
        assert_eq!(parse_key("{\"merge\":0}", "merge"), Some(0));
        assert_eq!(parse_key("nothing here", "merge"), None);
    }

    #[test]
    fn extreme_shift_is_only_the_local_window_edge() {
        let shifts = vec![-2, -1, 0, 1, 2];
        assert!(is_extreme_shift(-2, &shifts));
        assert!(is_extreme_shift(2, &shifts));
        assert!(!is_extreme_shift(-1, &shifts));
        assert!(!is_extreme_shift(1, &shifts));
        assert!(!is_extreme_shift(0, &shifts));
        // One-sided window: 0 can be an endpoint but is never "extreme"
        assert!(!is_extreme_shift(0, &[0, 1, 2]));
        assert!(is_extreme_shift(2, &[0, 1, 2]));
        // A punctuation jump is final — never re-asked; and a local pick is
        // not "extreme" when a farther option was available and declined
        let with_punct = vec![-2, -1, 0, 1, 2, 7];
        assert!(!is_extreme_shift(7, &with_punct));
        assert!(!is_extreme_shift(2, &with_punct));
        assert!(is_extreme_shift(-2, &with_punct));
    }

    #[test]
    fn no_far_candidates_when_cut_already_at_sentence_end() {
        // "...these platforms. | What do you call them? Are they..." — the
        // diarizer's cut agrees with the punctuation; grabbing the whole
        // next question must not even be an option.
        let l_words: Vec<String> =
            ["like", "these", "platforms."].map(String::from).to_vec();
        let r_words: Vec<String> =
            ["What", "do", "you", "call", "them?", "Are", "they", "moving"].map(String::from).to_vec();
        let shifts = candidate_shifts(&l_words, &r_words);
        assert_eq!(shifts, vec![-2, -1, 0, 1, 2]);
    }

    #[test]
    fn fragment_ending_does_not_earn_trust() {
        // "...to the Odyssey? Sorry? | Sorry. You go, George. Oh, yeah,
        // okay." — left ends on punctuation, but "Sorry?" is a one-word
        // fragment: the true cut (+4, after "George.") must stay reachable.
        let l_words: Vec<String> =
            ["been", "to", "the", "Odyssey?", "Sorry?"].map(String::from).to_vec();
        let r_words: Vec<String> = [
            "Sorry.", "You", "go,", "George.", "Oh,", "yeah,", "okay.", "Actually,", "I'm",
            "kind", "of", "missing",
        ]
        .map(String::from)
        .to_vec();
        let shifts = candidate_shifts(&l_words, &r_words);
        assert!(shifts.contains(&4), "cut after 'George.' must be offered: {:?}", shifts);
    }

    #[test]
    fn prompt_marks_the_current_split_option() {
        let (left, right) = are_boundary();
        let (prompt, shifts) = build_prompt(&left, &right).expect("prompt");
        let current = shifts.iter().position(|&s| s == 0).unwrap() + 1;
        assert!(prompt.contains(&format!("Option {} is the current split", current)));
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
