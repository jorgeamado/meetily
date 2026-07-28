// Confidence-gated micro-repair of transcription text.
//
// Whisper reports a probability per decoded token. Sentences containing
// low-probability tokens — garbled words, wrong-language transliterations in
// code-switched speech — are shown to the local LLM one at a time with tight
// context and a hard contract: fix at most two clearly mis-heard words or
// return the sentence unchanged. A word-level diff guard rejects any reply
// that rewrites more than allowed, so the model can never paraphrase the
// transcript, only patch it.

use std::time::{Duration, Instant};

use log::{info, warn};
use tauri::{AppHandle, Manager, Runtime};

use super::boundary_refine::{ends_sentence, pick_model, word_ranges};
use super::diarization::SpeakerTurn;
use crate::summary::summary_engine;

/// A sentence is flagged when any word's (min-token) probability falls
/// below this...
const MIN_WORD_PROB: f32 = 0.45;
/// ...or the sentence's mean word probability falls below this.
const MIN_SENT_MEAN_PROB: f32 = 0.70;

/// Sentences shorter than this are too noisy to judge ("Mm.", "Да."), and
/// longer ones dilute the context window.
const MIN_SENT_WORDS: usize = 3;
const MAX_SENT_WORDS: usize = 40;

/// The reply may change at most this many words (word-level edit
/// operations) or it is rejected wholesale.
const MAX_CHANGED_WORDS: usize = 2;

/// Hard caps so repair never dominates a run.
const MAX_REPAIRS: usize = 15;
const TOTAL_BUDGET: Duration = Duration::from_secs(150);
const PER_QUERY_TIMEOUT: Duration = Duration::from_secs(45);
const MAX_CONSECUTIVE_FAILURES: usize = 2;

const SYSTEM_PROMPT: &str = "You fix isolated speech-recognition errors in meeting transcripts. \
Answer with JSON only, no explanation.";

#[derive(Debug, Default, Clone, Copy)]
pub struct RepairStats {
    pub flagged: usize,
    pub queried: usize,
    pub repaired: usize,
    pub failures: usize,
}

/// One sentence of a turn: its display text and word-level confidence.
#[derive(Debug, Clone)]
struct Sentence {
    text: String,
    min_prob: f32,
    mean_prob: f32,
    word_count: usize,
}

/// Words of a turn as (trimmed text, min token probability).
fn word_list(turn: &SpeakerTurn) -> Vec<(String, f32)> {
    word_ranges(turn)
        .iter()
        .map(|&(a, b)| {
            let text: String = turn.words[a..b].iter().map(|w| w.text.as_str()).collect();
            let prob = turn.words[a..b]
                .iter()
                .map(|w| w.prob)
                .fold(f32::INFINITY, f32::min);
            (text.trim().to_string(), if prob.is_finite() { prob } else { 1.0 })
        })
        .filter(|(t, _)| !t.is_empty())
        .collect()
}

/// Split a turn into sentences at sentence-final punctuation.
fn sentences_of(turn: &SpeakerTurn) -> Vec<Sentence> {
    let words = word_list(turn);
    let mut out: Vec<Sentence> = Vec::new();
    let mut cur: Vec<(String, f32)> = Vec::new();
    for w in words {
        let is_end = ends_sentence(&w.0);
        cur.push(w);
        if is_end {
            out.push(make_sentence(std::mem::take(&mut cur)));
        }
    }
    if !cur.is_empty() {
        out.push(make_sentence(cur));
    }
    out
}

fn make_sentence(words: Vec<(String, f32)>) -> Sentence {
    let n = words.len().max(1);
    let min_prob = words.iter().map(|w| w.1).fold(f32::INFINITY, f32::min);
    let mean_prob = words.iter().map(|w| w.1).sum::<f32>() / n as f32;
    Sentence {
        text: words.iter().map(|w| w.0.as_str()).collect::<Vec<_>>().join(" "),
        min_prob: if min_prob.is_finite() { min_prob } else { 1.0 },
        mean_prob,
        word_count: words.len(),
    }
}

fn is_flagged(s: &Sentence) -> bool {
    (MIN_SENT_WORDS..=MAX_SENT_WORDS).contains(&s.word_count)
        && (s.min_prob < MIN_WORD_PROB || s.mean_prob < MIN_SENT_MEAN_PROB)
}

/// Word-level Levenshtein distance: number of insert/delete/substitute
/// operations between the two word sequences.
fn word_edit_ops(a: &[&str], b: &[&str]) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn build_prompt(before: &str, sentence: &str, after: &str) -> String {
    let mut prompt = String::from(
        "A sentence from a meeting transcript (the speech may mix languages mid-sentence). \
The speech recognizer may have mis-heard a word or two.\n\n",
    );
    if !before.is_empty() {
        prompt.push_str(&format!("Before: \"{}\"\n", before));
    }
    prompt.push_str(&format!("Sentence: \"{}\"\n", sentence));
    if !after.is_empty() {
        prompt.push_str(&format!("After: \"{}\"\n", after));
    }
    prompt.push_str(
        "\nIf a word in the Sentence is clearly a mis-recognition, return the corrected \
sentence, changing as few words as possible (at most 2). If it reads fine, return it \
exactly unchanged.\nReply with only JSON: {\"text\": \"<sentence>\"}",
    );
    prompt
}

/// Extract {"text": ...} from the reply, tolerating surrounding noise.
fn parse_text_reply(reply: &str) -> Option<String> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end <= start {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(&reply[start..=end]).ok()?;
    value.get("text")?.as_str().map(|s| s.trim().to_string())
}

/// Accept a repair only when it is a small patch, not a paraphrase.
fn accept_repair(original: &str, candidate: &str) -> bool {
    if candidate.is_empty() || candidate == original {
        return false;
    }
    let a: Vec<&str> = original.split_whitespace().collect();
    let b: Vec<&str> = candidate.split_whitespace().collect();
    if b.len() + 2 < a.len() || b.len() > a.len() + 2 {
        return false;
    }
    word_edit_ops(&a, &b) <= MAX_CHANGED_WORDS
}

/// Repair low-confidence sentences in place (turn display text only; word
/// timings are untouched). Never fails the run.
pub async fn repair_turns<R: Runtime>(
    app: &AppHandle<R>,
    turns: &mut [SpeakerTurn],
    cancelled: impl Fn() -> bool,
    mut on_progress: impl FnMut(usize, usize),
) -> RepairStats {
    let mut stats = RepairStats::default();

    // Sentence-split every turn once; remember which are flagged
    let mut per_turn: Vec<Vec<Sentence>> = turns.iter().map(sentences_of).collect();
    let mut flagged: Vec<(usize, usize, f32)> = Vec::new();
    for (t, sentences) in per_turn.iter().enumerate() {
        if turns[t].words.is_empty() {
            continue; // Parakeet / no token data — no confidence to gate on
        }
        for (s, sent) in sentences.iter().enumerate() {
            if is_flagged(sent) {
                flagged.push((t, s, sent.mean_prob));
            }
        }
    }
    stats.flagged = flagged.len();
    if flagged.is_empty() {
        return stats;
    }
    if flagged.len() > MAX_REPAIRS {
        // Worst first when over budget
        flagged.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        flagged.truncate(MAX_REPAIRS);
        flagged.sort_by_key(|&(t, s, _)| (t, s));
    }

    let Ok(app_data_dir) = app.path().app_data_dir() else {
        return stats;
    };
    let Some(model) = pick_model(&app_data_dir) else {
        info!("Transcript repair: no local LLM downloaded, skipping {} sentences", flagged.len());
        return stats;
    };
    info!(
        "Transcript repair: {} flagged sentences ({} queued), model {}",
        stats.flagged,
        flagged.len(),
        model.name
    );

    let started = Instant::now();
    let mut consecutive_failures = 0usize;
    let total = flagged.len();
    let mut changed_turns: Vec<usize> = Vec::new();

    for (done, &(t, s, _)) in flagged.iter().enumerate() {
        if cancelled() || started.elapsed() > TOTAL_BUDGET || consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            warn!("Transcript repair: stopping after {} of {} sentences", done, total);
            break;
        }
        on_progress(done, total);

        let sentence = per_turn[t][s].text.clone();
        let before = if s > 0 {
            per_turn[t][s - 1].text.clone()
        } else if t > 0 {
            per_turn[t - 1].last().map(|x| x.text.clone()).unwrap_or_default()
        } else {
            String::new()
        };
        let after = per_turn[t]
            .get(s + 1)
            .map(|x| x.text.clone())
            .or_else(|| per_turn.get(t + 1).and_then(|v| v.first().map(|x| x.text.clone())))
            .unwrap_or_default();

        let max_tokens = ((sentence.split_whitespace().count() * 4) + 24).min(160) as i32;
        stats.queried += 1;
        match summary_engine::generate_micro(
            &app_data_dir,
            &model.name,
            SYSTEM_PROMPT,
            &build_prompt(&before, &sentence, &after),
            max_tokens,
            PER_QUERY_TIMEOUT,
        )
        .await
        {
            Ok(reply) => {
                consecutive_failures = 0;
                match parse_text_reply(&reply) {
                    Some(fixed) if accept_repair(&sentence, &fixed) => {
                        info!("Transcript repair: {:?} -> {:?}", sentence, fixed);
                        per_turn[t][s].text = fixed;
                        changed_turns.push(t);
                        stats.repaired += 1;
                    }
                    Some(_) => {} // unchanged or too-large rewrite — keep original
                    None => {
                        warn!("Transcript repair: unparseable reply {:?}", reply);
                        stats.failures += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Transcript repair: LLM query failed: {}", e);
                stats.failures += 1;
                consecutive_failures += 1;
            }
        }
    }

    changed_turns.dedup();
    for t in changed_turns {
        turns[t].text = per_turn[t]
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }
    on_progress(total, total);
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::diarization::WordSpan;

    fn tok(text: &str, prob: f32) -> WordSpan {
        WordSpan { text: text.to_string(), start_ms: 0.0, end_ms: 0.0, prob }
    }

    fn turn(words: Vec<WordSpan>) -> SpeakerTurn {
        SpeakerTurn {
            text: words.iter().map(|w| w.text.as_str()).collect(),
            start_ms: 0.0,
            end_ms: 0.0,
            speaker: Some(0),
            words,
        }
    }

    #[test]
    fn sentences_split_at_punctuation_with_probs() {
        let t = turn(vec![
            tok("Have", 0.9),
            tok(" you", 0.95),
            tok(" been?", 0.9),
            tok(" I", 0.9),
            tok(" was", 0.3), // weak word
            tok(" there.", 0.9),
        ]);
        let sents = sentences_of(&t);
        assert_eq!(sents.len(), 2);
        assert_eq!(sents[0].text, "Have you been?");
        assert_eq!(sents[1].text, "I was there.");
        assert!((sents[1].min_prob - 0.3).abs() < 1e-6);
        assert!(is_flagged(&sents[1]));
        assert!(!is_flagged(&sents[0]));
    }

    #[test]
    fn multi_token_word_takes_min_token_prob() {
        let t = turn(vec![tok("hel", 0.9), tok("lo", 0.2), tok(" you", 0.9), tok(" two.", 0.9)]);
        let words = word_list(&t);
        assert_eq!(words[0].0, "hello");
        assert!((words[0].1 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn short_and_long_sentences_are_not_flagged() {
        let short = make_sentence(vec![("Да.".into(), 0.1)]);
        assert!(!is_flagged(&short));
        let long_words: Vec<(String, f32)> =
            (0..50).map(|i| (format!("w{}", i), 0.5)).collect();
        assert!(!is_flagged(&make_sentence(long_words)));
    }

    #[test]
    fn edit_ops_counts_word_changes() {
        let a = ["I", "was", "there."];
        assert_eq!(word_edit_ops(&a, &["I", "was", "there."]), 0);
        assert_eq!(word_edit_ops(&a, &["I", "went", "there."]), 1);
        assert_eq!(word_edit_ops(&a, &["We", "went", "home."]), 3);
    }

    #[test]
    fn repair_guard_rejects_paraphrase_and_accepts_patch() {
        let orig = "I think we should use the caching lair here.";
        assert!(accept_repair(orig, "I think we should use the caching layer here."));
        assert!(!accept_repair(orig, orig)); // unchanged is not a repair
        assert!(!accept_repair(orig, "Let's add a cache layer at this point instead."));
        assert!(!accept_repair(orig, ""));
    }

    #[test]
    fn parse_reply_tolerates_noise() {
        assert_eq!(
            parse_text_reply("{\"text\": \"Fixed sentence.\"}").as_deref(),
            Some("Fixed sentence.")
        );
        assert_eq!(
            parse_text_reply("Sure! {\"text\":\"Ok.\"} done").as_deref(),
            Some("Ok.")
        );
        assert_eq!(parse_text_reply("no json"), None);
    }
}
