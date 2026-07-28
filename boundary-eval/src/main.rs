// Offline scorer for the speaker-boundary tie-breaker.
//
// Replays the human-labeled cases from eval/out/*.jsonl through the EXACT
// production prompt builder (app_lib::audio::boundary_refine) against any
// GGUF model served by llama-helper, and reports cut-decision accuracy.
// This is the number that decides whether a smaller / fine-tuned model can
// replace the current one.
//
// Usage:
//   boundary_eval --model <path.gguf> --helper <llama-helper> <cases.jsonl>...

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use app_lib::audio::boundary_refine::{build_prompt, parse_cut};
use app_lib::audio::diarization::{SpeakerTurn, WordSpan};
use app_lib::summary::summary_engine::models::format_prompt;

const SYSTEM_PROMPT: &str = "You correct speaker-change points in conversation transcripts. \
Answer with JSON only, no explanation.";

#[derive(serde::Deserialize)]
struct Case {
    case: u32,
    meeting_id: String,
    left_tail: String,
    right_head: String,
    verdict: String,
    correct_cut_after: String,
}

struct Helper {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    model_path: String,
}

impl Helper {
    fn spawn(helper_path: &str, model_path: &str) -> Self {
        let mut child = Command::new(helper_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn llama-helper");
        let reader = BufReader::new(child.stdout.take().expect("helper stdout"));
        Helper { child, reader, model_path: model_path.to_string() }
    }

    fn generate(&mut self, prompt: &str) -> Option<String> {
        let formatted = format_prompt("qwen3.5_nonthinking", SYSTEM_PROMPT, prompt).ok()?;
        let req = serde_json::json!({
            "type": "generate",
            "prompt": formatted,
            "max_tokens": 16,
            "context_size": 4096,
            "model_path": self.model_path,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.0,
            "penalty_last_n": 0,
            "stop_tokens": ["<|im_end|>"],
        });
        let stdin = self.child.stdin.as_mut()?;
        writeln!(stdin, "{}", req).ok()?;
        let mut line = String::new();
        self.reader.read_line(&mut line).ok()?;
        let v: serde_json::Value = serde_json::from_str(&line).ok()?;
        v.get("text")?.as_str().map(|s| s.to_string())
    }
}

/// Words → synthetic token spans. Times are fabricated (300ms/word, 200ms
/// boundary gap): the prompt builder only reads token text; timings matter
/// only for candidate selection, which the labels bypass.
fn turn_from_text(text: &str, speaker: usize, start_ms: f64) -> SpeakerTurn {
    let words: Vec<WordSpan> = text
        .split_whitespace()
        .enumerate()
        .map(|(i, w)| WordSpan {
            text: if i == 0 { w.to_string() } else { format!(" {}", w) },
            start_ms: start_ms + i as f64 * 300.0,
            end_ms: start_ms + (i as f64 + 1.0) * 300.0,
            prob: 1.0,
        })
        .collect();
    let mut t = SpeakerTurn {
        text: String::new(),
        start_ms,
        end_ms: 0.0,
        speaker: Some(speaker),
        words,
    };
    t.rebuild_from_words();
    t
}

fn strip_ellipsis(s: &str) -> String {
    s.trim().trim_start_matches('…').trim_end_matches('…').trim().to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut helper = String::new();
    let mut dump: Option<PathBuf> = None;
    let mut files: Vec<PathBuf> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = args[i + 1].clone(); i += 2; }
            "--helper" => { helper = args[i + 1].clone(); i += 2; }
            "--dump" => { dump = Some(args[i + 1].clone().into()); i += 2; }
            f => { files.push(f.into()); i += 1; }
        }
    }
    assert!(
        !files.is_empty() && (dump.is_some() || (!model.is_empty() && !helper.is_empty())),
        "usage: [--model m.gguf --helper llama-helper] [--dump DIR] cases.jsonl..."
    );

    let mut h = if dump.is_none() { Some(Helper::spawn(&helper, &model)) } else { None };
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir).expect("create dump dir");
    }

    let (mut correct, mut wrong, mut unreachable, mut merge_skipped, mut skipped) = (0, 0, 0, 0, 0);
    let mut failures: Vec<String> = Vec::new();

    for file in &files {
        for line in std::fs::read_to_string(file).expect("read cases").lines() {
            let c: Case = serde_json::from_str(line).expect("parse case");
            match c.verdict.as_str() {
                "skip" => { skipped += 1; continue; }
                "wrong" if c.correct_cut_after.is_empty() => { merge_skipped += 1; continue; }
                _ => {}
            }

            let left = turn_from_text(&strip_ellipsis(&c.left_tail), 0, 0.0);
            let n_left = left.words.len() as f64;
            let right = turn_from_text(&strip_ellipsis(&c.right_head), 1, n_left * 300.0 + 200.0);
            let Some((prompt, shifts)) = build_prompt(&left, &right) else {
                skipped += 1;
                continue;
            };

            // Expected shift: 0 for "ok"; for "wrong", the shift that makes
            // the A side end exactly on correct_cut_after
            let expected: Option<i32> = if c.verdict == "ok" {
                Some(0)
            } else {
                let l_words: Vec<&str> = left.text.split_whitespace().collect();
                let r_words: Vec<&str> = right.text.split_whitespace().collect();
                let target = c.correct_cut_after.trim();
                l_words.iter().rposition(|w| *w == target)
                    .map(|p| -((l_words.len() - 1 - p) as i32))
                    .or_else(|| r_words.iter().position(|w| *w == target).map(|p| p as i32 + 1))
            };
            let Some(expected) = expected else {
                eprintln!("case {}/{}: cut word {:?} not found — skipping", &c.meeting_id[..16.min(c.meeting_id.len())], c.case, c.correct_cut_after);
                skipped += 1;
                continue;
            };
            if !shifts.contains(&expected) {
                unreachable += 1;
                eprintln!("case {} #{}: expected shift {} not among offered {:?} (candidate-gen gap)", &c.meeting_id[9..17], c.case, expected, shifts);
                continue;
            }

            if let Some(dir) = &dump {
                let meta = serde_json::json!({
                    "case": c.case,
                    "meeting": &c.meeting_id[9..17],
                    "shifts": shifts,
                    "expected_shift": expected,
                    "expected_option": shifts.iter().position(|&s| s == expected).map(|p| p + 1),
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                });
                let path = dir.join(format!("case-{}-{:02}.json", &c.meeting_id[9..17], c.case));
                std::fs::write(&path, serde_json::to_string_pretty(&meta).unwrap()).expect("write dump");
                continue;
            }
            let h = h.as_mut().unwrap();
            let reply = h.generate(&prompt).unwrap_or_default();
            let chosen = parse_cut(&reply)
                .filter(|n| (1..=shifts.len()).contains(n))
                .map(|n| shifts[n - 1]);
            if chosen == Some(expected) {
                correct += 1;
            } else {
                wrong += 1;
                failures.push(format!(
                    "  #{} ({}): expected shift {}, model chose {:?} [{} | {}]",
                    c.case, &c.meeting_id[9..17], expected, chosen,
                    strip_ellipsis(&c.left_tail), strip_ellipsis(&c.right_head)
                ));
            }
        }
    }

    let scored = correct + wrong;
    println!("MODEL {}", model.rsplit('/').next().unwrap_or(&model));
    println!(
        "  cut accuracy: {}/{} ({:.0}%)  | unreachable-by-candidates: {} | merge-type skipped: {} | unjudgeable skipped: {}",
        correct, scored,
        if scored > 0 { correct as f64 * 100.0 / scored as f64 } else { 0.0 },
        unreachable, merge_skipped, skipped
    );
    for f in &failures {
        println!("{}", f);
    }
    if let Some(mut h) = h {
        let _ = h.child.kill();
    }
}
