// Self-learning vocabulary for whisper's initial_prompt.
//
// After each retranscription the local LLM extracts recurring proper nouns
// and domain terms from the finished transcript; they accumulate in
// glossary.json (app data dir). The next retranscription feeds them to
// whisper as an initial_prompt, biasing recognition toward names and terms
// it would otherwise mangle — the one LLM step that improves recognition
// itself rather than repairing it afterwards.

use std::path::PathBuf;
use std::time::Duration;

use log::{info, warn};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime};

use super::boundary_refine::pick_model;
use crate::summary::summary_engine;

const GLOSSARY_FILE: &str = "glossary.json";
const MAX_TERMS: usize = 40;
const EXTRACT_TIMEOUT: Duration = Duration::from_secs(120);
/// Transcript sample cap — keeps prompt processing fast on long meetings.
const MAX_SAMPLE_CHARS: usize = 6000;

const SYSTEM_PROMPT: &str = "You extract vocabulary from meeting transcripts. \
Answer with JSON only, no explanation.";

#[derive(Debug, Default, Serialize, Deserialize)]
struct Glossary {
    terms: Vec<String>,
}

fn glossary_path<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join(GLOSSARY_FILE))
}

/// Terms learned so far (empty when none / unreadable).
pub fn load_terms<R: Runtime>(app: &AppHandle<R>) -> Vec<String> {
    let Some(path) = glossary_path(app) else { return Vec::new() };
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<Glossary>(&s).ok())
        .map(|g| g.terms)
        .unwrap_or_default()
}

/// Whisper initial_prompt built from the glossary, or None when empty.
pub fn initial_prompt<R: Runtime>(app: &AppHandle<R>) -> Option<String> {
    let terms = load_terms(app);
    (!terms.is_empty()).then(|| format!("Glossary: {}.", terms.join(", ")))
}

/// Evenly sample rows so long meetings stay under MAX_SAMPLE_CHARS.
pub fn sample_rows(rows: &[String]) -> String {
    let total: usize = rows.iter().map(|r| r.len() + 1).sum();
    let step = (total / MAX_SAMPLE_CHARS.max(1)).max(1);
    let mut out = String::new();
    for row in rows.iter().step_by(step) {
        if out.len() + row.len() > MAX_SAMPLE_CHARS {
            break;
        }
        out.push_str(row);
        out.push('\n');
    }
    out
}

fn parse_terms(reply: &str) -> Vec<String> {
    let (Some(start), Some(end)) = (reply.find('{'), reply.rfind('}')) else {
        return Vec::new();
    };
    if end <= start {
        return Vec::new();
    }
    serde_json::from_str::<serde_json::Value>(&reply[start..=end])
        .ok()
        .and_then(|v| v.get("terms").cloned())
        .and_then(|t| serde_json::from_value::<Vec<String>>(t).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.trim().to_string())
        // Single words or short phrases only; drop anything sentence-like
        .filter(|t| !t.is_empty() && t.len() <= 40 && t.split_whitespace().count() <= 3)
        .collect()
}

fn merge_terms(existing: Vec<String>, new_terms: Vec<String>) -> Vec<String> {
    let mut merged = existing;
    for term in new_terms {
        if !merged.iter().any(|t| t.eq_ignore_ascii_case(&term)) {
            merged.push(term);
        }
    }
    // Newest-learned terms live at the end; evict oldest when over cap
    if merged.len() > MAX_TERMS {
        merged.drain(..merged.len() - MAX_TERMS);
    }
    merged
}

/// Extract terms from a transcript sample and fold them into the glossary.
/// Spawned fire-and-forget after a retranscription completes.
pub async fn update_from_transcript<R: Runtime>(app: AppHandle<R>, sample: String) {
    if sample.len() < 400 {
        return; // too little text to learn from
    }
    let Ok(app_data_dir) = app.path().app_data_dir() else { return };
    let Some(model) = pick_model(&app_data_dir) else { return };

    let prompt = format!(
        "Excerpts from a work meeting transcript (mixed Russian and English):\n\n{}\n\n\
List up to 12 proper nouns — people, companies, products — and technical terms from \
these excerpts that a speech recognizer should know, exactly as spelled. Skip \
ordinary words.\nReply with only JSON: {{\"terms\": [\"...\"]}}",
        sample
    );

    match summary_engine::generate_micro(
        &app_data_dir,
        &model.name,
        SYSTEM_PROMPT,
        &prompt,
        256,
        EXTRACT_TIMEOUT,
    )
    .await
    {
        Ok(reply) => {
            let new_terms = parse_terms(&reply);
            if new_terms.is_empty() {
                info!("Glossary: no terms extracted");
                return;
            }
            let merged = merge_terms(load_terms(&app), new_terms.clone());
            let Some(path) = glossary_path(&app) else { return };
            match serde_json::to_string_pretty(&Glossary { terms: merged.clone() })
                .map_err(anyhow::Error::from)
                .and_then(|json| std::fs::write(&path, json).map_err(anyhow::Error::from))
            {
                Ok(()) => info!(
                    "Glossary: learned {:?}, {} terms total",
                    new_terms,
                    merged.len()
                ),
                Err(e) => warn!("Glossary: failed to save: {}", e),
            }
        }
        Err(e) => warn!("Glossary: extraction failed: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_terms_and_filters_sentences() {
        let terms = parse_terms(
            "{\"terms\": [\"Meetily\", \"Одиссея\", \"sherpa-onnx\", \
             \"this is a whole sentence not a term\", \"\"]}",
        );
        assert_eq!(terms, vec!["Meetily", "Одиссея", "sherpa-onnx"]);
    }

    #[test]
    fn merge_dedupes_case_insensitively_and_caps() {
        let merged = merge_terms(
            vec!["Meetily".into(), "Kafka".into()],
            vec!["meetily".into(), "Grafana".into()],
        );
        assert_eq!(merged, vec!["Meetily", "Kafka", "Grafana"]);

        let many: Vec<String> = (0..45).map(|i| format!("t{}", i)).collect();
        let capped = merge_terms(many, vec!["newest".into()]);
        assert_eq!(capped.len(), MAX_TERMS);
        assert_eq!(capped.last().map(String::as_str), Some("newest"));
    }

    #[test]
    fn sampling_caps_long_transcripts() {
        let rows: Vec<String> = (0..2000).map(|i| format!("row {} with some words", i)).collect();
        let sample = sample_rows(&rows);
        assert!(sample.len() <= MAX_SAMPLE_CHARS);
        assert!(sample.contains("row 0"));
    }
}
