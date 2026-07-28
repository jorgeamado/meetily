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

/// Approved terms are user-curated and the only ones ever fed to whisper;
/// suggested terms are auto-learned and wait for human review — learning
/// from whisper's own output can otherwise promote mis-hearings
/// ("Guggenheim") into vocabulary that reinforces the error.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Glossary {
    pub approved: Vec<String>,
    pub suggested: Vec<String>,
    /// Feed approved terms to whisper as initial_prompt. Off by default:
    /// prompt injection made whisper hallucinate continuations at unclear
    /// segment starts ("Geroen Gürtel-", 2026-07-28).
    pub inject: bool,
}

fn glossary_path<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join(GLOSSARY_FILE))
}

pub fn load<R: Runtime>(app: &AppHandle<R>) -> Glossary {
    let Some(path) = glossary_path(app) else { return Glossary::default() };
    let Ok(raw) = std::fs::read_to_string(path) else { return Glossary::default() };
    parse_glossary(&raw)
}

fn parse_glossary(raw: &str) -> Glossary {
    let mut g: Glossary = serde_json::from_str(raw).unwrap_or_default();
    if g.approved.is_empty() && g.suggested.is_empty() {
        // Legacy single-list format: everything auto-learned -> suggestions
        #[derive(Deserialize)]
        struct Legacy { terms: Vec<String> }
        if let Ok(legacy) = serde_json::from_str::<Legacy>(raw) {
            g.suggested = legacy.terms;
        }
    }
    g
}

fn save<R: Runtime>(app: &AppHandle<R>, g: &Glossary) -> Result<(), String> {
    let Some(path) = glossary_path(app) else { return Err("no app data dir".into()) };
    serde_json::to_string_pretty(g)
        .map_err(|e| e.to_string())
        .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
}

/// Whisper initial_prompt from APPROVED terms only, or None when disabled
/// or empty.
pub fn initial_prompt<R: Runtime>(app: &AppHandle<R>) -> Option<String> {
    let g = load(app);
    (g.inject && !g.approved.is_empty())
        .then(|| format!("Glossary: {}.", g.approved.join(", ")))
}

#[tauri::command]
pub async fn glossary_get<R: Runtime>(app: AppHandle<R>) -> Result<Glossary, String> {
    Ok(load(&app))
}

#[tauri::command]
pub async fn glossary_save<R: Runtime>(
    app: AppHandle<R>,
    approved: Vec<String>,
    suggested: Vec<String>,
    inject: bool,
) -> Result<(), String> {
    let clean = |v: Vec<String>| -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for t in v {
            let t = t.trim().to_string();
            if !t.is_empty() && t.len() <= 40 && !out.iter().any(|x| x.eq_ignore_ascii_case(&t)) {
                out.push(t);
            }
        }
        out
    };
    let approved = clean(approved);
    let suggested = clean(suggested)
        .into_iter()
        .filter(|s| !approved.iter().any(|a| a.eq_ignore_ascii_case(s)))
        .collect();
    save(&app, &Glossary { approved, suggested, inject })
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
        "Excerpts from a work meeting transcript (possibly multilingual):\n\n{}\n\n\
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
            let mut g = load(&app);
            // Suggestions only — never touch the approved list, and don't
            // re-suggest what the user already approved
            let new_terms: Vec<String> = parse_terms(&reply)
                .into_iter()
                .filter(|t| !g.approved.iter().any(|a| a.eq_ignore_ascii_case(t)))
                .collect();
            if new_terms.is_empty() {
                info!("Glossary: no new terms extracted");
                return;
            }
            g.suggested = merge_terms(g.suggested, new_terms.clone());
            match save(&app, &g) {
                Ok(()) => info!(
                    "Glossary: suggested {:?}, {} suggestions pending review",
                    new_terms,
                    g.suggested.len()
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
    fn legacy_single_list_becomes_suggestions() {
        let g = parse_glossary(r#"{"terms": ["Odyssey", "Ramesh"]}"#);
        assert!(g.approved.is_empty());
        assert_eq!(g.suggested, vec!["Odyssey", "Ramesh"]);
        assert!(!g.inject);

        let g = parse_glossary(r#"{"approved": ["Ramesh"], "suggested": ["Kafka"], "inject": true}"#);
        assert_eq!(g.approved, vec!["Ramesh"]);
        assert_eq!(g.suggested, vec!["Kafka"]);
        assert!(g.inject);
    }

    #[test]
    fn sampling_caps_long_transcripts() {
        let rows: Vec<String> = (0..2000).map(|i| format!("row {} with some words", i)).collect();
        let sample = sample_rows(&rows);
        assert!(sample.len() <= MAX_SAMPLE_CHARS);
        assert!(sample.contains("row 0"));
    }
}
