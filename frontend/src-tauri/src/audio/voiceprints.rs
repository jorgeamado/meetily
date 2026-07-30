// Cross-meeting speaker naming.
//
// diarize-helper exports a clean-speech voice centroid per final speaker
// cluster. This module stores those per meeting (speakers.json next to the
// recording), keeps a library of NAMED voiceprints (voiceprints.json in the
// app data dir — people enter it only through an explicit rename, never
// silently), and matches new meetings' clusters against the library so
// recurring colleagues open with real names instead of "Speaker N".
//
// Matching thresholds come from a cross-meeting calibration on real
// meetings (2026-07-29, campplus): the same person across different days /
// mics scored 0.805–0.957, different people <= 0.68. Auto-apply therefore
// requires 0.75 plus a clear margin over the runner-up; 0.60–0.75 is only
// ever surfaced as a suggestion.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime};

use super::diarization::{speaker_label, LOCAL_SPEAKER};

/// Auto-apply bar for a library match.
pub const MATCH_STRONG: f32 = 0.75;
/// A strong match must beat the second-best library voice by this much.
pub const MATCH_MARGIN: f32 = 0.10;
/// Above this (but below strong): shown as a suggestion, never auto-applied.
pub const MATCH_SUGGEST: f32 = 0.60;
/// Clusters with less clean speech are neither matched nor saved as prints.
pub const MIN_CLEAN_SECS: f32 = 10.0;
/// Existing print weight is capped when averaging in a new meeting's
/// centroid so a long-known voice still adapts to mic/room changes.
const PRINT_WEIGHT_CAP: f32 = 300.0;

const LIBRARY_FILENAME: &str = "voiceprints.json";
const SPEAKERS_FILENAME: &str = "speakers.json";

/// One named person in the cross-meeting library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voiceprint {
    pub name: String,
    /// Embedding model key (campplus/eres2net); prints from different
    /// models are never compared.
    pub model: String,
    pub embedding: Vec<f32>,
    pub clean_secs: f32,
    pub meetings: u32,
    pub updated_at: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VoiceLibrary {
    pub voices: Vec<Voiceprint>,
    /// Display name for the local (mic-channel) speaker; None shows "You".
    /// Never enters voice matching — the mic channel is hardware identity.
    #[serde(default)]
    pub local_name: Option<String>,
}

/// Clean-speech centroid of one cluster, as exported by diarize-helper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCentroid {
    pub cluster: usize,
    pub clean_secs: f32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suggestion {
    pub name: String,
    pub similarity: f32,
}

/// Per-meeting speaker identity (speakers.json in the meeting folder).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MeetingSpeakers {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub centroids: Vec<ClusterCentroid>,
    /// cluster -> user-confirmed display name (manual rename or confirm)
    #[serde(default)]
    pub names: HashMap<usize, String>,
    /// cluster -> auto-applied recognized name (correctable)
    #[serde(default)]
    pub auto: HashMap<usize, String>,
    /// cluster -> possible-band match, surfaced but not applied
    #[serde(default)]
    pub suggestions: HashMap<usize, Suggestion>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return -1.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn library_path<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| anyhow!("no app data dir: {}", e))?
        .join(LIBRARY_FILENAME))
}

pub fn load_library<R: Runtime>(app: &AppHandle<R>) -> VoiceLibrary {
    let Ok(path) = library_path(app) else {
        return VoiceLibrary::default();
    };
    match std::fs::read_to_string(&path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_else(|e| {
            warn!("voiceprints.json unreadable ({}), starting empty", e);
            VoiceLibrary::default()
        }),
        Err(_) => VoiceLibrary::default(),
    }
}

fn save_library<R: Runtime>(app: &AppHandle<R>, lib: &VoiceLibrary) -> Result<()> {
    let path = library_path(app)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(lib)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn load_meeting_speakers(folder: &Path) -> MeetingSpeakers {
    match std::fs::read_to_string(folder.join(SPEAKERS_FILENAME)) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => MeetingSpeakers::default(),
    }
}

fn save_meeting_speakers(folder: &Path, ms: &MeetingSpeakers) -> Result<()> {
    let path = folder.join(SPEAKERS_FILENAME);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string(ms)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// The display-name map for a meeting: confirmed names win over auto ones.
pub fn meeting_name_map(folder: &Path) -> HashMap<usize, String> {
    let ms = load_meeting_speakers(folder);
    let mut map = ms.auto;
    map.extend(ms.names);
    map
}

/// The configured display name for the local mic-channel speaker, if any.
pub fn local_display_name<R: Runtime>(app: &AppHandle<R>) -> Option<String> {
    let name = load_library(app).local_name?;
    let name = name.trim().to_string();
    (!name.is_empty()).then_some(name)
}

/// Display string for a cluster index under a name map.
pub fn display_name(idx: usize, names: &HashMap<usize, String>) -> String {
    names.get(&idx).cloned().unwrap_or_else(|| speaker_label(idx))
}

/// Match fresh centroids against the library, write speakers.json, and
/// return the auto-applied name map. Called after diarization on every
/// retranscribe; a fresh run renumbers clusters, so the file is rebuilt
/// (library matches bring remembered names back automatically).
pub fn recognize_and_store<R: Runtime>(
    app: &AppHandle<R>,
    folder: &Path,
    model_key: &str,
    centroids: Vec<ClusterCentroid>,
) -> HashMap<usize, String> {
    let mut ms = MeetingSpeakers {
        model: model_key.to_string(),
        centroids,
        ..Default::default()
    };

    let lib = load_library(app);
    let candidates: Vec<&Voiceprint> =
        lib.voices.iter().filter(|v| v.model == model_key).collect();
    if !candidates.is_empty() {
        for c in &ms.centroids {
            if c.clean_secs < MIN_CLEAN_SECS {
                continue;
            }
            let mut scored: Vec<(f32, &str)> = candidates
                .iter()
                .map(|v| (cosine(&c.embedding, &v.embedding), v.name.as_str()))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let (best, name) = scored[0];
            let runner_up = scored.get(1).map(|s| s.0).unwrap_or(-1.0);
            if best >= MATCH_STRONG && best - runner_up >= MATCH_MARGIN {
                info!(
                    "Voice recognition: cluster {} is {} (cos {:.3}, runner-up {:.3})",
                    c.cluster, name, best, runner_up
                );
                ms.auto.insert(c.cluster, name.to_string());
            } else if best >= MATCH_SUGGEST {
                info!(
                    "Voice recognition: cluster {} possibly {} (cos {:.3}) — suggested only",
                    c.cluster, name, best
                );
                ms.suggestions.insert(
                    c.cluster,
                    Suggestion { name: name.to_string(), similarity: best },
                );
            }
        }
    }

    if let Err(e) = save_meeting_speakers(folder, &ms) {
        warn!("Failed to write speakers.json: {}", e);
    }
    ms.auto
}

/// Add or refresh a named print from a meeting cluster centroid.
fn upsert_voice(lib: &mut VoiceLibrary, name: &str, model: &str, c: &ClusterCentroid) {
    let now = chrono::Utc::now().to_rfc3339();
    if let Some(v) = lib
        .voices
        .iter_mut()
        .find(|v| v.model == model && v.name.eq_ignore_ascii_case(name))
    {
        let w_old = v.clean_secs.min(PRINT_WEIGHT_CAP);
        let w_new = c.clean_secs;
        let mut merged: Vec<f32> = v
            .embedding
            .iter()
            .zip(&c.embedding)
            .map(|(a, b)| a * w_old + b * w_new)
            .collect();
        let norm = merged.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut merged {
                *x /= norm;
            }
            v.embedding = merged;
        }
        v.clean_secs += c.clean_secs;
        v.meetings += 1;
        v.updated_at = now;
    } else {
        lib.voices.push(Voiceprint {
            name: name.to_string(),
            model: model.to_string(),
            embedding: c.embedding.clone(),
            clean_secs: c.clean_secs,
            meetings: 1,
            updated_at: now,
        });
    }
}

fn strength_word(similarity: f32) -> &'static str {
    if similarity >= MATCH_STRONG {
        "strong"
    } else if similarity >= MATCH_SUGGEST {
        "possible"
    } else {
        "weak"
    }
}

// ---------- Tauri commands ----------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterOverview {
    pub cluster: usize,
    /// Generated label ("Speaker 1") — also the row text when unnamed.
    pub label: String,
    /// Current display name (name, auto name, or the label).
    pub display: String,
    pub source: String, // "manual" | "auto" | "none"
    pub suggestion: Option<Suggestion>,
    /// Library voices ranked by similarity to this cluster.
    pub candidates: Vec<CandidateMatch>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateMatch {
    pub name: String,
    pub strength: String, // "strong" | "possible" | "weak"
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakersOverview {
    pub clusters: Vec<ClusterOverview>,
    pub known_voices: Vec<String>,
}

async fn meeting_folder<R: Runtime>(app: &AppHandle<R>, meeting_id: &str) -> Result<PathBuf, String> {
    let state = app
        .try_state::<crate::state::AppState>()
        .ok_or("app state unavailable")?;
    let pool = state.db_manager.pool();
    let path = crate::database::repositories::meeting::MeetingsRepository::get_meeting_folder_path(
        pool, meeting_id,
    )
    .await
    .map_err(|e| e.to_string())?
    .ok_or("meeting has no recording folder")?;
    let path = PathBuf::from(path);
    if !path.is_dir() {
        return Err("recording folder not found on disk".to_string());
    }
    Ok(path)
}

#[tauri::command]
pub async fn speakers_overview<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
) -> Result<SpeakersOverview, String> {
    let folder = meeting_folder(&app, &meeting_id).await?;
    let ms = load_meeting_speakers(&folder);
    let lib = load_library(&app);

    let mut clusters: Vec<ClusterOverview> = ms
        .centroids
        .iter()
        .map(|c| {
            let label = speaker_label(c.cluster);
            let (display, source) = if let Some(n) = ms.names.get(&c.cluster) {
                (n.clone(), "manual")
            } else if let Some(n) = ms.auto.get(&c.cluster) {
                (n.clone(), "auto")
            } else {
                (label.clone(), "none")
            };
            let mut candidates: Vec<(f32, String)> = lib
                .voices
                .iter()
                .filter(|v| v.model == ms.model)
                .map(|v| (cosine(&c.embedding, &v.embedding), v.name.clone()))
                .collect();
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            ClusterOverview {
                cluster: c.cluster,
                label,
                display,
                source: source.to_string(),
                suggestion: ms.suggestions.get(&c.cluster).cloned(),
                candidates: candidates
                    .into_iter()
                    .take(5)
                    .map(|(s, name)| CandidateMatch { name, strength: strength_word(s).to_string() })
                    .collect(),
            }
        })
        .collect();
    clusters.sort_by_key(|c| c.cluster);

    let mut known: Vec<String> = lib.voices.iter().map(|v| v.name.clone()).collect();
    known.sort();
    known.dedup();

    Ok(SpeakersOverview { clusters, known_voices: known })
}

/// Rename a cluster in one meeting; optionally remember the voice in the
/// library. Returns how many transcript rows changed.
#[tauri::command]
pub async fn speaker_rename<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    cluster: usize,
    name: String,
    remember: bool,
) -> Result<u64, String> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("name is empty".to_string());
    }
    if cluster == LOCAL_SPEAKER {
        return Err("the local speaker is identified by the mic channel".to_string());
    }
    let folder = meeting_folder(&app, &meeting_id).await?;
    let mut ms = load_meeting_speakers(&folder);

    let old_display = ms
        .names
        .get(&cluster)
        .or_else(|| ms.auto.get(&cluster))
        .cloned()
        .unwrap_or_else(|| speaker_label(cluster));

    let state = app
        .try_state::<crate::state::AppState>()
        .ok_or("app state unavailable")?;
    let pool = state.db_manager.pool();
    let changed = crate::database::repositories::transcript::TranscriptsRepository::rename_speaker_rows(
        pool, &meeting_id, &old_display, &name,
    )
    .await
    .map_err(|e| e.to_string())?;

    ms.names.insert(cluster, name.clone());
    ms.auto.remove(&cluster);
    ms.suggestions.remove(&cluster);
    save_meeting_speakers(&folder, &ms).map_err(|e| e.to_string())?;

    if remember {
        if let Some(c) = ms.centroids.iter().find(|c| c.cluster == cluster) {
            if c.clean_secs >= MIN_CLEAN_SECS {
                let mut lib = load_library(&app);
                upsert_voice(&mut lib, &name, &ms.model, c);
                save_library(&app, &lib).map_err(|e| e.to_string())?;
                info!(
                    "Voiceprint saved: {} ({}, {:.0}s clean speech)",
                    name, ms.model, c.clean_secs
                );
            } else {
                warn!(
                    "Voiceprint NOT saved for {}: only {:.0}s clean speech (< {:.0}s)",
                    name, c.clean_secs, MIN_CLEAN_SECS
                );
            }
        }
    }

    info!(
        "Speaker renamed in {}: cluster {} '{}' -> '{}' ({} rows)",
        meeting_id, cluster, old_display, name, changed
    );
    Ok(changed)
}

/// "Not X": drop a name (auto or manual) from a cluster without touching
/// the library print. Rows revert to the generated label.
#[tauri::command]
pub async fn speaker_clear_name<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    cluster: usize,
) -> Result<u64, String> {
    let folder = meeting_folder(&app, &meeting_id).await?;
    let mut ms = load_meeting_speakers(&folder);
    let Some(old_display) = ms
        .names
        .get(&cluster)
        .or_else(|| ms.auto.get(&cluster))
        .cloned()
    else {
        return Ok(0);
    };

    let state = app
        .try_state::<crate::state::AppState>()
        .ok_or("app state unavailable")?;
    let pool = state.db_manager.pool();
    let label = speaker_label(cluster);
    let changed = crate::database::repositories::transcript::TranscriptsRepository::rename_speaker_rows(
        pool, &meeting_id, &old_display, &label,
    )
    .await
    .map_err(|e| e.to_string())?;

    ms.names.remove(&cluster);
    ms.auto.remove(&cluster);
    save_meeting_speakers(&folder, &ms).map_err(|e| e.to_string())?;
    info!("Speaker name cleared in {}: cluster {} was '{}'", meeting_id, cluster, old_display);
    Ok(changed)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceInfo {
    pub name: String,
    pub model: String,
    pub clean_secs: f32,
    pub meetings: u32,
    pub updated_at: String,
}

#[tauri::command]
pub async fn voices_list<R: Runtime>(app: AppHandle<R>) -> Result<Vec<VoiceInfo>, String> {
    let lib = load_library(&app);
    Ok(lib
        .voices
        .iter()
        .map(|v| VoiceInfo {
            name: v.name.clone(),
            model: v.model.clone(),
            clean_secs: v.clean_secs,
            meetings: v.meetings,
            updated_at: v.updated_at.clone(),
        })
        .collect())
}

#[tauri::command]
pub async fn local_name_get<R: Runtime>(app: AppHandle<R>) -> Result<Option<String>, String> {
    Ok(load_library(&app).local_name)
}

/// Set (or clear, with an empty string) the local speaker's display name.
/// Applies to rows the next time a meeting is enhanced or fixed up.
#[tauri::command]
pub async fn local_name_set<R: Runtime>(app: AppHandle<R>, name: String) -> Result<(), String> {
    let mut lib = load_library(&app);
    let trimmed = name.trim().to_string();
    lib.local_name = (!trimmed.is_empty() && trimmed != "You").then_some(trimmed.clone());
    save_library(&app, &lib).map_err(|e| e.to_string())?;
    info!(
        "Local speaker display name {}",
        if lib.local_name.is_some() { format!("set to '{}'", trimmed) } else { "cleared (shows as You)".to_string() }
    );
    Ok(())
}

#[tauri::command]
pub async fn voice_delete<R: Runtime>(
    app: AppHandle<R>,
    name: String,
    model: String,
) -> Result<(), String> {
    let mut lib = load_library(&app);
    let before = lib.voices.len();
    lib.voices
        .retain(|v| !(v.model == model && v.name.eq_ignore_ascii_case(&name)));
    if lib.voices.len() == before {
        return Err("no such voice".to_string());
    }
    save_library(&app, &lib).map_err(|e| e.to_string())?;
    info!("Voiceprint deleted: {} ({})", name, model);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[hot] = 1.0;
        v
    }

    fn cent(cluster: usize, secs: f32, emb: Vec<f32>) -> ClusterCentroid {
        ClusterCentroid { cluster, clean_secs: secs, embedding: emb }
    }

    #[test]
    fn upsert_creates_then_averages() {
        let mut lib = VoiceLibrary::default();
        upsert_voice(&mut lib, "Misha", "campplus", &cent(0, 100.0, unit(4, 0)));
        assert_eq!(lib.voices.len(), 1);
        assert_eq!(lib.voices[0].meetings, 1);

        // Second meeting, orthogonal direction — result normalized between
        upsert_voice(&mut lib, "misha", "campplus", &cent(1, 100.0, unit(4, 1)));
        assert_eq!(lib.voices.len(), 1, "case-insensitive same person");
        let v = &lib.voices[0];
        assert_eq!(v.meetings, 2);
        let norm: f32 = v.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);

        // Different model never merges
        upsert_voice(&mut lib, "Misha", "eres2net", &cent(0, 50.0, unit(4, 2)));
        assert_eq!(lib.voices.len(), 2);
    }

    #[test]
    fn name_map_prefers_confirmed_over_auto() {
        let dir = tempfile::tempdir().unwrap();
        let ms = MeetingSpeakers {
            model: "campplus".into(),
            centroids: vec![],
            names: HashMap::from([(0, "Misha".to_string())]),
            auto: HashMap::from([(0, "Wrong".to_string()), (1, "Anton".to_string())]),
            suggestions: HashMap::new(),
        };
        save_meeting_speakers(dir.path(), &ms).unwrap();
        let map = meeting_name_map(dir.path());
        assert_eq!(map.get(&0).unwrap(), "Misha");
        assert_eq!(map.get(&1).unwrap(), "Anton");
        assert_eq!(display_name(2, &map), "Speaker 3");
        assert_eq!(display_name(LOCAL_SPEAKER, &map), "You");
    }
}
