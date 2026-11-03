// Tauri commands for FluidAudio (Parakeet on Apple Neural Engine) model management.

use super::{
    is_available, models_dir, FluidAudioEngine, FluidLoadProgress, FluidLoadStage, FluidModel,
    FLUID_ENGINE,
};
use log::{error, info, warn};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex as StdMutex;
use tauri::{AppHandle, Emitter, Runtime};

/// Model availability status serialised identically to Parakeet/Whisper `ModelStatus`
/// so the frontend can reuse the same status contract:
/// `"Available"`, `"Missing"`, `{"Downloading":{"progress":<0-100>}}`, `{"Error":"..."}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FluidModelAvailability {
    Available,
    Missing,
    Downloading { progress: u8 },
    Error(String),
}

/// Metadata and availability for a single FluidAudio Parakeet model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FluidModelInfo {
    pub name: String,
    pub display_name: String,
    pub size_mb: u32,
    pub description: String,
    pub status: FluidModelAvailability,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadProgressPayload {
    model_name: String,
    stage: FluidLoadStage,
    progress: u8,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadCompletePayload {
    model_name: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadErrorPayload {
    model_name: String,
    error: String,
}

/// Per-model operation state tracked under a single mutex so download registration
/// and model deletion are mutually exclusive without TOCTOU gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelOperationState {
    Downloading(u8),
    Deleting,
}

/// Tracks in-progress model downloads (`Downloading(0..=100)`) and deletions (`Deleting`).
static IN_PROGRESS_DOWNLOADS: Lazy<StdMutex<HashMap<String, ModelOperationState>>> =
    Lazy::new(|| StdMutex::new(HashMap::new()));

/// RAII guard that registers an in-progress download and removes the entry on drop
/// so every exit path cleans up.
struct InProgressDownloadGuard {
    model_id: String,
}

impl InProgressDownloadGuard {
    fn try_acquire(model_id: &str) -> Result<Self, String> {
        let mut map = IN_PROGRESS_DOWNLOADS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.get(model_id) {
            Some(ModelOperationState::Downloading(_)) => {
                return Err(format!("Model '{}' is already downloading", model_id));
            }
            Some(ModelOperationState::Deleting) => {
                return Err(format!(
                    "Cannot download '{}' while it is being deleted",
                    model_id
                ));
            }
            None => {}
        }
        map.insert(model_id.to_string(), ModelOperationState::Downloading(0));
        Ok(Self {
            model_id: model_id.to_string(),
        })
    }

    fn update_progress(&self, progress: u8) {
        let mut map = IN_PROGRESS_DOWNLOADS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(ModelOperationState::Downloading(entry)) = map.get_mut(&self.model_id) {
            *entry = progress.min(100);
        }
    }
}

impl Drop for InProgressDownloadGuard {
    fn drop(&mut self) {
        let mut map = IN_PROGRESS_DOWNLOADS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.remove(&self.model_id);
    }
}

/// RAII guard that marks a model as being deleted in `IN_PROGRESS_DOWNLOADS`
/// (refusing if a download or delete is already active) and clears the marker on drop.
struct InProgressDeleteGuard {
    model_id: String,
}

impl InProgressDeleteGuard {
    fn try_acquire(model_id: &str) -> Result<Self, String> {
        let mut map = IN_PROGRESS_DOWNLOADS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match map.get(model_id) {
            Some(ModelOperationState::Downloading(_)) => {
                return Err(format!(
                    "Cannot delete '{}' while it is downloading",
                    model_id
                ));
            }
            Some(ModelOperationState::Deleting) => {
                return Err(format!("Model '{}' is already being deleted", model_id));
            }
            None => {}
        }
        map.insert(model_id.to_string(), ModelOperationState::Deleting);
        Ok(Self {
            model_id: model_id.to_string(),
        })
    }
}

impl Drop for InProgressDeleteGuard {
    fn drop(&mut self) {
        let mut map = IN_PROGRESS_DOWNLOADS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        map.remove(&self.model_id);
    }
}

fn get_download_progress(model_id: &str) -> Option<u8> {
    let map = IN_PROGRESS_DOWNLOADS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    match map.get(model_id) {
        Some(ModelOperationState::Downloading(progress)) => Some(*progress),
        Some(ModelOperationState::Deleting) | None => None,
    }
}

pub(crate) fn parse_fluid_model(model_name: &str) -> Result<FluidModel, String> {
    match model_name {
        "fluid-parakeet-v3" => Ok(FluidModel::V3),
        "fluid-parakeet-ultra" => Ok(FluidModel::Ultra),
        _ => Err(format!("Unknown FluidAudio model: '{}'", model_name)),
    }
}

fn catalog_entry(model: FluidModel, status: FluidModelAvailability) -> FluidModelInfo {
    match model {
        FluidModel::V3 => FluidModelInfo {
            name: FluidModel::V3.model_id().to_string(),
            display_name: FluidModel::V3.display_name().to_string(),
            size_mb: 470,
            description: "Multilingual (25 languages), keeps English terms in Latin script in mixed-language speech. Recommended.".to_string(),
            status,
        },
        FluidModel::Ultra => FluidModelInfo {
            name: FluidModel::Ultra.model_id().to_string(),
            display_name: FluidModel::Ultra.display_name().to_string(),
            size_mb: 600,
            description: "Russian-focused; best for Russian-only meetings, transliterates English terms into Cyrillic.".to_string(),
            status,
        },
    }
}

/// Canonicalize `base_dir` and `candidate_path` and verify that `candidate_path` is a strict
/// child directory inside `base_dir` (rejecting `..`, symlinks pointing outside, and `base_dir` itself).
pub(crate) fn verify_model_path_within_dir(
    base_dir: &Path,
    candidate_path: &Path,
) -> Result<PathBuf, String> {
    if candidate_path.as_os_str().is_empty() {
        return Err("Model path is empty".to_string());
    }

    for comp in candidate_path.components() {
        if matches!(comp, Component::ParentDir) {
            return Err(format!(
                "Refusing to delete path containing '..': {}",
                candidate_path.display()
            ));
        }
    }

    let canonical_base = std::fs::canonicalize(base_dir).map_err(|e| {
        format!(
            "Failed to canonicalize FluidAudio models directory '{}': {}",
            base_dir.display(),
            e
        )
    })?;

    let canonical_target = if candidate_path.exists() {
        std::fs::canonicalize(candidate_path).map_err(|e| {
            format!(
                "Failed to canonicalize model path '{}': {}",
                candidate_path.display(),
                e
            )
        })?
    } else {
        // Resolve the longest existing ancestor and append remaining normal components.
        let mut existing = candidate_path;
        let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
        while !existing.exists() {
            let Some(name) = existing.file_name() else {
                return Err(format!(
                    "Cannot resolve non-existent model path '{}'",
                    candidate_path.display()
                ));
            };
            tail.push(name);
            let Some(parent) = existing.parent() else {
                return Err(format!(
                    "Cannot resolve parent of model path '{}'",
                    candidate_path.display()
                ));
            };
            if parent.as_os_str().is_empty() {
                return Err(format!(
                    "Model path '{}' is not an absolute path inside '{}'",
                    candidate_path.display(),
                    base_dir.display()
                ));
            }
            existing = parent;
        }
        let mut resolved = std::fs::canonicalize(existing).map_err(|e| {
            format!(
                "Failed to canonicalize ancestor '{}' of '{}': {}",
                existing.display(),
                candidate_path.display(),
                e
            )
        })?;
        for segment in tail.into_iter().rev() {
            resolved.push(segment);
        }
        resolved
    };

    if !canonical_target.starts_with(&canonical_base) || canonical_target == canonical_base {
        return Err(format!(
            "Refusing to delete path '{}' outside FluidAudio models directory '{}'",
            canonical_target.display(),
            canonical_base.display()
        ));
    }

    Ok(canonical_target)
}

/// Check whether FluidAudio transcription is available on this system.
#[tauri::command]
pub fn fluid_is_available() -> bool {
    is_available()
}

/// List all supported FluidAudio models and their current availability status.
/// Returns an empty list (not an error) when FluidAudio is not available on this platform.
#[tauri::command]
pub async fn fluid_get_available_models<R: Runtime>(
    app: AppHandle<R>,
) -> Result<Vec<FluidModelInfo>, String> {
    if !is_available() {
        return Ok(Vec::new());
    }

    let dir = models_dir(&app).map_err(|e| e.to_string())?;
    let variants = [FluidModel::V3, FluidModel::Ultra];
    let mut results = Vec::with_capacity(variants.len());

    for model in variants {
        let model_id = model.model_id();
        let status = if let Some(progress) = get_download_progress(model_id) {
            FluidModelAvailability::Downloading { progress }
        } else {
            match FLUID_ENGINE.model_status(model, &dir).await {
                Ok(st) if st.downloaded => FluidModelAvailability::Available,
                Ok(_) => FluidModelAvailability::Missing,
                Err(e) => {
                    warn!(
                        "Failed to query status for FluidAudio model '{}': {}",
                        model_id, e
                    );
                    FluidModelAvailability::Error(e.to_string())
                }
            }
        };

        results.push(catalog_entry(model, status));
    }

    Ok(results)
}

/// Download and compile a FluidAudio model for the Apple Neural Engine.
/// Emits `fluid-model-download-progress`, then `fluid-model-download-complete` or `fluid-model-download-error`.
#[tauri::command]
pub async fn fluid_download_model<R: Runtime>(
    app: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let model = parse_fluid_model(&model_name)?;
    let model_id = model.model_id().to_string();

    let download_guard = InProgressDownloadGuard::try_acquire(&model_id)?;

    if !is_available() {
        return Err("FluidAudio transcription is not available on this system".to_string());
    }

    let dir = models_dir(&app).map_err(|e| e.to_string())?;

    info!("Starting FluidAudio model download & compile for '{}'", model_id);

    let last_emitted: StdMutex<Option<(FluidLoadStage, u8)>> = StdMutex::new(None);
    let app_for_progress = app.clone();
    let model_id_for_progress = model_id.clone();

    let job_result = FLUID_ENGINE
        .begin_job(model, &dir, |progress: FluidLoadProgress| {
            let pct = (progress.fraction.clamp(0.0, 1.0) * 100.0).round() as u8;
            let pct = pct.min(100);

            let should_emit = {
                let mut last = last_emitted.lock().unwrap_or_else(|e| e.into_inner());
                if *last != Some((progress.stage, pct)) {
                    *last = Some((progress.stage, pct));
                    true
                } else {
                    false
                }
            };

            if should_emit {
                download_guard.update_progress(pct);
                let _ = app_for_progress.emit(
                    "fluid-model-download-progress",
                    DownloadProgressPayload {
                        model_name: model_id_for_progress.clone(),
                        stage: progress.stage,
                        progress: pct,
                    },
                );
            }
        })
        .await;

    match job_result {
        Ok(job_guard) => {
            drop(job_guard);
            FLUID_ENGINE.unload_if_idle().await;
            drop(download_guard);

            info!("Completed FluidAudio model download & compile for '{}'", model_id);
            let _ = app.emit(
                "fluid-model-download-complete",
                DownloadCompletePayload {
                    model_name: model_id,
                },
            );
            Ok(())
        }
        Err(e) => {
            let err_msg = e.to_string();
            error!(
                "Failed FluidAudio model download/compile for '{}': {}",
                model_id, err_msg
            );
            FLUID_ENGINE.unload_if_idle().await;
            drop(download_guard);

            let _ = app.emit(
                "fluid-model-download-error",
                DownloadErrorPayload {
                    model_name: model_id,
                    error: err_msg.clone(),
                },
            );
            Err(err_msg)
        }
    }
}

async fn delete_model_with_engine(
    engine: &FluidAudioEngine,
    model: FluidModel,
    dir: &Path,
) -> Result<(), String> {
    let model_id = model.model_id();
    let _delete_guard = InProgressDeleteGuard::try_acquire(model_id)?;

    let verify_dir = dir.to_path_buf();
    engine
        .delete_model_files(model, dir, move |candidate_path| {
            verify_model_path_within_dir(&verify_dir, candidate_path)
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!(e))
        })
        .await
        .map_err(|e| e.to_string())
}

/// Delete a downloaded FluidAudio model from disk.
#[tauri::command]
pub async fn fluid_delete_model<R: Runtime>(
    app: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let model = parse_fluid_model(&model_name)?;
    let dir = models_dir(&app).map_err(|e| e.to_string())?;
    delete_model_with_engine(&FLUID_ENGINE, model, &dir).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    static TEST_MAP_LOCK: Lazy<StdMutex<()>> = Lazy::new(|| StdMutex::new(()));

    #[test]
    fn status_serialization_matches_frontend_contract() {
        assert_eq!(
            serde_json::to_value(FluidModelAvailability::Available).unwrap(),
            serde_json::json!("Available")
        );
        assert_eq!(
            serde_json::to_value(FluidModelAvailability::Missing).unwrap(),
            serde_json::json!("Missing")
        );
        assert_eq!(
            serde_json::to_value(FluidModelAvailability::Downloading { progress: 42 }).unwrap(),
            serde_json::json!({"Downloading": {"progress": 42}})
        );
        assert_eq!(
            serde_json::to_value(FluidModelAvailability::Error("network error".to_string()))
                .unwrap(),
            serde_json::json!({"Error": "network error"})
        );
    }

    #[test]
    fn rejects_unknown_model_ids() {
        assert_eq!(
            parse_fluid_model("fluid-parakeet-v3").unwrap(),
            FluidModel::V3
        );
        assert_eq!(
            parse_fluid_model("fluid-parakeet-ultra").unwrap(),
            FluidModel::Ultra
        );

        assert!(parse_fluid_model("").is_err());
        assert!(parse_fluid_model("v3").is_err());
        assert!(parse_fluid_model("ultra").is_err());
        assert!(parse_fluid_model("parakeet-tdt-0.6b-v3-int8").is_err());
        assert!(parse_fluid_model("whisper-large-v3").is_err());
    }

    #[test]
    fn path_prefix_check_accepts_child_and_rejects_outside_or_parent_traversal() {
        let root = std::env::temp_dir().join(format!("meetily-fluid-cmd-test-{}", Uuid::new_v4()));
        let base_dir = root.join("models").join("fluidaudio");
        let child_dir = base_dir.join("parakeet-tdt-0.6b-v3");
        let outside_dir = root.join("outside");

        std::fs::create_dir_all(&child_dir).unwrap();
        std::fs::create_dir_all(&outside_dir).unwrap();

        // 1. Existing child directory inside base_dir is accepted
        let verified_existing = verify_model_path_within_dir(&base_dir, &child_dir).unwrap();
        assert_eq!(
            verified_existing,
            std::fs::canonicalize(&child_dir).unwrap()
        );

        // 2. Non-existent child directory inside base_dir is accepted
        let non_existent_child = base_dir.join("parakeet-ultra");
        let verified_missing =
            verify_model_path_within_dir(&base_dir, &non_existent_child).unwrap();
        assert_eq!(
            verified_missing,
            std::fs::canonicalize(&base_dir)
                .unwrap()
                .join("parakeet-ultra")
        );

        // 3. Base directory itself is rejected
        assert!(verify_model_path_within_dir(&base_dir, &base_dir).is_err());

        // 4. Path containing `..` traversal is rejected
        let traversal = base_dir.join("..").join("outside");
        assert!(verify_model_path_within_dir(&base_dir, &traversal).is_err());

        let internal_traversal = base_dir.join("parakeet-tdt-0.6b-v3").join("..");
        assert!(verify_model_path_within_dir(&base_dir, &internal_traversal).is_err());

        // 5. Path outside base_dir is rejected
        assert!(verify_model_path_within_dir(&base_dir, &outside_dir).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn delete_refuses_when_model_is_in_use_and_clears_delete_marker() {
        let _lock = TEST_MAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let engine = FluidAudioEngine::new();
        {
            let mut jobs = engine.inner.jobs.lock().unwrap();
            jobs.begin(FluidModel::V3).unwrap();
        }

        let err = delete_model_with_engine(&engine, FluidModel::V3, Path::new("/tmp"))
            .await
            .unwrap_err();
        assert!(
            err.contains("Cannot delete 'fluid-parakeet-v3' while it is in use"),
            "Unexpected error: {}",
            err
        );

        // The delete guard must have cleaned up on exit so downloads are not blocked afterward
        let dl_guard = InProgressDownloadGuard::try_acquire("fluid-parakeet-v3")
            .expect("download guard should succeed after failed delete");
        drop(dl_guard);
    }

    #[tokio::test]
    async fn delete_refuses_when_model_is_currently_downloading() {
        let _lock = TEST_MAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let engine = FluidAudioEngine::new();

        let dl_guard = InProgressDownloadGuard::try_acquire("fluid-parakeet-v3").unwrap();
        dl_guard.update_progress(35);
        assert_eq!(get_download_progress("fluid-parakeet-v3"), Some(35));

        let err = delete_model_with_engine(&engine, FluidModel::V3, Path::new("/tmp"))
            .await
            .unwrap_err();
        assert_eq!(
            err,
            "Cannot delete 'fluid-parakeet-v3' while it is downloading"
        );

        // Download progress remains intact while dl_guard is alive
        assert_eq!(get_download_progress("fluid-parakeet-v3"), Some(35));

        drop(dl_guard);
        assert_eq!(get_download_progress("fluid-parakeet-v3"), None);
    }

    #[test]
    fn download_refuses_when_model_is_being_deleted() {
        let _lock = TEST_MAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let del_guard = InProgressDeleteGuard::try_acquire("fluid-parakeet-v3").unwrap();
        // While deleting, get_download_progress must not report the model as downloading
        assert_eq!(get_download_progress("fluid-parakeet-v3"), None);

        let dl_err = match InProgressDownloadGuard::try_acquire("fluid-parakeet-v3") {
            Ok(_) => panic!("Expected download registration to be refused while deleting"),
            Err(e) => e,
        };
        assert_eq!(
            dl_err,
            "Cannot download 'fluid-parakeet-v3' while it is being deleted"
        );

        // Concurrent delete for the same model is also refused
        let del_err = match InProgressDeleteGuard::try_acquire("fluid-parakeet-v3") {
            Ok(_) => panic!("Expected second delete registration to be refused"),
            Err(e) => e,
        };
        assert_eq!(
            del_err,
            "Model 'fluid-parakeet-v3' is already being deleted"
        );

        drop(del_guard);

        // After delete guard drops, download registration succeeds
        let dl_guard = InProgressDownloadGuard::try_acquire("fluid-parakeet-v3")
            .expect("download should be allowed after delete guard drops");
        drop(dl_guard);
    }
}
