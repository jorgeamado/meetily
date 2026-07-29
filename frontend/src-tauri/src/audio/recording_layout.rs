//! Keeps the on-disk recording folders in sync with in-app organization.
//!
//! The sidebar's folders and meeting titles used to live only in the
//! database, so the audio in the recordings directory stayed flat under
//! stale names. Renaming a meeting now renames its recording folder,
//! moving a meeting into a sidebar folder moves the recording folder into
//! a matching subdirectory of the recordings root, and folder rename /
//! delete follow suit. Disk changes run first; the database update only
//! happens after the files actually moved.

use std::path::{Path, PathBuf};

use log::{info, warn};
use sqlx::SqlitePool;
use tauri::{AppHandle, Runtime};

use crate::audio::audio_processing::sanitize_filename;
use crate::audio::recording_preferences::load_recording_preferences;
use crate::audio::retranscription::is_retranscription_in_progress;
use crate::database::repositories::meeting::MeetingsRepository;

/// Refuse disk reorganization while audio is being read/written.
fn ensure_idle() -> Result<(), String> {
    if is_retranscription_in_progress() {
        return Err("A transcription is running — try again when it finishes".to_string());
    }
    Ok(())
}

fn sanitized_or_default(title: &str) -> String {
    let s = sanitize_filename(title);
    if s.is_empty() {
        "Meeting".to_string()
    } else {
        s
    }
}

/// First free path for `name` under `parent` ("name", "name (2)", ...).
fn unique_child(parent: &Path, name: &str) -> Result<PathBuf, String> {
    let direct = parent.join(name);
    if !direct.exists() {
        return Ok(direct);
    }
    for i in 2..=20 {
        let candidate = parent.join(format!("{} ({})", name, i));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(format!("Too many folders named '{}' in {}", name, parent.display()))
}

/// Recordings root from preferences, with the meeting's own path as a
/// sanity fallback for meetings recorded before the preference existed.
async fn recordings_root<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    match load_recording_preferences(app).await {
        Ok(p) => Some(p.save_folder),
        Err(e) => {
            warn!("recording_layout: cannot load preferences: {}", e);
            None
        }
    }
}

/// The trailing "YYYY-MM-DD_HH-MM" stamp of a recording folder name
/// ("<title>_<stamp>", as produced by create_meeting_folder), if present.
fn timestamp_suffix(dir_name: &str) -> Option<&str> {
    if dir_name.len() < 17 || !dir_name.is_char_boundary(dir_name.len() - 16) {
        return None;
    }
    let (head, tail) = dir_name.split_at(dir_name.len() - 16);
    let shape_ok = head.ends_with('_')
        && tail.chars().enumerate().all(|(i, c)| match i {
            4 | 7 | 13 => c == '-',
            10 => c == '_',
            _ => c.is_ascii_digit(),
        });
    if shape_ok {
        Some(tail)
    } else {
        None
    }
}

async fn update_db_path(
    pool: &SqlitePool,
    meeting_id: &str,
    new_path: &Path,
) -> Result<(), String> {
    MeetingsRepository::update_meeting_folder_path(pool, meeting_id, &new_path.to_string_lossy())
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Best-effort: keep metadata.json's meeting_name matching the app title.
fn update_metadata_name(folder: &Path, new_title: &str) {
    let path = folder.join("metadata.json");
    let Ok(raw) = std::fs::read_to_string(&path) else { return };
    let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&raw) else { return };
    json["meeting_name"] = serde_json::Value::String(new_title.to_string());
    let Ok(out) = serde_json::to_string_pretty(&json) else { return };
    let temp = folder.join("metadata.json.tmp");
    if std::fs::write(&temp, out).is_ok() {
        let _ = std::fs::rename(&temp, &path);
    }
}

/// Remove `dir` if it is an empty, non-root subdirectory of `root`.
fn cleanup_empty_subdir(dir: &Path, root: &Path) {
    if dir != root
        && dir.parent() == Some(root)
        && std::fs::read_dir(dir).map(|mut d| d.next().is_none()).unwrap_or(false)
    {
        let _ = std::fs::remove_dir(dir);
    }
}

/// Rename the recording folder when the meeting title changes.
pub async fn sync_meeting_rename<R: Runtime>(
    _app: &AppHandle<R>,
    pool: &SqlitePool,
    meeting_id: &str,
    new_title: &str,
) -> Result<(), String> {
    let Some(old_path) = MeetingsRepository::get_meeting_folder_path(pool, meeting_id)
        .await
        .map_err(|e| e.to_string())?
        .filter(|p| !p.trim().is_empty())
    else {
        return Ok(());
    };
    let old_dir = PathBuf::from(&old_path);
    if !old_dir.is_dir() {
        return Ok(());
    }
    ensure_idle()?;

    let old_name = old_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let new_base = match timestamp_suffix(&old_name) {
        Some(ts) => format!("{}_{}", sanitized_or_default(new_title), ts),
        None => sanitized_or_default(new_title),
    };
    if new_base == old_name {
        update_metadata_name(&old_dir, new_title);
        return Ok(());
    }
    let parent = old_dir.parent().ok_or("Recording folder has no parent")?;
    let new_dir = unique_child(parent, &new_base)?;
    std::fs::rename(&old_dir, &new_dir)
        .map_err(|e| format!("Could not rename recording folder: {}", e))?;
    info!(
        "recording_layout: renamed {} -> {}",
        old_dir.display(),
        new_dir.display()
    );
    update_metadata_name(&new_dir, new_title);
    update_db_path(pool, meeting_id, &new_dir).await
}

/// Move the recording folder into (or out of) a sidebar folder's directory.
pub async fn sync_meeting_move<R: Runtime>(
    app: &AppHandle<R>,
    pool: &SqlitePool,
    meeting_id: &str,
    target_folder_id: Option<&str>,
) -> Result<(), String> {
    let Some(old_path) = MeetingsRepository::get_meeting_folder_path(pool, meeting_id)
        .await
        .map_err(|e| e.to_string())?
        .filter(|p| !p.trim().is_empty())
    else {
        return Ok(());
    };
    let old_dir = PathBuf::from(&old_path);
    if !old_dir.is_dir() {
        return Ok(());
    }
    let Some(root) = recordings_root(app).await else { return Ok(()) };

    let dest_parent = match target_folder_id {
        Some(fid) => {
            let title = MeetingsRepository::get_folder_title(pool, fid)
                .await
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("No folder found with id {}", fid))?;
            root.join(sanitized_or_default(&title))
        }
        None => root.clone(),
    };
    let old_parent = old_dir.parent().map(Path::to_path_buf);
    if old_parent.as_deref() == Some(dest_parent.as_path()) {
        return Ok(());
    }
    ensure_idle()?;

    std::fs::create_dir_all(&dest_parent)
        .map_err(|e| format!("Could not create folder directory: {}", e))?;
    let name = old_dir
        .file_name()
        .ok_or("Recording folder has no name")?
        .to_string_lossy()
        .to_string();
    let new_dir = unique_child(&dest_parent, &name)?;
    std::fs::rename(&old_dir, &new_dir)
        .map_err(|e| format!("Could not move recording folder: {}", e))?;
    info!(
        "recording_layout: moved {} -> {}",
        old_dir.display(),
        new_dir.display()
    );
    update_db_path(pool, meeting_id, &new_dir).await?;
    if let Some(p) = old_parent {
        cleanup_empty_subdir(&p, &root);
    }
    Ok(())
}

/// Rename a sidebar folder's directory and re-point contained meetings.
pub async fn sync_folder_rename<R: Runtime>(
    app: &AppHandle<R>,
    pool: &SqlitePool,
    folder_id: &str,
    new_title: &str,
) -> Result<(), String> {
    let Some(old_title) = MeetingsRepository::get_folder_title(pool, folder_id)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Ok(());
    };
    let Some(root) = recordings_root(app).await else { return Ok(()) };
    let old_dir = root.join(sanitized_or_default(&old_title));
    let new_base = sanitized_or_default(new_title);
    if !old_dir.is_dir() || old_dir == root.join(&new_base) {
        return Ok(());
    }
    ensure_idle()?;

    let new_dir = unique_child(&root, &new_base)?;
    std::fs::rename(&old_dir, &new_dir)
        .map_err(|e| format!("Could not rename folder directory: {}", e))?;
    info!(
        "recording_layout: renamed {} -> {}",
        old_dir.display(),
        new_dir.display()
    );
    for (meeting_id, path) in MeetingsRepository::get_meeting_paths_in_folder(pool, folder_id)
        .await
        .map_err(|e| e.to_string())?
    {
        let Some(path) = path else { continue };
        if let Ok(rel) = Path::new(&path).strip_prefix(&old_dir) {
            update_db_path(pool, &meeting_id, &new_dir.join(rel)).await?;
        }
    }
    Ok(())
}

/// Move a deleted folder's recordings back to the recordings root.
pub async fn sync_folder_delete<R: Runtime>(
    app: &AppHandle<R>,
    pool: &SqlitePool,
    folder_id: &str,
) -> Result<(), String> {
    let Some(title) = MeetingsRepository::get_folder_title(pool, folder_id)
        .await
        .map_err(|e| e.to_string())?
    else {
        return Ok(());
    };
    let Some(root) = recordings_root(app).await else { return Ok(()) };
    let dir = root.join(sanitized_or_default(&title));
    if !dir.is_dir() {
        return Ok(());
    }
    ensure_idle()?;

    for (meeting_id, path) in MeetingsRepository::get_meeting_paths_in_folder(pool, folder_id)
        .await
        .map_err(|e| e.to_string())?
    {
        let Some(path) = path else { continue };
        let old = PathBuf::from(&path);
        if !old.starts_with(&dir) || !old.is_dir() {
            continue;
        }
        let name = old
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let new_dir = unique_child(&root, &name)?;
        std::fs::rename(&old, &new_dir)
            .map_err(|e| format!("Could not move recording folder: {}", e))?;
        info!(
            "recording_layout: moved {} -> {}",
            old.display(),
            new_dir.display()
        );
        update_db_path(pool, &meeting_id, &new_dir).await?;
    }
    cleanup_empty_subdir(&dir, &root);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_suffix_detected() {
        assert_eq!(
            timestamp_suffix("Meeting 2026-07-27_19-06-36_2026-07-27_17-06"),
            Some("2026-07-27_17-06")
        );
        assert_eq!(timestamp_suffix("My Call_2026-01-02_03-04"), Some("2026-01-02_03-04"));
        assert_eq!(timestamp_suffix("no stamp here"), None);
        assert_eq!(timestamp_suffix("short_1-2"), None);
    }

    #[test]
    fn unique_child_skips_existing() {
        let tmp = std::env::temp_dir().join(format!("layout-test-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("a")).unwrap();
        assert_eq!(unique_child(&tmp, "b").unwrap(), tmp.join("b"));
        assert_eq!(unique_child(&tmp, "a").unwrap(), tmp.join("a (2)"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
