// CoreML encoder auto-provisioning (macOS).
//
// whisper.cpp uses the Apple Neural Engine for the encoder when a compiled
// CoreML model sits next to the ggml file: <stem-minus-quant>-encoder.mlmodelc.
// Benchmarked on an M1 base with large-v3-q5_0: 46.5s -> 39.0s on a 3-minute
// meeting (16% faster), identical output. This module fetches the prebuilt
// encoder from ggerganov/whisper.cpp on Hugging Face in the background the
// first time a model is loaded without one; the encoder is picked up on the
// NEXT model load. macOS compiles the encoder for the ANE once on first use
// (minutes for large models) and caches it system-wide.

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;

/// Base model names for which ggerganov/whisper.cpp publishes a prebuilt
/// CoreML encoder. Quantization suffixes are stripped before matching —
/// the encoder is quantization-independent.
const KNOWN_ENCODERS: &[&str] = &[
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo",
];

/// "ggml-large-v3-q5_0" -> "ggml-large-v3" (whisper.cpp strips the
/// quantization suffix the same way when deriving the encoder path).
fn strip_quant_suffix(stem: &str) -> &str {
    if let Some(pos) = stem.rfind("-q") {
        let tail = &stem[pos + 2..];
        if !tail.is_empty()
            && tail.chars().all(|c| c.is_ascii_digit() || c == '_')
        {
            return &stem[..pos];
        }
    }
    stem
}

/// The encoder directory whisper.cpp will look for next to this model, and
/// the Hugging Face zip name for it — None when no prebuilt encoder exists.
pub fn encoder_paths_for(model_file: &Path) -> Option<(PathBuf, String)> {
    let stem = model_file.file_stem()?.to_str()?;
    let base_stem = strip_quant_suffix(stem);
    let base_model = base_stem.strip_prefix("ggml-")?;
    if !KNOWN_ENCODERS.contains(&base_model) {
        return None;
    }
    let dir = model_file.with_file_name(format!("{}-encoder.mlmodelc", base_stem));
    let zip_name = format!("{}-encoder.mlmodelc.zip", base_stem);
    Some((dir, zip_name))
}

// One attempt per model per app run; failures retry on next launch.
static ATTEMPTED: Lazy<Mutex<HashSet<PathBuf>>> = Lazy::new(|| Mutex::new(HashSet::new()));

/// Fire-and-forget: fetch the prebuilt CoreML encoder for `model_file` if
/// it is missing. Never blocks or fails the caller.
pub fn ensure_encoder_in_background(model_file: PathBuf) {
    let Some((encoder_dir, zip_name)) = encoder_paths_for(&model_file) else {
        return;
    };
    if encoder_dir.exists() {
        return;
    }
    {
        let mut attempted = ATTEMPTED.lock().unwrap_or_else(|e| e.into_inner());
        if !attempted.insert(encoder_dir.clone()) {
            return;
        }
    }
    tauri::async_runtime::spawn(async move {
        log::info!(
            "CoreML encoder missing for {} — downloading {} in background",
            model_file.display(),
            zip_name
        );
        match download_and_unpack(&encoder_dir, &zip_name, &|_| {}).await {
            Ok(()) => log::info!(
                "CoreML encoder installed at {} — used from the next model load (first use compiles for the Neural Engine, which can take minutes)",
                encoder_dir.display()
            ),
            Err(e) => log::warn!("CoreML encoder download failed ({}): {}", zip_name, e),
        }
    });
}

/// Download the encoder for `model_file` with progress (0–100). Returns
/// Ok(false) when no prebuilt encoder applies or it is already installed.
pub async fn download_encoder(
    model_file: &Path,
    progress: &(dyn Fn(u8) + Send + Sync),
) -> Result<bool> {
    let Some((encoder_dir, zip_name)) = encoder_paths_for(model_file) else {
        return Ok(false);
    };
    if encoder_dir.exists() {
        return Ok(false);
    }
    download_and_unpack(&encoder_dir, &zip_name, progress).await?;
    Ok(true)
}

async fn download_and_unpack(
    encoder_dir: &Path,
    zip_name: &str,
    progress: &(dyn Fn(u8) + Send + Sync),
) -> Result<()> {
    use futures_util::StreamExt;

    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        zip_name
    );
    let parent = encoder_dir
        .parent()
        .ok_or_else(|| anyhow!("encoder dir has no parent"))?
        .to_path_buf();
    let zip_path = parent.join(format!("{}.part", zip_name));

    let response = reqwest::get(&url).await?;
    if !response.status().is_success() {
        return Err(anyhow!("HTTP {} for {}", response.status(), url));
    }
    let total = response.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&zip_path).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            progress(((downloaded as f64 / total as f64) * 100.0) as u8);
        }
    }
    drop(file);

    // ditto ships with macOS and preserves the .mlmodelc bundle structure
    let status = tokio::process::Command::new("ditto")
        .arg("-x")
        .arg("-k")
        .arg(&zip_path)
        .arg(&parent)
        .status()
        .await?;
    let _ = tokio::fs::remove_file(&zip_path).await;
    if !status.success() {
        return Err(anyhow!("ditto extraction failed with {}", status));
    }
    if !encoder_dir.exists() {
        return Err(anyhow!(
            "archive did not contain {}",
            encoder_dir.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_quantization_suffixes() {
        assert_eq!(strip_quant_suffix("ggml-large-v3-q5_0"), "ggml-large-v3");
        assert_eq!(strip_quant_suffix("ggml-medium-q8_0"), "ggml-medium");
        assert_eq!(strip_quant_suffix("ggml-large-v3-turbo-q5_0"), "ggml-large-v3-turbo");
        assert_eq!(strip_quant_suffix("ggml-large-v3"), "ggml-large-v3");
        assert_eq!(strip_quant_suffix("ggml-base.en"), "ggml-base.en");
    }

    #[test]
    fn encoder_paths_derived_for_known_models() {
        let (dir, zip) =
            encoder_paths_for(Path::new("/models/ggml-large-v3-q5_0.bin")).expect("known");
        assert_eq!(dir, Path::new("/models/ggml-large-v3-encoder.mlmodelc"));
        assert_eq!(zip, "ggml-large-v3-encoder.mlmodelc.zip");

        let (dir, _) = encoder_paths_for(Path::new("/m/ggml-medium.bin")).expect("known");
        assert_eq!(dir, Path::new("/m/ggml-medium-encoder.mlmodelc"));
    }

    #[test]
    fn unknown_models_are_skipped() {
        assert!(encoder_paths_for(Path::new("/m/ggml-distil-large-v3.bin")).is_none());
        assert!(encoder_paths_for(Path::new("/m/notggml-large-v3.bin")).is_none());
    }
}
