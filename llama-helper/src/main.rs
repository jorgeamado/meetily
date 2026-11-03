use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use encoding_rs;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use serde::{Deserialize, Serialize};

// ============================================================================
// Protocol Messages (JSON over stdin/stdout)
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Request {
    Generate {
        prompt: String,
        max_tokens: Option<i32>,
        context_size: Option<u32>,
        model_path: Option<String>,
        // Sampling parameters
        temperature: Option<f32>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        repeat_penalty: Option<f32>,
        penalty_last_n: Option<i32>,
        stop_tokens: Option<Vec<String>>,
    },
    Ping,
    Shutdown,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Response {
    Response { text: String, error: Option<String> },
    Pong,
    Goodbye,
    Error { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SamplingConfig {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_penalty: f32,
    penalty_last_n: i32,
}

impl SamplingConfig {
    fn from_request(
        temperature: Option<f32>,
        top_k: Option<i32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        repeat_penalty: Option<f32>,
        penalty_last_n: Option<i32>,
    ) -> Self {
        let temperature = temperature.unwrap_or(1.0);
        let temperature = if temperature.is_finite() {
            temperature.max(0.0)
        } else {
            0.0
        };
        let top_k = top_k.unwrap_or(64).max(1);
        let top_p = top_p.unwrap_or(0.95);
        let top_p = if top_p.is_finite() && top_p > 0.0 && top_p <= 1.0 {
            top_p
        } else {
            1.0
        };
        let presence_penalty = presence_penalty.unwrap_or(0.0);
        let presence_penalty = if presence_penalty.is_finite() {
            presence_penalty.max(0.0)
        } else {
            0.0
        };
        let frequency_penalty = frequency_penalty.unwrap_or(0.0);
        let frequency_penalty = if frequency_penalty.is_finite() {
            frequency_penalty.max(0.0)
        } else {
            0.0
        };
        let repeat_penalty = repeat_penalty.unwrap_or(1.0);
        let repeat_penalty = if repeat_penalty.is_finite() && repeat_penalty > 0.0 {
            repeat_penalty
        } else {
            1.0
        };
        let penalty_last_n = penalty_last_n.unwrap_or(0).max(0);

        Self {
            temperature,
            top_k,
            top_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            penalty_last_n,
        }
    }

    fn uses_penalties(&self) -> bool {
        self.penalty_last_n > 0
            && (self.presence_penalty > 0.0
                || self.frequency_penalty > 0.0
                || (self.repeat_penalty - 1.0).abs() > f32::EPSILON)
    }
}

// ============================================================================
// VRAM Detection and GPU Layer Calculation
// ============================================================================

/// Detect available VRAM in GB
fn detect_vram_gb() -> f32 {
    #[cfg(feature = "metal")]
    {
        // macOS Metal: Query recommended max working set size
        if let Some(vram) = detect_metal_vram() {
            eprintln!("Metal VRAM detected: {:.2} GB", vram);
            return vram;
        }
    }

    #[cfg(feature = "cuda")]
    {
        // NVIDIA CUDA: Query device memory
        if let Some(vram) = detect_cuda_vram() {
            eprintln!("CUDA VRAM detected: {:.2} GB", vram);
            return vram;
        }
    }

    /// TODO: Vulkan VRAM detection

    eprintln!("VRAM detection not available, using conservative estimate");
    4.0 // Conservative fallback
}

#[cfg(feature = "metal")]
fn detect_metal_vram() -> Option<f32> {
    if let Ok(output) = std::process::Command::new("sysctl")
        .arg("hw.memsize")
        .output()
    {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            if let Some(bytes_str) = stdout.split(':').nth(1) {
                if let Ok(bytes) = bytes_str.trim().parse::<u64>() {
                    let gb = bytes as f32 / (1024.0 * 1024.0 * 1024.0);
                    // Assume GPU can use ~60% of system memory on Apple Silicon
                    return Some(gb * 0.6);
                }
            }
        }
    }
    None
}

#[cfg(feature = "cuda")]
fn detect_cuda_vram() -> Option<f32> {
    // Use nvidia-smi to query VRAM
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
    {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            if let Ok(mb) = stdout.trim().parse::<f32>() {
                return Some(mb / 1024.0); // Convert MB to GB
            }
        }
    }
    None
}

/// Minimal GGUF model metadata needed for KV cache and layer estimation
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct GgufModelMeta {
    arch: String,
    block_count: u32,
    n_attention_layers: u32,
    embedding_length: u32,
    head_count: u32,
    head_count_kv: u32,
    key_length: u32,
    value_length: u32,
}

const GGUF_MAX_TENSOR_COUNT: u64 = 1_000_000;
const GGUF_MAX_KV_COUNT: u64 = 100_000;
const GGUF_MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const GGUF_MAX_KEY_BYTES: u64 = 64 * 1024;
const GGUF_MAX_STR_BYTES: u64 = 1024 * 1024;
const GGUF_MAX_SKIP_STR_BYTES: u64 = 64 * 1024 * 1024;
const GGUF_MAX_META_ARRAY_LEN: u64 = 4096;
const GGUF_MAX_SKIP_ARRAY_LEN: u64 = 4_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufScalar {
    Int(i128),
    NonInt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GgufMetaValue {
    Int(i128),
    IntArray(Vec<i128>),
    Invalid,
}

fn gguf_scalar_byte_len(val_type: u32) -> Option<u64> {
    match val_type {
        0 | 1 | 7 => Some(1),    // UINT8, INT8, BOOL
        2 | 3 => Some(2),        // UINT16, INT16
        4 | 5 | 6 => Some(4),    // UINT32, INT32, FLOAT32
        10 | 11 | 12 => Some(8), // UINT64, INT64, FLOAT64
        _ => None,
    }
}

struct GgufReader<R> {
    inner: R,
    remaining_budget: u64,
}

impl<R: Read> GgufReader<R> {
    fn new(inner: R, budget: u64) -> Self {
        Self {
            inner,
            remaining_budget: budget,
        }
    }

    fn charge(&mut self, bytes: u64) -> Option<()> {
        self.remaining_budget = self.remaining_budget.checked_sub(bytes)?;
        Some(())
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Option<()> {
        self.charge(buf.len() as u64)?;
        self.inner.read_exact(buf).ok()
    }

    fn read_u32(&mut self) -> Option<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Some(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self) -> Option<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Some(u64::from_le_bytes(b))
    }

    fn read_u64_unmetered(&mut self) -> Option<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b).ok()?;
        Some(u64::from_le_bytes(b))
    }

    fn skip_bytes(&mut self, mut remaining: u64) -> Option<()> {
        self.charge(remaining)?;
        let mut buf = [0u8; 4096];
        while remaining > 0 {
            let to_read = remaining.min(buf.len() as u64) as usize;
            match self.inner.read(&mut buf[..to_read]) {
                Ok(0) => return None,
                Ok(n) => remaining = remaining.checked_sub(n as u64)?,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => return None,
            }
        }
        Some(())
    }

    fn read_string(&mut self, len: u64, max_len: u64) -> Option<String> {
        if len > max_len {
            return None;
        }
        self.charge(len)?;
        let mut buf = Vec::with_capacity(len.min(256) as usize);
        let read_bytes = (&mut self.inner).take(len).read_to_end(&mut buf).ok()?;
        if read_bytes as u64 != len {
            return None;
        }
        String::from_utf8(buf).ok()
    }

    fn read_scalar(&mut self, val_type: u32) -> Option<GgufScalar> {
        match val_type {
            0 => {
                let mut b = [0u8; 1];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int(b[0] as i128))
            }
            1 => {
                let mut b = [0u8; 1];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int((b[0] as i8) as i128))
            }
            2 => {
                let mut b = [0u8; 2];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int(u16::from_le_bytes(b) as i128))
            }
            3 => {
                let mut b = [0u8; 2];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int(i16::from_le_bytes(b) as i128))
            }
            4 => {
                let mut b = [0u8; 4];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int(u32::from_le_bytes(b) as i128))
            }
            5 => {
                let mut b = [0u8; 4];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int(i32::from_le_bytes(b) as i128))
            }
            6 => {
                self.skip_bytes(4)?;
                Some(GgufScalar::NonInt)
            }
            7 => {
                let mut b = [0u8; 1];
                self.read_exact(&mut b)?;
                match b[0] {
                    0 | 1 => Some(GgufScalar::Int(b[0] as i128)),
                    _ => Some(GgufScalar::NonInt),
                }
            }
            10 => {
                let mut b = [0u8; 8];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int(u64::from_le_bytes(b) as i128))
            }
            11 => {
                let mut b = [0u8; 8];
                self.read_exact(&mut b)?;
                Some(GgufScalar::Int(i64::from_le_bytes(b) as i128))
            }
            12 => {
                self.skip_bytes(8)?;
                Some(GgufScalar::NonInt)
            }
            _ => None,
        }
    }
}

fn is_tracked_gguf_key(key: &str) -> bool {
    const SUFFIXES: &[&str] = &[
        ".block_count",
        ".nextn_predict_layers",
        ".embedding_length",
        ".attention.head_count",
        ".attention.head_count_kv",
        ".attention.key_length",
        ".attention.value_length",
        ".attention.recurrent_layers",
        ".full_attention_interval",
    ];
    SUFFIXES.iter().any(|suffix| key.ends_with(suffix))
}

/// Parse the GGUF v2/v3 key/value header section without loading model weights
fn parse_gguf_model_meta<R: Read>(reader: R) -> Option<GgufModelMeta> {
    let mut reader = GgufReader::new(reader, GGUF_MAX_HEADER_BYTES);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return None;
    }

    let version = reader.read_u32()?;
    if version != 2 && version != 3 {
        return None;
    }

    let n_tensors = reader.read_u64()?;
    let n_kv = reader.read_u64()?;
    if n_tensors > GGUF_MAX_TENSOR_COUNT || n_kv > GGUF_MAX_KV_COUNT {
        return None;
    }

    let mut arch: Option<String> = None;
    let mut meta_values: HashMap<String, GgufMetaValue> = HashMap::new();

    for _ in 0..n_kv {
        let key_len = reader.read_u64()?;
        let key = reader.read_string(key_len, GGUF_MAX_KEY_BYTES)?;
        let tracked = is_tracked_gguf_key(&key);

        let val_type = reader.read_u32()?;
        match val_type {
            8 => {
                // GGUF_TYPE_STRING
                let str_len = reader.read_u64()?;
                if key == "general.architecture" {
                    if arch.is_some() || str_len == 0 {
                        return None;
                    }
                    arch = Some(reader.read_string(str_len, GGUF_MAX_STR_BYTES)?);
                } else {
                    if str_len > GGUF_MAX_SKIP_STR_BYTES {
                        return None;
                    }
                    reader.skip_bytes(str_len)?;
                    if tracked && meta_values.insert(key, GgufMetaValue::Invalid).is_some() {
                        return None;
                    }
                }
            }
            9 => {
                // GGUF_TYPE_ARRAY
                let elem_type = reader.read_u32()?;
                let arr_len = reader.read_u64()?;
                if key == "general.architecture" || arr_len > GGUF_MAX_SKIP_ARRAY_LEN {
                    return None;
                }
                if tracked
                    && arr_len <= GGUF_MAX_META_ARRAY_LEN
                    && matches!(elem_type, 0 | 1 | 2 | 3 | 4 | 5 | 7 | 10 | 11)
                {
                    let elem_size = gguf_scalar_byte_len(elem_type)?;
                    if arr_len.checked_mul(elem_size)? > reader.remaining_budget {
                        return None;
                    }
                    let mut vals = Vec::with_capacity(arr_len.min(64) as usize);
                    let mut valid = true;
                    for _ in 0..arr_len {
                        match reader.read_scalar(elem_type)? {
                            GgufScalar::Int(v) => vals.push(v),
                            GgufScalar::NonInt => valid = false,
                        }
                    }
                    let entry = if valid {
                        GgufMetaValue::IntArray(vals)
                    } else {
                        GgufMetaValue::Invalid
                    };
                    if meta_values.insert(key, entry).is_some() {
                        return None;
                    }
                } else {
                    if let Some(elem_size) = gguf_scalar_byte_len(elem_type) {
                        reader.skip_bytes(arr_len.checked_mul(elem_size)?)?;
                    } else if elem_type == 8 {
                        // Pre-charge all 8-byte string length prefixes up front so an array
                        // whose element headers + payloads exceed the budget fails early.
                        reader.charge(arr_len.checked_mul(8)?)?;
                        for _ in 0..arr_len {
                            let slen = reader.read_u64_unmetered()?;
                            if slen > GGUF_MAX_SKIP_STR_BYTES {
                                return None;
                            }
                            reader.skip_bytes(slen)?;
                        }
                    } else {
                        return None;
                    }
                    if tracked && meta_values.insert(key, GgufMetaValue::Invalid).is_some() {
                        return None;
                    }
                }
            }
            _ => {
                if key == "general.architecture" {
                    return None;
                }
                let scalar = reader.read_scalar(val_type)?;
                if tracked {
                    let entry = match scalar {
                        GgufScalar::Int(v) => GgufMetaValue::Int(v),
                        GgufScalar::NonInt => GgufMetaValue::Invalid,
                    };
                    if meta_values.insert(key, entry).is_some() {
                        return None;
                    }
                }
            }
        }
    }

    let arch = arch?;
    let req_pos_u32 = |k: &str| -> Option<u32> {
        match meta_values.get(k)? {
            GgufMetaValue::Int(v) => u32::try_from(*v).ok().filter(|&x| x > 0),
            _ => None,
        }
    };
    let opt_u32 = |k: &str| -> std::result::Result<Option<u32>, ()> {
        match meta_values.get(k) {
            None => Ok(None),
            Some(GgufMetaValue::Int(v)) => u32::try_from(*v).map(Some).map_err(|_| ()),
            Some(_) => Err(()),
        }
    };
    let opt_scalar_or_max_u32 = |k: &str| -> std::result::Result<Option<u32>, ()> {
        match meta_values.get(k) {
            None => Ok(None),
            Some(GgufMetaValue::Int(v)) => u32::try_from(*v).map(Some).map_err(|_| ()),
            Some(GgufMetaValue::IntArray(arr)) if !arr.is_empty() => {
                let mut max_v = 0u32;
                for &v in arr {
                    let u = u32::try_from(v).map_err(|_| ())?;
                    max_v = max_v.max(u);
                }
                Ok(Some(max_v))
            }
            Some(_) => Err(()),
        }
    };

    let block_count = req_pos_u32(&format!("{}.block_count", arch))?;
    let nextn = opt_u32(&format!("{}.nextn_predict_layers", arch))
        .ok()?
        .unwrap_or(0);
    if nextn >= block_count {
        return None;
    }
    let trunk_layers = block_count - nextn;

    let embedding_length = req_pos_u32(&format!("{}.embedding_length", arch))?;
    let head_count = opt_scalar_or_max_u32(&format!("{}.attention.head_count", arch)).ok()??;
    if head_count == 0 {
        return None;
    }
    let head_count_kv = opt_scalar_or_max_u32(&format!("{}.attention.head_count_kv", arch))
        .ok()?
        .unwrap_or(head_count);
    if head_count_kv == 0 {
        return None;
    }

    let default_head_dim = embedding_length / head_count;
    let key_length = opt_u32(&format!("{}.attention.key_length", arch))
        .ok()?
        .unwrap_or(default_head_dim);
    let value_length = opt_u32(&format!("{}.attention.value_length", arch))
        .ok()?
        .unwrap_or(default_head_dim);
    if key_length == 0 || value_length == 0 {
        return None;
    }

    let full_attn_interval = opt_u32(&format!("{}.full_attention_interval", arch)).ok()?;
    if matches!(full_attn_interval, Some(0)) {
        return None;
    }

    let n_attention_layers = match meta_values.get(&format!("{}.attention.recurrent_layers", arch))
    {
        Some(GgufMetaValue::IntArray(recr)) => {
            if u32::try_from(recr.len()).ok() != Some(block_count)
                || !recr.iter().all(|&v| v == 0 || v == 1)
            {
                return None;
            }
            recr.iter()
                .take(trunk_layers as usize)
                .filter(|&&v| v == 0)
                .count() as u32
        }
        Some(_) => return None,
        None => {
            if let Some(interval) = full_attn_interval {
                trunk_layers / interval
            } else {
                trunk_layers
            }
        }
    };
    if n_attention_layers == 0 || n_attention_layers > block_count {
        return None;
    }

    Some(GgufModelMeta {
        arch,
        block_count,
        n_attention_layers,
        embedding_length,
        head_count,
        head_count_kv,
        key_length,
        value_length,
    })
}

fn read_gguf_model_meta(model_path: &Path) -> Option<GgufModelMeta> {
    let file = File::open(model_path).ok()?;
    parse_gguf_model_meta(BufReader::with_capacity(64 * 1024, file))
}

/// Compute f16 KV cache size in bytes:
/// n_attention_layers * ctx * n_head_kv * (key_len + value_len) * 2
fn kv_cache_bytes(meta: &GgufModelMeta, context_size: u32) -> u64 {
    (meta.n_attention_layers as u64)
        .saturating_mul(context_size as u64)
        .saturating_mul(meta.head_count_kv as u64)
        .saturating_mul((meta.key_length as u64).saturating_add(meta.value_length as u64))
        .saturating_mul(2)
}

/// Fallback layer count and KV cache estimate when GGUF metadata cannot be read
fn heuristic_layers_and_kv_gb(file_size_gb: f32, context_size: u32) -> (u32, f32) {
    let estimated_layers = if file_size_gb > 2.5 { 33 } else { 28 };
    let kv_per_1k_gb = if file_size_gb > 2.5 { 0.25 } else { 0.12 };
    let total_kv_gb = (context_size as f32 / 1000.0) * kv_per_1k_gb;
    (estimated_layers, total_kv_gb)
}

/// Pure GPU layer math given model weight size, layer counts, KV size, and available VRAM
fn compute_gpu_layers(
    file_size_gb: f32,
    model_layers: u32,
    n_attention_layers: u32,
    total_kv_gb: f32,
    vram_gb: f32,
) -> u32 {
    let safe_vram = vram_gb - 0.5;
    if !file_size_gb.is_finite()
        || !total_kv_gb.is_finite()
        || !safe_vram.is_finite()
        || file_size_gb <= 0.0
        || model_layers == 0
        || n_attention_layers == 0
        || total_kv_gb < 0.0
        || safe_vram <= 0.0
    {
        return 0;
    }

    if file_size_gb + total_kv_gb <= safe_vram {
        return model_layers;
    }

    let attn_layers = n_attention_layers.min(model_layers);
    let weight_per_layer = file_size_gb / model_layers as f32;
    let kv_per_layer = total_kv_gb / attn_layers as f32;
    let total_per_layer = weight_per_layer + kv_per_layer;
    if total_per_layer <= 0.0 {
        return 0;
    }

    let safe_layers = (safe_vram / total_per_layer).floor() as u32;
    safe_layers.min(model_layers)
}

/// Calculate safe GPU layer count based on VRAM, model weights, and KV cache size
fn calculate_gpu_layers(
    file_size_gb: f32,
    model_layers: u32,
    n_attention_layers: u32,
    total_kv_gb: f32,
    vram_gb: f32,
    context_size: u32,
) -> u32 {
    if file_size_gb <= 0.0 || model_layers == 0 || n_attention_layers == 0 {
        eprintln!("⚠️ Could not determine model file size, using conservative default");
        return 0;
    }

    // Safety buffer (500MB) for OS/Display
    let safe_vram = vram_gb - 0.5;

    // For debugging
    eprintln!("📊 VRAM Analysis:");
    eprintln!("   • Available: {:.2} GB", vram_gb);
    eprintln!("   • Safe Limit: {:.2} GB", safe_vram);
    eprintln!("   • Model Weights: {:.2} GB", file_size_gb);
    eprintln!(
        "   • KV Cache ({} ctx): {:.2} GB",
        context_size, total_kv_gb
    );

    if safe_vram <= 0.0 {
        eprintln!("⚠️ No safe VRAM available, using CPU only");
        return 0;
    }

    // Calculate conservative cost per offloaded layer
    let attn_layers = n_attention_layers.min(model_layers);
    let weight_per_layer = file_size_gb / model_layers as f32;
    let kv_per_layer = total_kv_gb / attn_layers as f32;
    let total_per_layer = weight_per_layer + kv_per_layer;

    let layers = compute_gpu_layers(
        file_size_gb,
        model_layers,
        n_attention_layers,
        total_kv_gb,
        vram_gb,
    );

    eprintln!(
        "   • Cost per layer: {:.2} MB (Weights) + {:.2} MB (KV) = {:.2} MB",
        weight_per_layer * 1024.0,
        kv_per_layer * 1024.0,
        total_per_layer * 1024.0
    );

    if layers < model_layers {
        eprintln!(
            "⚠️ Memory constrained. Offloading {}/{} layers ({:.1}%)",
            layers,
            model_layers,
            (layers as f32 / model_layers as f32) * 100.0
        );
    } else {
        eprintln!("✅ Full offload possible ({} layers)", layers);
    }

    layers
}

/// Get default GPU layer count using GGUF header metadata with heuristic fallback
fn get_default_gpu_layers(model_path: &Path, context_size: u32) -> u32 {
    let vram = detect_vram_gb();
    let file_size_gb = std::fs::metadata(model_path)
        .map(|m| m.len() as f32 / 1024.0 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    if file_size_gb == 0.0 {
        eprintln!("⚠️ Could not determine model file size, using conservative default");
        return 0;
    }

    let (model_layers, n_attention_layers, total_kv_gb) = match read_gguf_model_meta(model_path) {
        Some(meta) => {
            let kv_gb = kv_cache_bytes(&meta, context_size) as f32 / (1024.0 * 1024.0 * 1024.0);
            (meta.block_count, meta.n_attention_layers, kv_gb)
        }
        None => {
            eprintln!("⚠️ Could not read GGUF metadata, falling back to file-size heuristic");
            let (layers, kv_gb) = heuristic_layers_and_kv_gb(file_size_gb, context_size);
            (layers, layers, kv_gb)
        }
    };

    let layers = calculate_gpu_layers(
        file_size_gb,
        model_layers,
        n_attention_layers,
        total_kv_gb,
        vram,
        context_size,
    );
    // llama.cpp counts the output head as an extra offloadable layer (n_layer_all + 1)
    if layers >= model_layers && layers > 0 {
        layers.saturating_add(1)
    } else {
        layers
    }
}

// ============================================================================
// Model State Management
// ============================================================================

/// Right-size per-request n_ctx (multiple of 256, capped at context_size) and n_batch (<= 2048)
fn size_request_context(
    prompt_tokens: usize,
    max_tokens: i32,
    context_size: u32,
) -> Result<(u32, u32)> {
    const CTX_MARGIN: u32 = 64;
    const CTX_ALIGN: u32 = 256;
    const MAX_BATCH: u32 = 2048;

    if context_size == 0 {
        bail!("Invalid context size: 0");
    }
    if prompt_tokens == 0 {
        bail!("Prompt tokenized to 0 tokens");
    }
    if prompt_tokens.saturating_add(1) > context_size as usize {
        bail!(
            "Prompt length ({} tokens) exceeds context size ({})",
            prompt_tokens,
            context_size
        );
    }

    let max_new = u32::try_from(max_tokens.max(1)).unwrap_or(context_size);
    let needed = (prompt_tokens as u32)
        .saturating_add(max_new)
        .saturating_add(CTX_MARGIN);
    let aligned = (needed.saturating_add(CTX_ALIGN - 1) / CTX_ALIGN).saturating_mul(CTX_ALIGN);
    let n_ctx = aligned.min(context_size);
    let n_batch = n_ctx.min(MAX_BATCH);

    Ok((n_ctx, n_batch))
}

struct ModelState {
    backend: LlamaBackend,
    model: Option<LlamaModel>,
    model_path: Option<PathBuf>,
    context_size: u32,
    last_activity: Arc<AtomicU64>,
}

impl ModelState {
    fn new() -> Result<Self> {
        let backend = LlamaBackend::init().context("Failed to init LlamaBackend")?;
        Ok(Self {
            backend,
            model: None,
            model_path: None,
            context_size: 2048,
            last_activity: Arc::new(AtomicU64::new(Self::current_timestamp())),
        })
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    fn update_activity(&self) {
        self.last_activity
            .store(Self::current_timestamp(), Ordering::SeqCst);
    }

    fn seconds_since_activity(&self) -> u64 {
        Self::current_timestamp() - self.last_activity.load(Ordering::SeqCst)
    }

    fn load_model_if_needed(&mut self, model_path: PathBuf, context_size: u32) -> Result<()> {
        // Check if model is already loaded
        if let Some(ref loaded_path) = self.model_path {
            if loaded_path == &model_path && self.context_size == context_size {
                eprintln!("✓ Model already loaded");
                self.update_activity();
                return Ok(());
            }
        }

        eprintln!("📥 Loading model: {}", model_path.display());

        // Detect GPU layers
        let gpu_layers = get_default_gpu_layers(&model_path, context_size);

        // Configure model parameters with GPU offload
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let model_params = pin!(model_params);

        let model = LlamaModel::load_from_file(&self.backend, model_path.clone(), &model_params)
            .with_context(|| format!("unable to load model at {:?}", model_path))?;

        self.model = Some(model);
        self.model_path = Some(model_path);
        self.context_size = context_size;
        self.update_activity();

        eprintln!("✅ Model loaded successfully");
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: String,
        max_tokens: i32,
        sampling: SamplingConfig,
        stop_tokens: Vec<String>,
    ) -> Result<String> {
        let start_time = Instant::now();
        let model = self.model.as_ref().context("Model not loaded")?;

        let tokens_list = model
            .str_to_token(&prompt, AddBos::Always)
            .with_context(|| "failed to tokenize prompt")?;

        eprintln!("📝 Tokenized prompt: {} tokens", tokens_list.len());

        let (n_ctx, n_batch) =
            size_request_context(tokens_list.len(), max_tokens, self.context_size)?;

        // Calculate thread count (conservative default: max(1, (Cores / 2) + 2))
        // This ensures the UI thread is never starved
        let threads: i32 = std::thread::available_parallelism()
            .map(|n| {
                let cores = n.get() as i32;
                ((cores / 2) + 2).max(1)
            })
            .unwrap_or(2);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(
                NonZeroU32::new(n_ctx).context("Invalid ctx size")?,
            ))
            .with_n_batch(n_batch)
            .with_n_threads(threads)
            .with_n_threads_batch(threads);

        let mut ctx = model
            .new_context(&self.backend, ctx_params)
            .context("unable to create the llama_context")?;

        let batch_size = n_batch as usize;
        let mut batch = LlamaBatch::new(batch_size, 1);

        let n_prompt = tokens_list.len();
        let last_index = n_prompt - 1;
        for (chunk_start, chunk) in tokens_list.chunks(batch_size).enumerate() {
            batch.clear();
            let base_pos = chunk_start * batch_size;
            for (offset, &token) in chunk.iter().enumerate() {
                let pos = base_pos + offset;
                let is_last = pos == last_index;
                batch
                    .add(token, pos as i32, &[0], is_last)
                    .context("Failed to add token to batch")?;
            }
            ctx.decode(&mut batch).context("llama_decode() failed")?;
        }
        let prompt_time = start_time.elapsed();

        let n_prompt_tokens = n_prompt as i32;
        let mut n_cur = n_prompt_tokens;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();

        eprintln!(
            "🔄 Starting generation (max_tokens: {}, n_ctx: {}, n_batch: {})",
            max_tokens, n_ctx, n_batch
        );

        use llama_cpp_2::sampling::LlamaSampler;

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u32;
        let mut samplers: Vec<LlamaSampler> = Vec::with_capacity(5);
        if sampling.uses_penalties() {
            samplers.push(LlamaSampler::penalties(
                sampling.penalty_last_n,
                sampling.repeat_penalty,
                sampling.frequency_penalty,
                sampling.presence_penalty,
            ));
        }
        if sampling.temperature <= 0.0 {
            samplers.push(LlamaSampler::greedy());
        } else {
            samplers.push(LlamaSampler::top_k(sampling.top_k));
            samplers.push(LlamaSampler::top_p(sampling.top_p, 1));
            samplers.push(LlamaSampler::temp(sampling.temperature));
            samplers.push(LlamaSampler::dist(seed));
        }
        let sampler = LlamaSampler::chain_simple(samplers);
        let mut sampler = pin!(sampler);

        loop {
            // Check if we've generated enough tokens
            if (n_cur - n_prompt_tokens) >= max_tokens {
                eprintln!("✓ Reached max_tokens limit");
                break;
            }

            // sample() already calls llama_sampler_accept() internally
            let token = sampler.as_mut().sample(&ctx, batch.n_tokens() - 1);

            if model.is_eog_token(token) {
                eprintln!(
                    "✓ End-of-generation token reached (generated {} chars)",
                    output.len()
                );
                break;
            }

            let output_bytes = match model.token_to_piece_bytes(token, 32, true, None) {
                Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(size)) => {
                    let required_size: usize = size
                        .checked_neg()
                        .context("Invalid token piece buffer size")?
                        .try_into()
                        .context("Invalid token piece buffer size")?;
                    model.token_to_piece_bytes(token, required_size, true, None)
                }
                result => result,
            }
            .context("Failed to convert token to bytes")?;

            let mut token_text = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut token_text, false);
            output.push_str(&token_text);

            // Check for model-specific stop tokens
            let mut should_stop = false;
            for stop_token in &stop_tokens {
                if output.contains(stop_token) {
                    eprintln!(
                        "✓ Stop token '{}' detected (generated {} chars)",
                        stop_token,
                        output.len()
                    );
                    // Remove the stop token from output
                    output = output.replace(stop_token, "").trim_end().to_string();
                    should_stop = true;
                    break;
                }
            }
            if should_stop {
                break;
            }

            let pos = n_cur;
            n_cur += 1;
            if (n_cur - n_prompt_tokens) >= max_tokens {
                eprintln!("✓ Reached max_tokens limit");
                break;
            }
            if pos >= n_ctx as i32 {
                eprintln!("✓ Reached context size limit");
                break;
            }

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .context("Failed to add generated token to batch")?;
            ctx.decode(&mut batch).context("failed to eval")?;
        }

        // Generation statistics
        let total_time = start_time.elapsed();
        let gen_time = total_time.saturating_sub(prompt_time);
        let output_tokens = (n_cur - n_prompt_tokens) as u64;
        let prompt_tokens = n_prompt_tokens as u64;

        let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
            output_tokens as f64 / gen_time.as_secs_f64()
        } else {
            0.0
        };

        eprintln!("📊 Generation Statistics:");
        eprintln!("   • Prompt tokens: {}", prompt_tokens);
        eprintln!("   • Output tokens: {}", output_tokens);
        eprintln!("   • Prompt processing: {:.2}s", prompt_time.as_secs_f64());
        eprintln!("   • Generation time: {:.2}s", gen_time.as_secs_f64());
        eprintln!("   • Total time: {:.2}s", total_time.as_secs_f64());
        eprintln!("   • Speed: {:.2} tokens/sec", tokens_per_sec);

        self.update_activity();
        Ok(output)
    }
}

// ============================================================================
// Main Loop with Keep-Alive Protocol
// ============================================================================

fn send_response(response: &Response) -> Result<()> {
    let json = serde_json::to_string(response)?;
    println!("{}", json);
    io::stdout().flush()?;
    Ok(())
}

fn main() -> Result<()> {
    // Get idle timeout from environment variable (default 5 minutes)
    let idle_timeout_secs = std::env::var("LLAMA_IDLE_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(300); // 5 minutes default

    eprintln!(
        "🦙 llama-helper starting (idle timeout: {}s)",
        idle_timeout_secs
    );

    let mut state = ModelState::new()?;

    let stdin = io::stdin();
    let mut stdin_lock = stdin.lock();
    let mut buffer = String::new();

    loop {
        // Check idle timeout
        if state.seconds_since_activity() > idle_timeout_secs {
            eprintln!("💤 Idle timeout reached, shutting down");
            send_response(&Response::Goodbye)?;
            break;
        }

        // Read line from stdin
        buffer.clear();
        match stdin_lock.read_line(&mut buffer) {
            Ok(0) => {
                // EOF reached
                eprintln!("📪 EOF received, shutting down");
                break;
            }
            Ok(_) => {
                let line = buffer.trim();
                if line.is_empty() {
                    continue;
                }

                // Parse request
                match serde_json::from_str::<Request>(line) {
                    Ok(Request::Generate {
                        prompt,
                        max_tokens,
                        context_size,
                        model_path,
                        temperature,
                        top_k,
                        top_p,
                        presence_penalty,
                        frequency_penalty,
                        repeat_penalty,
                        penalty_last_n,
                        stop_tokens,
                    }) => {
                        let max_tokens = max_tokens.unwrap_or(512);
                        let context_size = context_size.unwrap_or(2048);

                        let sampling = SamplingConfig::from_request(
                            temperature,
                            top_k,
                            top_p,
                            presence_penalty,
                            frequency_penalty,
                            repeat_penalty,
                            penalty_last_n,
                        );
                        let stop_tokens = stop_tokens.unwrap_or_else(Vec::new);

                        // Load model if path provided
                        if let Some(path_str) = model_path {
                            let path = PathBuf::from(path_str);
                            if let Err(e) = state.load_model_if_needed(path, context_size) {
                                send_response(&Response::Response {
                                    text: String::new(),
                                    error: Some(format!("Failed to load model: {}", e)),
                                })?;
                                continue;
                            }
                        }

                        // Generate response with sampling parameters
                        match state.generate(
                            prompt,
                            max_tokens,
                            sampling,
                            stop_tokens,
                        ) {
                            Ok(text) => {
                                send_response(&Response::Response { text, error: None })?;
                            }
                            Err(e) => {
                                send_response(&Response::Response {
                                    text: String::new(),
                                    error: Some(format!("Generation failed: {}", e)),
                                })?;
                            }
                        }
                    }
                    Ok(Request::Ping) => {
                        state.update_activity();
                        send_response(&Response::Pong)?;
                    }
                    Ok(Request::Shutdown) => {
                        eprintln!("🛑 Shutdown requested");
                        send_response(&Response::Goodbye)?;
                        break;
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to parse request: {}", e);
                        send_response(&Response::Error {
                            message: format!("Invalid request: {}", e),
                        })?;
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Error reading stdin: {}", e);
                break;
            }
        }
    }

    eprintln!("👋 llama-helper exiting");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_gguf_header(buf: &mut Vec<u8>, version: u32, n_kv: u64) {
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&n_kv.to_le_bytes());
    }

    fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn push_gguf_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_gguf_str(buf, val);
    }

    fn push_gguf_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn push_gguf_kv_i32_arr(buf: &mut Vec<u8>, key: &str, vals: &[i32]) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes()); // GGUF_TYPE_INT32
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn push_gguf_kv_bool_raw_arr(buf: &mut Vec<u8>, key: &str, vals: &[u8]) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&7u32.to_le_bytes()); // GGUF_TYPE_BOOL
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        buf.extend_from_slice(vals);
    }

    fn push_gguf_kv_str_arr(buf: &mut Vec<u8>, key: &str, vals: &[&str]) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            push_gguf_str(buf, v);
        }
    }

    fn push_base_qwen35_kvs(buf: &mut Vec<u8>, block_count: u32) {
        push_gguf_kv_str(buf, "general.architecture", "qwen35");
        push_gguf_kv_u32(buf, "qwen35.block_count", block_count);
        push_gguf_kv_u32(buf, "qwen35.embedding_length", 2560);
        push_gguf_kv_u32(buf, "qwen35.attention.head_count", 16);
        push_gguf_kv_u32(buf, "qwen35.attention.head_count_kv", 4);
        push_gguf_kv_u32(buf, "qwen35.attention.key_length", 256);
        push_gguf_kv_u32(buf, "qwen35.attention.value_length", 256);
    }

    #[test]
    fn gguf_reader_parses_hybrid_qwen35_buffer_and_skips_unused_entries() {
        let mut buf = Vec::new();
        push_gguf_header(&mut buf, 3, 10);

        push_gguf_kv_str(&mut buf, "general.architecture", "qwen35");
        push_gguf_kv_str(&mut buf, "general.name", "Qwen3.5-4B");
        push_gguf_kv_u32(&mut buf, "qwen35.block_count", 32);
        push_gguf_kv_u32(&mut buf, "qwen35.embedding_length", 2560);
        push_gguf_kv_u32(&mut buf, "qwen35.attention.head_count", 16);
        push_gguf_kv_u32(&mut buf, "qwen35.attention.head_count_kv", 4);
        push_gguf_kv_u32(&mut buf, "qwen35.attention.key_length", 256);
        push_gguf_kv_u32(&mut buf, "qwen35.attention.value_length", 256);
        push_gguf_kv_u32(&mut buf, "qwen35.full_attention_interval", 4);
        push_gguf_kv_str_arr(&mut buf, "tokenizer.ggml.tokens", &["<s>", "hello", "{}"]);

        let meta = parse_gguf_model_meta(&buf[..]).expect("should parse GGUF header");
        assert_eq!(
            meta,
            GgufModelMeta {
                arch: "qwen35".to_string(),
                block_count: 32,
                n_attention_layers: 8,
                embedding_length: 2560,
                head_count: 16,
                head_count_kv: 4,
                key_length: 256,
                value_length: 256,
            }
        );

        // Wrong magic returns None
        assert!(parse_gguf_model_meta(&b"NOT_GGUF"[..]).is_none());
        let mut bad_magic = buf.clone();
        bad_magic[..4].copy_from_slice(b"GGML");
        assert!(parse_gguf_model_meta(&bad_magic[..]).is_none());

        // Truncated buffer (including mid-KV cuts) returns None without panicking
        assert!(parse_gguf_model_meta(&buf[..buf.len() / 2]).is_none());
        for cut in 0..buf.len() {
            assert!(parse_gguf_model_meta(&buf[..cut]).is_none());
        }
    }

    #[test]
    fn gguf_reader_parses_v2_header_and_validates_hybrid_metadata() {
        // GGUF v2 header with valid boolean recurrent_layers parses
        let mut v2_buf = Vec::new();
        push_gguf_header(&mut v2_buf, 2, 8);
        push_base_qwen35_kvs(&mut v2_buf, 4);
        push_gguf_kv_bool_raw_arr(&mut v2_buf, "qwen35.attention.recurrent_layers", &[1, 1, 1, 0]);
        let v2_meta = parse_gguf_model_meta(&v2_buf[..]).expect("should parse GGUF v2 header");
        assert_eq!(v2_meta.block_count, 4);
        assert_eq!(v2_meta.n_attention_layers, 1);

        // Valid integer recurrent_layers parses
        let mut i32_recr_buf = Vec::new();
        push_gguf_header(&mut i32_recr_buf, 3, 8);
        push_base_qwen35_kvs(&mut i32_recr_buf, 4);
        push_gguf_kv_i32_arr(
            &mut i32_recr_buf,
            "qwen35.attention.recurrent_layers",
            &[1, 0, 1, 0],
        );
        let i32_meta =
            parse_gguf_model_meta(&i32_recr_buf[..]).expect("should parse integer recurrent_layers");
        assert_eq!(i32_meta.n_attention_layers, 2);

        // Unsupported GGUF v1 header is rejected
        let mut v1_buf = Vec::new();
        push_gguf_header(&mut v1_buf, 1, 7);
        push_base_qwen35_kvs(&mut v1_buf, 4);
        assert!(parse_gguf_model_meta(&v1_buf[..]).is_none());

        // recurrent_layers with wrong length (shorter or longer than block_count) returns None
        for bad_recr in [&[1i32, 1, 0][..], &[1i32, 1, 0, 1, 0][..]] {
            let mut buf = Vec::new();
            push_gguf_header(&mut buf, 3, 8);
            push_base_qwen35_kvs(&mut buf, 4);
            push_gguf_kv_i32_arr(&mut buf, "qwen35.attention.recurrent_layers", bad_recr);
            assert!(parse_gguf_model_meta(&buf[..]).is_none());
        }

        // recurrent_layers with value 2, negative value, or zero attention layers returns None
        for bad_recr in [
            &[1i32, 2, 1, 0][..],
            &[1i32, -1, 1, 0][..],
            &[1i32, 1, 1, 1][..],
        ] {
            let mut buf = Vec::new();
            push_gguf_header(&mut buf, 3, 8);
            push_base_qwen35_kvs(&mut buf, 4);
            push_gguf_kv_i32_arr(&mut buf, "qwen35.attention.recurrent_layers", bad_recr);
            assert!(parse_gguf_model_meta(&buf[..]).is_none());
        }

        // recurrent_layers BOOL array with raw byte 2 returns None
        let mut bad_bool_buf = Vec::new();
        push_gguf_header(&mut bad_bool_buf, 3, 8);
        push_base_qwen35_kvs(&mut bad_bool_buf, 4);
        push_gguf_kv_bool_raw_arr(
            &mut bad_bool_buf,
            "qwen35.attention.recurrent_layers",
            &[1, 2, 1, 0],
        );
        assert!(parse_gguf_model_meta(&bad_bool_buf[..]).is_none());

        // full_attention_interval = 0 (or > block_count yielding 0 attention layers) returns None
        for bad_interval in [0u32, 64] {
            let mut buf = Vec::new();
            push_gguf_header(&mut buf, 3, 8);
            push_base_qwen35_kvs(&mut buf, 32);
            push_gguf_kv_u32(&mut buf, "qwen35.full_attention_interval", bad_interval);
            assert!(parse_gguf_model_meta(&buf[..]).is_none());
        }
    }

    enum TestChunk {
        Bytes(Vec<u8>, usize),
        Zeroes(u64),
    }

    struct SegmentedReader {
        chunks: Vec<TestChunk>,
        idx: usize,
        bytes_read: u64,
    }

    impl SegmentedReader {
        fn new(chunks: Vec<TestChunk>) -> Self {
            Self {
                chunks,
                idx: 0,
                bytes_read: 0,
            }
        }
    }

    impl Read for SegmentedReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            while self.idx < self.chunks.len() {
                let n = match &mut self.chunks[self.idx] {
                    TestChunk::Bytes(data, pos) => {
                        let avail = data.len().saturating_sub(*pos);
                        let to_copy = avail.min(buf.len());
                        buf[..to_copy].copy_from_slice(&data[*pos..*pos + to_copy]);
                        *pos += to_copy;
                        to_copy
                    }
                    TestChunk::Zeroes(remaining) => {
                        let to_fill = (*remaining).min(buf.len() as u64) as usize;
                        buf[..to_fill].fill(0);
                        *remaining -= to_fill as u64;
                        to_fill
                    }
                };
                if n > 0 {
                    self.bytes_read += n as u64;
                    return Ok(n);
                }
                self.idx += 1;
            }
            Ok(0)
        }
    }

    fn push_gguf_header_with_tensors(buf: &mut Vec<u8>, version: u32, n_tensors: u64, n_kv: u64) {
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&n_tensors.to_le_bytes());
        buf.extend_from_slice(&n_kv.to_le_bytes());
    }

    #[test]
    fn gguf_reader_enforces_aggregate_read_budget_and_header_count_caps() {
        // Positive control: a 16 MiB tokenizer payload (comparable to Gemma 4 / Qwen3.5) parses
        let ok_str_len: u64 = 16 * 1024 * 1024;
        let mut ok_prefix = Vec::new();
        push_gguf_header(&mut ok_prefix, 3, 8);
        push_base_qwen35_kvs(&mut ok_prefix, 32);
        push_gguf_str(&mut ok_prefix, "tokenizer.chat_template");
        ok_prefix.extend_from_slice(&8u32.to_le_bytes());
        ok_prefix.extend_from_slice(&ok_str_len.to_le_bytes());
        let mut ok_reader = SegmentedReader::new(vec![
            TestChunk::Bytes(ok_prefix, 0),
            TestChunk::Zeroes(ok_str_len),
        ]);
        assert!(parse_gguf_model_meta(&mut ok_reader).is_some());

        // Single string whose declared length pushes header + KV past 64 MiB bails out
        // before reading any payload bytes (both on a short buffer and a large backing stream)
        let mut over_prefix = Vec::new();
        push_gguf_header(&mut over_prefix, 3, 8);
        push_base_qwen35_kvs(&mut over_prefix, 32);
        push_gguf_str(&mut over_prefix, "tokenizer.chat_template");
        over_prefix.extend_from_slice(&8u32.to_le_bytes());
        over_prefix.extend_from_slice(&GGUF_MAX_SKIP_STR_BYTES.to_le_bytes());
        let over_prefix_len = over_prefix.len() as u64;
        assert!(parse_gguf_model_meta(&over_prefix[..]).is_none());
        let mut over_reader = SegmentedReader::new(vec![
            TestChunk::Bytes(over_prefix, 0),
            TestChunk::Zeroes(GGUF_MAX_SKIP_STR_BYTES),
        ]);
        assert!(parse_gguf_model_meta(&mut over_reader).is_none());
        assert_eq!(over_reader.bytes_read, over_prefix_len);

        // String array whose elements are each < 64 MiB (2 x 36 MiB = 72 MiB) bails out
        // as soon as the second element's declared length exceeds the remaining budget
        let elem_str_len: u64 = 36 * 1024 * 1024;
        let mut arr_head = Vec::new();
        push_gguf_header(&mut arr_head, 3, 8);
        push_base_qwen35_kvs(&mut arr_head, 32);
        push_gguf_str(&mut arr_head, "tokenizer.ggml.tokens");
        arr_head.extend_from_slice(&9u32.to_le_bytes());
        arr_head.extend_from_slice(&8u32.to_le_bytes()); // elem_type = GGUF_TYPE_STRING
        arr_head.extend_from_slice(&2u64.to_le_bytes());
        arr_head.extend_from_slice(&elem_str_len.to_le_bytes());
        let arr_head_len = arr_head.len() as u64;
        let arr_second_hdr = elem_str_len.to_le_bytes().to_vec();
        let arr_second_hdr_len = arr_second_hdr.len() as u64;
        let mut str_arr_reader = SegmentedReader::new(vec![
            TestChunk::Bytes(arr_head, 0),
            TestChunk::Zeroes(elem_str_len),
            TestChunk::Bytes(arr_second_hdr, 0),
            TestChunk::Zeroes(elem_str_len),
        ]);
        assert!(parse_gguf_model_meta(&mut str_arr_reader).is_none());
        assert_eq!(
            str_arr_reader.bytes_read,
            arr_head_len + elem_str_len + arr_second_hdr_len
        );

        // Multiple scalar arrays (3 x 3_000_000 u64 = 72 MB > 64 MiB, each < GGUF_MAX_SKIP_ARRAY_LEN)
        // bail out before skipping the third array's payload
        let scalar_arr_len: u64 = 3_000_000;
        let scalar_arr_bytes: u64 = scalar_arr_len * 8;
        let push_u64_arr_hdr = |dst: &mut Vec<u8>, key: &str| {
            push_gguf_str(dst, key);
            dst.extend_from_slice(&9u32.to_le_bytes());
            dst.extend_from_slice(&10u32.to_le_bytes()); // elem_type = GGUF_TYPE_UINT64
            dst.extend_from_slice(&scalar_arr_len.to_le_bytes());
        };
        let mut s0 = Vec::new();
        push_gguf_header(&mut s0, 3, 10);
        push_base_qwen35_kvs(&mut s0, 32);
        push_u64_arr_hdr(&mut s0, "pad.0");
        let mut s1 = Vec::new();
        push_u64_arr_hdr(&mut s1, "pad.1");
        let mut s2 = Vec::new();
        push_u64_arr_hdr(&mut s2, "pad.2");
        let expected_scalar_read = s0.len() as u64
            + scalar_arr_bytes
            + s1.len() as u64
            + scalar_arr_bytes
            + s2.len() as u64;
        let mut scalar_arr_reader = SegmentedReader::new(vec![
            TestChunk::Bytes(s0, 0),
            TestChunk::Zeroes(scalar_arr_bytes),
            TestChunk::Bytes(s1, 0),
            TestChunk::Zeroes(scalar_arr_bytes),
            TestChunk::Bytes(s2, 0),
            TestChunk::Zeroes(scalar_arr_bytes),
        ]);
        assert!(parse_gguf_model_meta(&mut scalar_arr_reader).is_none());
        assert_eq!(scalar_arr_reader.bytes_read, expected_scalar_read);

        // Absurd n_tensors or n_kv is rejected up front after the 24-byte fixed header
        for (bad_tensors, bad_kv) in [
            (GGUF_MAX_TENSOR_COUNT + 1, 7),
            (u64::MAX, 7),
            (0, GGUF_MAX_KV_COUNT + 1),
            (0, u64::MAX),
        ] {
            let mut hdr_buf = Vec::new();
            push_gguf_header_with_tensors(&mut hdr_buf, 3, bad_tensors, bad_kv);
            push_base_qwen35_kvs(&mut hdr_buf, 32);
            let mut hdr_reader = SegmentedReader::new(vec![TestChunk::Bytes(hdr_buf, 0)]);
            assert!(parse_gguf_model_meta(&mut hdr_reader).is_none());
            assert_eq!(hdr_reader.bytes_read, 24);
        }
    }

    #[test]
    fn gguf_reader_rejects_nextn_predict_layers_at_or_above_block_count() {
        // Valid nextn_predict_layers (< block_count) subtracts MTP layers from trunk_layers
        let mut valid_buf = Vec::new();
        push_gguf_header(&mut valid_buf, 3, 9);
        push_base_qwen35_kvs(&mut valid_buf, 34);
        push_gguf_kv_u32(&mut valid_buf, "qwen35.nextn_predict_layers", 2);
        push_gguf_kv_u32(&mut valid_buf, "qwen35.full_attention_interval", 4);
        let valid_meta =
            parse_gguf_model_meta(&valid_buf[..]).expect("nextn < block_count should parse");
        assert_eq!(valid_meta.block_count, 34);
        assert_eq!(valid_meta.n_attention_layers, 8);

        // Explicit nextn_predict_layers = 0 (< block_count) also parses
        let mut zero_buf = Vec::new();
        push_gguf_header(&mut zero_buf, 3, 9);
        push_base_qwen35_kvs(&mut zero_buf, 32);
        push_gguf_kv_u32(&mut zero_buf, "qwen35.nextn_predict_layers", 0);
        push_gguf_kv_u32(&mut zero_buf, "qwen35.full_attention_interval", 4);
        let zero_meta =
            parse_gguf_model_meta(&zero_buf[..]).expect("nextn == 0 < block_count should parse");
        assert_eq!(zero_meta.n_attention_layers, 8);

        // nextn_predict_layers >= block_count returns None (both hybrid and dense)
        for bad_nextn in [32u32, 33, u32::MAX] {
            let mut hybrid_buf = Vec::new();
            push_gguf_header(&mut hybrid_buf, 3, 9);
            push_base_qwen35_kvs(&mut hybrid_buf, 32);
            push_gguf_kv_u32(&mut hybrid_buf, "qwen35.nextn_predict_layers", bad_nextn);
            push_gguf_kv_u32(&mut hybrid_buf, "qwen35.full_attention_interval", 4);
            assert!(parse_gguf_model_meta(&hybrid_buf[..]).is_none());

            let mut dense_buf = Vec::new();
            push_gguf_header(&mut dense_buf, 3, 8);
            push_base_qwen35_kvs(&mut dense_buf, 32);
            push_gguf_kv_u32(&mut dense_buf, "qwen35.nextn_predict_layers", bad_nextn);
            assert!(parse_gguf_model_meta(&dense_buf[..]).is_none());
        }
    }

    #[test]
    fn kv_and_gpu_layer_math_handles_hybrid_and_dense_models() {
        let qwen35_4b = GgufModelMeta {
            arch: "qwen35".to_string(),
            block_count: 32,
            n_attention_layers: 8,
            embedding_length: 2560,
            head_count: 16,
            head_count_kv: 4,
            key_length: 256,
            value_length: 256,
        };
        // 8 * 32768 * 4 * (256 + 256) * 2 = 1,073,741,824 bytes (1.00 GiB)
        let kv_bytes = kv_cache_bytes(&qwen35_4b, 32768);
        assert_eq!(kv_bytes, 1_073_741_824);
        let kv_gb = kv_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
        assert!((kv_gb - 1.0).abs() < 1e-6);

        // On 16 GB M1 (9.6 GB Metal budget), 2.55 GB Qwen3.5-4B fits completely (32/32)
        assert_eq!(
            compute_gpu_layers(
                2.55,
                qwen35_4b.block_count,
                qwen35_4b.n_attention_layers,
                kv_gb,
                9.6
            ),
            32
        );
        // Full offload still applies when weights + total KV (3.55 GB) fits in safe_vram (4.5 GB)
        assert_eq!(
            compute_gpu_layers(
                2.55,
                qwen35_4b.block_count,
                qwen35_4b.n_attention_layers,
                kv_gb,
                5.0
            ),
            32
        );

        // Old file-size heuristic spilled Qwen3.5-4B layers to CPU (27/33)
        let (heur_layers, heur_kv_gb) = heuristic_layers_and_kv_gb(2.55, 32768);
        assert_eq!(heur_layers, 33);
        assert!((heur_kv_gb - 8.192).abs() < 1e-4);
        assert_eq!(
            compute_gpu_layers(2.55, heur_layers, heur_layers, heur_kv_gb, 9.6),
            27
        );

        // Memory-constrained partial offload when full model does not fit (safe_vram = 2.0 < 3.55):
        // Hybrid charges weight_per_layer (2.55/32 = 0.0796875) + total_kv_gb/8 (0.125) = 0.2046875 -> 9 layers,
        // whereas dense (n_attention_layers == 32) charges 0.0796875 + 1.0/32 (0.03125) = 0.1109375 -> 18 layers.
        assert_eq!(
            compute_gpu_layers(
                2.55,
                qwen35_4b.block_count,
                qwen35_4b.n_attention_layers,
                kv_gb,
                2.5
            ),
            9
        );
        assert_eq!(compute_gpu_layers(2.55, 32, 32, kv_gb, 2.5), 18);
        assert_eq!(
            compute_gpu_layers(
                2.55,
                qwen35_4b.block_count,
                qwen35_4b.n_attention_layers,
                kv_gb,
                0.4
            ),
            0
        );
    }

    #[test]
    fn size_request_context_right_sizes_and_guards_bounds() {
        // 300-token micro-query + 16 max_tokens + 64 margin = 380 -> 512 ctx, 512 batch
        assert_eq!(size_request_context(300, 16, 32768).unwrap(), (512, 512));

        // 2500-token summary + 4096 max_tokens + 64 margin = 6660 -> 6912 ctx, 2048 batch
        assert_eq!(size_request_context(2500, 4096, 32768).unwrap(), (6912, 2048));

        // Clamped to model context_size when prompt + max_tokens exceeds cap
        assert_eq!(size_request_context(30000, 4096, 32768).unwrap(), (32768, 2048));

        // Exact upper bound (prompt_tokens + 1 == context_size) succeeds
        assert_eq!(size_request_context(32767, 16, 32768).unwrap(), (32768, 2048));

        // Prompt leaving no slot for generation returns an error instead of panicking
        assert!(size_request_context(32768, 16, 32768).is_err());
        assert!(size_request_context(0, 16, 32768).is_err());
    }

    #[test]
    fn generate_request_defaults_penalties_when_omitted() {
        let json = r#"{"type":"generate","prompt":"summarize","temperature":0.5,"top_k":20,"top_p":0.8}"#;
        let request: Request = serde_json::from_str(json).unwrap();
        let Request::Generate {
            temperature,
            top_k,
            top_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            penalty_last_n,
            ..
        } = request else {
            panic!("expected generate request");
        };

        let sampling = SamplingConfig::from_request(
            temperature,
            top_k,
            top_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            penalty_last_n,
        );

        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
        assert_eq!(sampling.repeat_penalty, 1.0);
        assert_eq!(sampling.penalty_last_n, 0);
        assert!(!sampling.uses_penalties());
    }

    #[test]
    fn generate_request_deserializes_qwen_penalties() {
        let json = r#"{"type":"generate","prompt":"summarize","temperature":0.5,"top_k":20,"top_p":0.8,"presence_penalty":0.3,"frequency_penalty":0.0,"repeat_penalty":1.05,"penalty_last_n":256}"#;
        let request: Request = serde_json::from_str(json).unwrap();
        let Request::Generate {
            temperature,
            top_k,
            top_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            penalty_last_n,
            ..
        } = request else {
            panic!("expected generate request");
        };

        let sampling = SamplingConfig::from_request(
            temperature,
            top_k,
            top_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            penalty_last_n,
        );

        assert_eq!(sampling.temperature, 0.5);
        assert_eq!(sampling.top_k, 20);
        assert_eq!(sampling.top_p, 0.8);
        assert_eq!(sampling.presence_penalty, 0.3);
        assert_eq!(sampling.frequency_penalty, 0.0);
        assert_eq!(sampling.repeat_penalty, 1.05);
        assert_eq!(sampling.penalty_last_n, 256);
        assert!(sampling.uses_penalties());
    }
}
