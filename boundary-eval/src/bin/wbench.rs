// Minimal whisper transcriber for eval/build_boundary_cases.py.
//
// Transcribes a 16 kHz mono PCM16 WAV with auto language detection and
// prints one `TEXT <segment>` line per segment on stdout.
//
// Usage:
//   wbench <ggml model> <wav> <threads>

use std::fs;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Read the `data` chunk of a PCM16 WAV into f32 samples in [-1, 1].
fn read_pcm16_wav(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).expect("read wav");
    assert!(
        &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV file"
    );
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"data" {
            let end = (body + len).min(bytes.len());
            return bytes[body..end]
                .chunks_exact(2)
                .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                .collect();
        }
        pos = body + len + (len & 1);
    }
    panic!("WAV has no data chunk");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: wbench <ggml model> <wav> [threads]");
        std::process::exit(2);
    }
    let threads: i32 = args.get(3).and_then(|t| t.parse().ok()).unwrap_or(4);
    let audio = read_pcm16_wav(&args[2]);

    let ctx = WhisperContext::new_with_params(&args[1], WhisperContextParameters::default())
        .expect("load whisper model");
    let mut state = ctx.create_state().expect("create state");

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("auto"));
    params.set_n_threads(threads);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    state.full(params, &audio).expect("transcribe");

    for i in 0..state.full_n_segments().expect("segments") {
        if let Ok(text) = state.full_get_segment_text_lossy(i) {
            println!("TEXT {}", text.trim());
        }
    }
}
