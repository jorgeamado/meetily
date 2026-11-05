# Plan: FluidAudio Parakeet (CoreML / ANE) as a macOS transcription engine

Status: draft, 2026-11-03. Gate (phase 0) run; outcome changed the default
model from Ultra to v3 and repositioned the engine as "same quality, ~15x
faster" (see Phase 0 results).

## Why

Offline STT comparison on the user's own meetings (2-minute clips, mostly
Russian with English terms), word error rate against ElevenLabs Scribe v2 as
reference, Gemini 4 as a second reference on one clip:

| Engine | WER vs Scribe | WER vs Gemini (1 clip) | Time per 2-min clip (M1) |
|---|---|---|---|
| FluidAudio Parakeet Ultra (CoreML, ANE) | 25.3% | 20.7% | ~1.5–2 s |
| whisper.cpp v1.9.5 large-v3 q5 | 26.4% | 20.7% | 27–36 s |
| FluidAudio Parakeet v3 | 27.0% | 26.6% | 1.5–2.8 s |
| Current app engine (whisper-rs 0.13, large-v3-turbo) | 29.4% | 28.8% | 20–35 s |
| whisper.cpp v1.9.5 large-v3-turbo | 32.1% | 42.4% | 16–26 s (drops speech) |
| Groq whisper-large-v3 | 38.2% | 40.2% | cloud (skips sentences) |

On these mostly-Russian clips Parakeet Ultra is the most accurate and ~15x
faster than the current engine. The phase 0 gate below qualifies that.

## Phase 0 — gate results

Three more clips, picked for the densest Russian/English switching
(50dee676 @ 20:00, 1a4fe304 @ 37:30) and an English-only meeting
(warehouse tools @ 10:00). WER vs Scribe v2:

| Clips | Current (whisper-rs turbo) | Parakeet Ultra | Parakeet v3 | whisper.cpp large-v3 q5 |
|---|---|---|---|---|
| Code-switching (2) | **17.1%** | 18.9% | 21.5% | 21.8% |
| English terms recovered (43 in ref) | **86%** | 65% | 81% | 86% |
| English-only (1) | 14.3% | 16.4% | **13.0%** | 35.1% |
| Mostly Russian (original 3) | 29.4% | **25.3%** | 27.0% | 26.4% |
| **All 6 clips (12 min)** | 21.8% | 21.1% | 22.0% | 26.7% |

Findings:

- Overall accuracy is a tie (21–22%). FluidAudio's case is speed at equal quality, not better quality.
- Ultra transliterates English terms into Cyrillic ("forecast" → "форкасты"), which hurts search, the glossary and LLM summaries in mixed meetings. It fails the gate as a default.
- v3 keeps English terms in Latin script, is the best engine on English-only speech, and is close on Russian. It becomes the default FluidAudio model; Ultra stays an option labelled for Russian-only meetings.
- Dense code-switching is still Whisper's strength (17% vs 21.5%), so Whisper stays available and stays the default on non-Apple platforms.
- Caveat: 12 minutes of audio, a single reference (Scribe); differences under ~2 points are noise.

Follow-up to evaluate in phase 3: a hybrid where Parakeet v3 transcribes
everything and Whisper re-decodes only segments that are low-confidence or
mix scripts. That could keep the speed and recover the code-switch accuracy.

## Architecture

A Swift sidecar, `fluid-helper`, in the same shape as `llama-helper`: one
persistent process, line-delimited JSON over stdin/stdout. Swift is required
because FluidAudio is a Swift package; Rust talks to it only through the
protocol, so the Rust core stays platform-neutral and the sidecar is simply
absent on Windows/Linux.

```
Rust (TranscriptionProvider "fluidAudio")
   │  JSON lines: load / transcribe / download / ping / shutdown
   ▼
fluid-helper (Swift, FluidAudio pinned to v0.18.2)
   │  AsrModels.downloadAndLoad(to: <app models>/fluidaudio/<version>, version:)
   ▼
CoreML (ANE encoder, int8)
```

Protocol (one JSON object per line, request ids for correlation):

- `{"type":"load","version":"ultra"|"v3","models_dir":"..."}` → `{"type":"loaded"}` or error; progress lines `{"type":"progress","stage":"download"|"compile","fraction":0.4}` while downloading or compiling for the ANE on first use.
- `{"type":"transcribe","id":7,"wav_path":"/tmp/....wav"}` → `{"type":"result","id":7,"text":"...","confidence":0.93,"tokens":[{"t":"▁при","start":1.23,"end":1.41},...]}`. Audio goes through a temp 16 kHz mono WAV, like diarize-helper, so large buffers never pass through stdin.
- `{"type":"ping"}`, `{"type":"shutdown"}`; idle timeout like llama-helper.

Language: never pass FluidAudio's `language` hint. It filters tokens by script,
which would break mixed Cyrillic/Latin speech.

## Phases

### 1. Engine plumbing refactor (no behaviour change)

From the code map (paths under `frontend/src-tauri/src`):

- `audio/transcription/provider.rs`: add `words: Option<Vec<WordTiming>>` to `TranscriptResult` (None for engines without timestamps).
- Replace the `use_parakeet: bool` in `audio/retranscription.rs` (`:118`, `:213`, `:442-448`, `:576-599`, plus the glossary/repair gates at `:458`, `:765`, `:803`), `audio/import.rs` (`:269`, `:425`, `:630-636`, `:747`) and `audio/common.rs:20` with an enum (`BatchEngine::{Whisper, ParakeetOnnx, FluidAudio}`).
- `audio/transcription/engine.rs:91` and `:186-218`: accept the new provider string; stop the `_ =>` fallback from silently choosing Whisper for unknown providers.
- Normalise provider strings: the config uses `localWhisper`, `src/hooks/useTranscriptionModels.ts` uses `whisper`.
- Decide whether the glossary `initial_prompt` gate (`retranscription.rs:458`) applies; Parakeet has no prompt, so for now the gate is "is Whisper".

Tests: existing suite plus enum-dispatch unit tests. Ships alone.

### 2. `fluid-helper` sidecar

- New Swift package `fluid-helper/` at the repo root depending on FluidAudio `exact: "0.18.2"`.
- Uses `AsrManager` batch transcription (sliding window already handles long audio) and returns token timings.
- Serialise CoreML work inside the helper. FluidAudio reports crashes when ASR and diarization run on CoreML concurrently (FluidAudio issue #661). Our diarization is sherpa-onnx on CPU, so this is low risk, but keep one request in flight.
- Build: `frontend/build-gpu.sh` / `dev-gpu.sh` gain a macOS-only `swift build -c release --product fluid-helper` and copy to `src-tauri/binaries/fluid-helper-<triple>` (uses the target-dir fix from 031e7fc3).
- Bundle: add `binaries/fluid-helper` to `externalBin` in a macOS-only override (`tauri.macos.conf.json`) so Windows/Linux builds do not look for it.
- Signing: ad-hoc like the other sidecars; CoreML/ANE needs no entitlement. Verify the bundled app launches it under the hardened runtime.
- Licences: FluidAudio Apache-2.0; Parakeet v3 / Ultra CoreML models CC-BY-4.0, so add attribution in the About/licences screen.

### 3. Rust provider and batch use

- `FluidAudioProvider` implementing `TranscriptionProvider`, with a sidecar manager modelled on `summary/summary_engine/sidecar.rs` (resolve binary, ensure_running, send_request with timeout, idle shutdown).
- Token timings → `WordSpan` (`audio/diarization.rs:346`): merge SentencePiece pieces into words on the `▁` boundary, carry start/end, set `prob` from confidence. That gives `split_block_into_turns` word-accurate boundaries, which the current Parakeet path lacks (it returns no words, `retranscription.rs:582`).
- Wire into retranscription and import first; live recording later.
- Model storage `<app_data>/models/fluidaudio/{v3,ultra}`, about 0.5–0.6 GB each. First load compiles for the ANE (tens of seconds), so surface it through the existing download/compile progress toast.

### 4. Settings and onboarding UI

- `TranscriptSettings.tsx`: add provider `fluidAudio` (macOS only) with models "Parakeet v3" (default) and "Parakeet Ultra (Russian-only meetings; transliterates English terms)".
- A download manager modelled on `ParakeetModelManager.tsx`, with progress driven by the helper's progress lines through Tauri events.
- Readiness checks (`src/lib/transcription-model-readiness.ts`, `useRecordingStart.ts`) and the Retranscribe/Import model pickers (`RetranscribeDialog.tsx:159-163`, `ImportAudioDialog.tsx`).
- `LanguageSelection.tsx`: same as the current Parakeet (auto only).

### 5. Live recording

- Use the same persistent helper per VAD segment from `audio/transcription/worker.rs` (`:400` dispatch). At ~30–60x real time, a 5–10 s segment costs about 0.1–0.3 s.
- True streaming partials (FluidAudio's `SlidingWindowAsrManager`) are a separate follow-up.

### 6. Optional follow-ups

- Feed the approved glossary to FluidAudio's CTC custom-vocabulary rescoring. This is Parakeet's equivalent of Whisper's `initial_prompt`.
- Benchmark FluidAudio's Sortformer / Nemotron diarization against the current sherpa-onnx pipeline on the labeled boundary cases.
- Make FluidAudio Parakeet v3 the default on Apple Silicon once phases 3–5 have held up in daily use; consider the Parakeet + Whisper hybrid for code-switched segments.

## Acceptance criteria

- On all 6 clips (3 original + 3 gate clips), Parakeet v3's overall WER vs Scribe is within 1 point of the current engine (measured: 22.0% vs 21.8%), and English terms in code-switched speech stay in Latin script.
- One-hour retranscription is at least 5x faster end to end, including diarization.
- The boundary eval (`boundary-eval`, 66 labeled cases) does not regress when the turn text comes from Parakeet words instead of Whisper DTW.
- No regression on Windows/Linux builds: no sidecar is expected, and the provider is hidden.

## Risks

- Mid-sentence code-switching quality (phase 0 gate).
- FluidAudio API churn: pin the exact version; the helper isolates it from Rust.
- First-run ANE compile time and about 1.1 GB of models: make download and compile explicit and visible.
- A Swift toolchain in CI: GitHub macOS runners have Xcode; Windows/Linux jobs skip the helper.

## Phase 3 status

Implemented in `frontend/src-tauri/src`:

- `fluid_engine` (`src/fluid_engine/mod.rs`): `FluidModel` (`V3` / `Ultra`), `FluidAudioEngine` (`FLUID_ENGINE` singleton), `fluid-helper` sidecar resolution and lifecycle management (`load_model` with 600 s inactivity timeout and download/compile progress callbacks, `transcribe` via temp 16 kHz mono WAV with crash/idle-exit respawn + reload retry, `unload_model` with 2 s graceful shutdown), `tokens_to_words` SentencePiece merger into `WordSpan`s, `TranscriptionProvider` implementation, unit tests, and an `#[ignore]` integration test (`FLUID_TEST_WAV`).
- `BatchEngine::FluidAudio` and `LoadedBatchEngine::FluidAudio` in `src/audio/common.rs`, wired into batch retranscription (`src/audio/retranscription.rs`, including word timestamp shifting for speaker turn splitting) and audio import (`src/audio/import.rs`).
- Live recording fallback arms for `"fluidAudio"` in `src/audio/transcription/engine.rs` and `src/audio/recording_commands.rs` (logs a warning and uses Whisper until phase 5).

## Phase 4 status

Implemented across `fluid-helper/`, `frontend/src-tauri/src/`, and `frontend/src/`:

- Sidecar `status` request/response (`fluid-helper/Sources/FluidHelperCore/ProtocolTypes.swift`, `ProtocolRunner.swift`, `fluid-helper/Sources/FluidHelper/FluidAudioTranscriber.swift`, `fluid-helper/Tests/FluidHelperCoreTests/FluidHelperCoreTests.swift`, `fluid-helper/README.md`): queries `AsrModels.modelsExist(at:version:)` using the same directory resolution as `load`, without downloading or loading any model.
- Rust engine & Tauri commands (`frontend/src-tauri/src/fluid_engine/mod.rs`, `frontend/src-tauri/src/fluid_engine/commands.rs`, `frontend/src-tauri/src/lib.rs`): `FluidAudioEngine::model_status` (30 s timeout), `FluidAudioEngine::is_model_in_use`, `FluidAudioEngine::delete_model_files` (atomic in-use check, idle unload, status query, path verification, and directory removal under the process mutex), and Tauri commands `fluid_is_available`, `fluid_get_available_models`, `fluid_download_model` (rate-limited `fluid-model-download-progress`, `fluid-model-download-complete`, `fluid-model-download-error`, idle unload on both success and error), and `fluid_delete_model` (mutex-protected download/delete map guard + canonicalized path-prefix verification).
- Frontend UI (`frontend/src/components/FluidAudioModelManager.tsx`, `frontend/src/components/TranscriptSettings.tsx`, `frontend/src/hooks/useTranscriptionModels.ts`, `frontend/src/lib/transcription-model-readiness.ts`, `frontend/src/components/MeetingDetails/RetranscribeDialog.tsx`, `frontend/src/components/ImportAudio/ImportAudioDialog.tsx`): Neural Engine model manager section in Transcript settings when `fluid_is_available` is true (without adding FluidAudio to the global live provider dropdown), downloaded FluidAudio models in Retranscribe/Import model pickers (`🍎 Parakeet (Neural Engine): v3` / `Ultra`), and shared `isAutoLanguageOnlyProvider` gating for auto-language and LLM repair.

## Phase 5 status

Implemented across `frontend/src-tauri/src/` and `frontend/src/`:

- Live transcription provider (`frontend/src-tauri/src/fluid_engine/mod.rs`, `frontend/src-tauri/src/audio/transcription/engine.rs`, `frontend/src-tauri/src/audio/recording_commands.rs`): `FluidLiveProvider` wraps a `FluidJobGuard` acquired via `FLUID_ENGINE.begin_job` for the duration of a live recording (preventing concurrent model switches or mid-recording helper unload), maps `FluidTranscript` to `TranscriptResult` via `fluid_transcript_to_result`, and ignores language hints with a debug log. `validate_transcription_model_ready` and `get_or_init_transcription_engine` validate sidecar availability, resolve the configured model (defaulting empty to `fluid-parakeet-v3`), and verify `model_status(...).downloaded` without downloading during validation. `stop_recording` awaits transcription task completion (dropping `Arc<FluidLiveProvider>` and its `FluidJobGuard`) and calls `FLUID_ENGINE.unload_if_idle()`.
- Readiness commands & settings persistence (`frontend/src-tauri/src/fluid_engine/commands.rs`, `frontend/src-tauri/src/lib.rs`, `frontend/src-tauri/src/database/repositories/setting.rs`): added `fluid_init` and `fluid_has_available_models`, registered them in `lib.rs`, and updated `save_transcript_api_key` / `get_transcript_api_key` to treat `fluidAudio` as a keyless local provider alongside `parakeet`.
- Frontend provider selection & readiness (`frontend/src/lib/transcription-model-readiness.ts`, `frontend/src/constants/modelDefaults.ts`, `frontend/src/components/TranscriptSettings.tsx`, `frontend/src/components/FluidAudioModelManager.tsx`, `frontend/src/components/LanguageSelection.tsx`, `frontend/src/hooks/useModalState.ts`, `frontend/src/hooks/useTranscriptionModels.ts`): wired `fluidAudio` into `PROVIDER_COMMANDS` and `MODEL_DEFAULTS`, added `🍎 Parakeet on Neural Engine (macOS, fastest)` to the Transcript Model selector when `fluid_is_available` is true, added model selection (`selectedModel`, `onModelSelect`, `"Use for transcription"`, auto-save) to `FluidAudioModelManager`, hid the standalone Neural Engine section when `fluidAudio` is the active provider, and updated `LanguageSelection` to use `isAutoLanguageOnlyProvider`.
- Review fixes: stop always runs the FluidAudio idle unload (provider switched mid-recording, timed-out stop); `FluidJobGuard::transcribe` reloads the job's model under the process lock after a helper timeout/crash, so one failure no longer silences the rest of a live recording; FluidAudio validation errors are shown to the user and the worker no longer emits a second "Recording failed" when recording continues audio-only. Known, not fixed: `stop_recording` returns early if the final stream flush fails (pre-existing for all engines); a delete between validation and `begin_job` re-downloads the model.

## Acceptance results (2026-11-03)

Measured on four real meetings (34–69 min, Russian/English, 2–6 speakers) with `boundary-eval`'s `batchbench`, on an M-series Mac. Raw outputs stay in the gitignored `eval/private/bench/`.

- **WER**: Parakeet v3 22.0% vs current engine 21.8% (phase 0), within 1 point. Met.
- **Speed**: transcription alone is 11–13x faster end to end (Whisper large-v3-turbo 3–6x real time, Parakeet v3 50–98x). With the current sherpa-onnx diarization (CPU, 155–449 s per meeting) running in parallel, end to end is only **1.8–2.9x faster**: diarization is now the critical path. **Not met** with the current diarizer.
- **Boundary eval**: speaker-change scores are identical for Whisper and Parakeet turn text on all four meetings (ok-hit 13/24, wrong-rejected 37/44 for both). Met.
- **Windows/Linux**: `fluid_engine` reports unavailable off macOS and the UI hides it; the helper build steps are macOS-only. CI is the check.
- Fixed during the benchmark: FluidAudio rejects audio under 300 ms, which failed whole retranscriptions; short clips are now padded with trailing silence.

### Diarization benchmark (phase 6 item)

FluidAudio's offline diarizer (pyannote community-1 segmentation + WeSpeaker embeddings + VBx on the ANE), via `fluidaudiocli process --mode offline` on a 16 kHz mono mixdown:

| | current (sherpa-onnx, CPU) | FluidAudio offline (ANE) |
|---|---|---|
| time per meeting | 155–449 s | 9–24 s (~17x faster) |
| speakers found | 2 / 3 / 6 / 4 | 2 / 3 / 6 / 4 (same) |
| ok-hit (real changes found) | 13/24 | 15/24 |
| wrong-rejected (spurious changes avoided) | 37/44 | 36/44 |

With it, the end-to-end retranscription estimate is about 9–13x faster than today, which meets the 5x criterion. Notes: it must get a mono mixdown (on stereo input the CLI used one channel and found a single speaker); Sortformer was slower here (261 s on the 34-min meeting) and covered under half of the speech, so it is not a candidate.

**Recommendation**: add a `diarize` request to `fluid-helper` using `OfflineDiarizerManager` and use it on macOS in place of the sherpa-onnx helper, keeping the stereo local-speaker overlay and the existing Windows/Linux path.

### Other phase 6 items

- Glossary → CTC custom vocabulary: not started. It needs FluidAudio's separate CTC model download and a vocabulary request in the helper; worth doing after the diarizer, since the glossary matters less now that Parakeet keeps English terms in Latin script.
- Default switch: not yet. The plan's condition (phases 3–5 holding up in daily use) has not been met; revisit after a couple of weeks of real meetings on the FluidAudio provider.

## Diarization status

Implemented across `fluid-helper/`, `frontend/src-tauri/src/`, and `frontend/src/`:

- One-shot `fluid-helper diarize --audio <wav> --models-dir <dir> [--num-speakers N]` subcommand (`fluid-helper/Sources/FluidHelperCore/ProtocolTypes.swift`, `ProtocolRunner.swift`, `fluid-helper/Sources/FluidHelper/FluidAudioDiarizer.swift`, `FluidHelperMain.swift`, `fluid-helper/Tests/FluidHelperCoreTests/FluidHelperCoreTests.swift`, `fluid-helper/README.md`): runs in a separate short-lived process so diarization never contends with the persistent JSON-lines transcription session, loads/downloads offline diarizer models into `<models-dir>/speaker-diarization` via `OfflineDiarizerModels.load(from:)` / `OfflineDiarizerManager`, emits `PROGRESS <0-100>` and 2 s `PROGRESS -1` heartbeats on `stderr`, and writes a single `diarize-helper`-compatible JSON object (`segments`, `num_speakers`, L2-normalised duration-weighted `centroids`) to `stdout`.
- Rust diarization routing, fallback, and voiceprint isolation (`frontend/src-tauri/src/audio/diarization.rs`, `frontend/src-tauri/src/audio/retranscription.rs`): routes embedding model key `"fluidAudio"` to `fluid-helper diarize` when `fluid_engine::is_available()`, shares child process spawning/progress/timeout/cancellation handling with `diarize-helper`, falls back to sherpa-onnx (`"campplus"`) if FluidAudio is unavailable or fails, returns the effective model key from `diarize()` so `voiceprints::recognize_and_store` never mixes 256-d FluidAudio centroids with sherpa-onnx centroids, and lists `"fluidAudio"` ("Speaker voices — Neural Engine (macOS, fastest)") as ready in `diarization_list_models` when available.
- Frontend settings & retranscription dialog (`frontend/src/services/configService.ts`, `frontend/src/components/TranscriptSettings.tsx`, `frontend/src/components/MeetingDetails/RetranscribeDialog.tsx`): extended `EmbeddingModelOption` with `'fluidAudio'`, gated the Neural Engine speaker-model option on `fluid_is_available`, normalized saved `'fluidAudio'` settings to `'campplus'` when unavailable, and added helper text noting that switching models uses a separate voice library.
