# fluid-helper

`fluid-helper` is a macOS command-line sidecar for Meetily that runs FluidAudio's Parakeet TDT speech-to-text models (`v3` and `ultra`) on Apple Silicon via CoreML and the Apple Neural Engine (ANE). It communicates with the Rust host process using line-delimited JSON over `stdin` and `stdout`, while sending all diagnostic logs to `stderr`.

## Build

```bash
swift build -c release --product fluid-helper
```

Run unit tests for the protocol and session layer (`FluidHelperCore`, no FluidAudio build or model downloads required):

```bash
swift test --filter FluidHelperCoreTests
```

## Process Model & Protocol Summary

- **Transport**: Persistent process reading one UTF-8 JSON object per line from `stdin` and writing one UTF-8 JSON object per line to the original `stdout` descriptor (unbuffered per line). At startup, `STDOUT_FILENO` (`fd 1`) is duplicated for the JSON channel and `fd 1` is redirected to `STDERR_FILENO` (`fd 2`), ensuring any output from `print()`, `FileHandle.standardOutput`, or C `stdout` goes to `stderr` in both persistent and one-shot modes.
- **Concurrency**: Requests are processed sequentially, one at a time, to avoid concurrent CoreML/ANE execution.
- **Lifecycle**: Exits cleanly (`0`) on `stdin` EOF, a `shutdown` request, or after `FLUID_HELPER_IDLE_SECS` seconds of inactivity (default `300`).

### Requests and Responses

1. **Load model**
   - Request:
     ```json
     {"type":"load","version":"v3","models_dir":"/abs/path"}
     ```
     (`"version"` may be `"v3"` or `"ultra"`. Models live in FluidAudio's repository layout at `<models_dir>/<repo.folderName>` — e.g. `parakeet-tdt-0.6b-v3`, `parakeet-ultra` / `parakeet-ultra-coreml` — without creating any `<models_dir>/<version>` directory.)
   - Progress lines (emitted during download and CoreML compilation; each stage has its own monotonic non-decreasing fraction in `[0, 0.99]` while work is in progress, `compile` starts at `0` and never switches back to `download`, rate-limited, with a final `compile` `1.0` emitted right before `loaded`):
     ```json
     {"type":"progress","stage":"download","fraction":0.45}
     {"type":"progress","stage":"compile","fraction":0.75}
     {"type":"progress","stage":"compile","fraction":1.0}
     ```
   - Completion response:
     ```json
     {"type":"loaded","version":"v3","load_secs":12.3}
     ```
     Loading an already-loaded version is a fast no-op that still responds with `loaded`. Loading a different version replaces the currently loaded model.

2. **Transcribe WAV file**
   - Request:
     ```json
     {"type":"transcribe","id":7,"wav_path":"/abs/file.wav"}
     ```
     Optional field `"tokens": false` omits the `"tokens"` array from the response (defaults to `true`).
   - Response:
     ```json
     {"type":"result","id":7,"text":"...","confidence":0.93,"duration":120.1,"processing_secs":1.6,"words":[{"w":"forecast","start":12.34,"end":12.8}],"tokens":[{"t":"▁fore","start":12.34,"end":12.5,"conf":0.98}]}
     ```
      FluidAudio's `language` script filter is intentionally omitted so mixed Cyrillic/Latin speech is preserved.

3. **Ping**
   - Request: `{"type":"ping"}`
   - Response: `{"type":"pong"}`

4. **Shutdown**
   - Request: `{"type":"shutdown"}`
   - Response: `{"type":"goodbye"}` (then exits `0`)

5. **Errors**
   - Any request or runtime failure emits:
     ```json
     {"type":"error","id":7,"message":"..."}
     ```
     (`"id"` is `null` for malformed JSON or requests without an `id`.) The process remains alive for subsequent requests.

## One-Shot Mode

For direct CLI testing without the `stdin`/`stdout` protocol loop:

```bash
fluid-helper --transcribe <wav> [--version v3|ultra] [--models-dir <dir>]
```

- `--version`: `v3` (default) or `ultra`
- `--models-dir`: defaults to `~/Library/Application Support/com.meetily.ai/models/fluidaudio`

Loads the requested model, transcribes `<wav>`, prints a single `result` JSON object to `stdout`, and exits.

## Licences

- **FluidAudio**: [Apache License 2.0](https://github.com/FluidInference/FluidAudio/blob/main/LICENSE) (`Apache-2.0`).
- **Parakeet v3 / Parakeet Ultra CoreML Models**: Creative Commons Attribution 4.0 International (`CC-BY-4.0`). Attribution is required when redistributing or surfacing model information in the application UI.
