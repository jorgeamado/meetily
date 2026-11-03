#!/usr/bin/env bash
# Convert an audio file to a 16 kHz mono PCM16 WAV in a temporary directory and
# run diarize-helper with the default segmentation and speaker-embedding models.
#
# Usage:
#   eval/run_diarize_helper.sh <audio> <out.json>

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <audio> <out.json>" >&2
    exit 2
fi

AUDIO_IN="$1"
OUT_JSON="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOST_TRIPLE="$(rustc -vV | awk '/^host:/ { print $2 }')"

DIARIZE_BIN="${MEETILY_DIARIZE_HELPER:-$REPO_ROOT/frontend/src-tauri/binaries/diarize-helper-$HOST_TRIPLE}"

MODELS_DIR="${MEETILY_MODELS_DIR:-$HOME/Library/Application Support/com.meetily.ai/models}"
SEGMENTATION_MODEL="$MODELS_DIR/diarization/segmentation-3.0.onnx"
# DEFAULT_EMBEDDING_MODEL = "campplus" -> "campplus-zh-en-advanced-16k.onnx" (from audio/diarization.rs)
EMBEDDING_MODEL="$MODELS_DIR/diarization/campplus-zh-en-advanced-16k.onnx"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

WAV_PATH="$TMP_DIR/audio_16k_mono.wav"

ffmpeg -hide_banner -loglevel error -y \
    -i "$AUDIO_IN" \
    -vn -ac 1 -ar 16000 -c:a pcm_s16le \
    "$WAV_PATH"

mkdir -p "$(dirname "$OUT_JSON")"

START_TS="$(python3 -c 'import time; print(f"{time.time():.6f}")')"
"$DIARIZE_BIN" \
    --audio "$WAV_PATH" \
    --segmentation-model "$SEGMENTATION_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    > "$OUT_JSON"
END_TS="$(python3 -c 'import time; print(f"{time.time():.6f}")')"

ELAPSED_SECS="$(python3 -c "print(f'{float(\"$END_TS\") - float(\"$START_TS\"):.2f}')")"
echo "diarize-helper elapsed: ${ELAPSED_SECS}s" >&2
