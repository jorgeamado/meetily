#!/bin/bash
# Build the fluid-helper sidecar (FluidAudio Parakeet on the Neural Engine, Swift)
# and stage it as src-tauri/binaries/fluid-helper-<target-triple>.
# macOS only; a no-op elsewhere. Called by build-gpu.sh, dev-gpu.sh,
# clean_build.sh, clean_run.sh and the macOS CI jobs.
set -e

if [[ "$OSTYPE" != "darwin"* ]]; then
    exit 0
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FRONTEND_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
FLUID_HELPER_DIR="$( cd "$FRONTEND_DIR/../fluid-helper" && pwd )"
BINARIES_DIR="$FRONTEND_DIR/src-tauri/binaries"
TARGET_TRIPLE="${1:-$(rustc -vV | grep "host:" | awk '{print $2}')}"

# Build for the requested target, not just the host, so the staged binary
# always matches the -<triple> suffix Tauri bundles it under
case "$TARGET_TRIPLE" in
    aarch64-apple-darwin) ARCH=arm64 ;;
    x86_64-apple-darwin) ARCH=x86_64 ;;
    *) echo "Unsupported fluid-helper target: $TARGET_TRIPLE" >&2; exit 1 ;;
esac
SWIFT_ARGS=(-c release --product fluid-helper --arch "$ARCH")

echo "Building fluid-helper sidecar (release, Swift, $ARCH)..."
if ! (cd "$FLUID_HELPER_DIR" && swift build "${SWIFT_ARGS[@]}"); then
    echo "Failed to build fluid-helper (needs Xcode / Swift 6)" >&2
    exit 1
fi

SRC_PATH="$(cd "$FLUID_HELPER_DIR" && swift build "${SWIFT_ARGS[@]}" --show-bin-path)/fluid-helper"
if [ ! -f "$SRC_PATH" ]; then
    echo "fluid-helper binary not found at $SRC_PATH" >&2
    exit 1
fi
if ! lipo -archs "$SRC_PATH" | grep -qw "$ARCH"; then
    echo "fluid-helper at $SRC_PATH is not built for $ARCH ($(lipo -archs "$SRC_PATH"))" >&2
    exit 1
fi

mkdir -p "$BINARIES_DIR"
find "$BINARIES_DIR" -name "fluid-helper*" -delete
cp "$SRC_PATH" "$BINARIES_DIR/fluid-helper-$TARGET_TRIPLE"
echo "Copied fluid-helper to $BINARIES_DIR/fluid-helper-$TARGET_TRIPLE"
