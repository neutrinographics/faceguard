#!/usr/bin/env bash
# Build a macOS .app bundle for Video Blur.
# Usage: ./build_app.sh [--release]
#
# Builds with static FFmpeg linking (no dylib bundling needed).
# Icons and Info.plist are bundled into a self-contained .app.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_NAME="Video Blur"
BUNDLE_DIR="$WORKSPACE_DIR/target/bundle"
APP_DIR="$BUNDLE_DIR/$APP_NAME.app"
FRAMEWORKS_DIR="$APP_DIR/Contents/Frameworks"

PROFILE="debug"
CARGO_FLAGS=""
if [[ "${1:-}" == "--release" ]]; then
    PROFILE="release"
    CARGO_FLAGS="--release"
fi

echo "Building video-blur-desktop ($PROFILE) with static FFmpeg..."
cargo build -p video-blur-desktop $CARGO_FLAGS --features static-ffmpeg

echo "Creating app bundle..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
mkdir -p "$FRAMEWORKS_DIR"

BINARY="$WORKSPACE_DIR/target/$PROFILE/video-blur-desktop"

# Copy binary
cp "$BINARY" "$APP_DIR/Contents/MacOS/"

# Copy Info.plist and icon
cp "$SCRIPT_DIR/assets/Info.plist" "$APP_DIR/Contents/"
cp "$SCRIPT_DIR/assets/VideoBlur.icns" "$APP_DIR/Contents/Resources/"

# ---------------------------------------------------------------------------
# Bundle any remaining non-system dylibs (e.g. ort/onnxruntime)
# With static FFmpeg, most dylibs are eliminated, but other deps may remain.
# ---------------------------------------------------------------------------

# File to track already-processed dylibs
PROCESSED_FILE="$(mktemp)"
trap "rm -f '$PROCESSED_FILE'" EXIT

collect_dylibs() {
    local binary="$1"
    otool -L "$binary" 2>/dev/null \
        | awk 'NR>1 {print $1}' \
        | grep -v '^/System' \
        | grep -v '^/usr/lib' \
        | grep -v '@rpath' \
        | grep -v '@executable_path' \
        || true
}

bundle_dylib() {
    local dylib_path="$1"

    # Resolve symlinks to get the actual file
    local real_path
    real_path="$(python3 -c "import os; print(os.path.realpath('$dylib_path'))" 2>/dev/null || echo "$dylib_path")"

    # Skip if already processed
    if grep -qxF "$real_path" "$PROCESSED_FILE" 2>/dev/null; then
        return
    fi
    echo "$real_path" >> "$PROCESSED_FILE"

    if [[ ! -f "$real_path" ]]; then
        echo "  WARNING: dylib not found: $dylib_path"
        return
    fi

    local basename
    basename="$(basename "$dylib_path")"

    # Skip if already copied
    if [[ -f "$FRAMEWORKS_DIR/$basename" ]]; then
        return
    fi

    echo "  Bundling: $basename"

    # Copy into Frameworks
    cp "$real_path" "$FRAMEWORKS_DIR/$basename"
    chmod 755 "$FRAMEWORKS_DIR/$basename"

    # Update the install name to use @rpath
    install_name_tool -id "@rpath/$basename" "$FRAMEWORKS_DIR/$basename" 2>/dev/null || true

    # Recursively process this dylib's dependencies
    local deps
    deps="$(collect_dylibs "$FRAMEWORKS_DIR/$basename")"
    local dep
    for dep in $deps; do
        local dep_basename
        dep_basename="$(basename "$dep")"
        # Rewrite reference to point to @rpath
        install_name_tool -change "$dep" "@rpath/$dep_basename" "$FRAMEWORKS_DIR/$basename" 2>/dev/null || true
        # Recurse
        bundle_dylib "$dep"
    done
}

echo "Checking for non-system dylibs..."
BINARY_IN_APP="$APP_DIR/Contents/MacOS/video-blur-desktop"

# Get direct dependencies of the binary
DIRECT_DEPS="$(collect_dylibs "$BINARY_IN_APP")"
for dep in $DIRECT_DEPS; do
    dep_basename="$(basename "$dep")"
    bundle_dylib "$dep"
    # Rewrite binary's reference to @rpath
    install_name_tool -change "$dep" "@rpath/$dep_basename" "$BINARY_IN_APP" 2>/dev/null || true
done

# Add @executable_path/../Frameworks to rpath
install_name_tool -add_rpath "@executable_path/../Frameworks" "$BINARY_IN_APP" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Codesign
# ---------------------------------------------------------------------------

echo "Signing app bundle..."
codesign --deep --force --sign - "$APP_DIR"

DYLIB_COUNT="$(ls -1 "$FRAMEWORKS_DIR" 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "App bundle created at: $APP_DIR"
echo "Bundled $DYLIB_COUNT dylibs into Frameworks/"
echo "To run: open \"$APP_DIR\""
