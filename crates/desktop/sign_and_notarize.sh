#!/usr/bin/env bash
set -euo pipefail

# Sign, package, and notarize the FaceGuard macOS app.
#
# Prerequisites:
#   1. Run build_app.sh --release first to create the .app bundle.
#   2. A valid "Developer ID Application" signing identity in your keychain.
#   3. A notarytool keychain profile named "notarytool-profile".
#      Set it up once with:
#        xcrun notarytool store-credentials "notarytool-profile" \
#            --apple-id <your-apple-id> --team-id S79EWNF9K2 --password <app-specific-password>
#
# Usage:
#   ./sign_and_notarize.sh              # sign, DMG, notarize
#   ./sign_and_notarize.sh --no-notarize  # sign + DMG only (for local testing)

# --- Configuration ---
APP_NAME="FaceGuard"
SIGNING_IDENTITY="Developer ID Application: Neutrino Graphics LLC (S79EWNF9K2)"
BUNDLE_ID="com.da1nerd.faceguard"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE_DIR="$WORKSPACE_DIR/target/bundle"
APP_PATH="$BUNDLE_DIR/$APP_NAME.app"
DMG_PATH="$BUNDLE_DIR/$APP_NAME.dmg"

NOTARIZE=true
if [[ "${1:-}" == "--no-notarize" ]]; then
    NOTARIZE=false
fi

# --- Preflight checks ---
if [[ ! -d "$APP_PATH" ]]; then
    echo "ERROR: App bundle not found at $APP_PATH"
    echo "       Run ./build_app.sh --release first."
    exit 1
fi

# Verify signing identity exists
if ! security find-identity -v -p codesigning | grep -q "$SIGNING_IDENTITY"; then
    echo "ERROR: Signing identity not found: $SIGNING_IDENTITY"
    echo "       Make sure your Developer ID certificate is installed."
    exit 1
fi

# --- Step 1: Sign all binaries (inside-out) ---
echo "==> Signing app bundle..."

# Sign frameworks and embedded dylibs first
find "$APP_PATH/Contents/Frameworks" "$APP_PATH/Contents/Resources" \
    -type f \( -name "*.so" -o -name "*.dylib" -o -name "*.framework" -o -perm +111 \) \
    -exec codesign --force --options runtime --timestamp \
    --sign "$SIGNING_IDENTITY" {} \; 2>/dev/null || true

# Sign the main executable
codesign --force --options runtime --timestamp \
    --sign "$SIGNING_IDENTITY" "$APP_PATH/Contents/MacOS/faceguard-desktop"

# Sign the app bundle itself
codesign --force --deep --options runtime --timestamp \
    --sign "$SIGNING_IDENTITY" "$APP_PATH"

echo "==> Verifying signature..."
codesign --verify --deep --strict "$APP_PATH"
echo "    Signature OK"

# --- Step 2: Create DMG ---
echo "==> Creating DMG..."
rm -f "$DMG_PATH"

# Stage the .app and an Applications shortcut in a temp folder
DMG_STAGING="$BUNDLE_DIR/dmg_staging"
rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
cp -R "$APP_PATH" "$DMG_STAGING/"

# Create a read-write DMG first so we can configure the Finder view
DMG_RW="$BUNDLE_DIR/${APP_NAME}_rw.dmg"
rm -f "$DMG_RW"
hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_STAGING" \
    -ov -format UDRW "$DMG_RW"
rm -rf "$DMG_STAGING"

# Mount the read-write DMG and configure Finder layout
MOUNT_DIR="/Volumes/$APP_NAME"
hdiutil attach "$DMG_RW" -mountpoint "$MOUNT_DIR" -noverify

# Create a Finder alias to /Applications (aliases show the target's icon)
osascript <<'ALIAS_SCRIPT'
tell application "Finder"
    make new alias file at POSIX file "/Volumes/FaceGuard" to POSIX file "/Applications" with properties {name:"Applications"}
end tell
ALIAS_SCRIPT

osascript <<APPLESCRIPT
tell application "Finder"
    tell disk "$APP_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {100, 100, 640, 400}
        set theViewOptions to icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 80
        set position of item "$APP_NAME.app" of container window to {140, 150}
        set position of item "Applications" of container window to {400, 150}
        close
        open
        close
    end tell
end tell
APPLESCRIPT

# Give Finder time to write .DS_Store
sync
sleep 3

# Detach: Finder eject first (also unmounts), fall back to hdiutil
if [ -d "$MOUNT_DIR" ]; then
    osascript -e "tell application \"Finder\" to eject disk \"$APP_NAME\"" 2>/dev/null || true
    sleep 2
fi
if [ -d "$MOUNT_DIR" ]; then
    hdiutil detach "$MOUNT_DIR" -force
fi

# Convert to compressed read-only DMG
hdiutil convert "$DMG_RW" -format UDZO -o "$DMG_PATH"
rm -f "$DMG_RW"

# Sign the DMG too
codesign --force --timestamp --sign "$SIGNING_IDENTITY" "$DMG_PATH"

# --- Step 3: Notarize ---
if [[ "$NOTARIZE" == true ]]; then
    echo "==> Submitting for notarization..."
    echo "    (This may take a few minutes)"
    xcrun notarytool submit "$DMG_PATH" \
        --keychain-profile "notarytool-profile" \
        --wait

    echo "==> Stapling notarization ticket..."
    xcrun stapler staple "$DMG_PATH"
else
    echo "==> Skipping notarization (--no-notarize)"
fi

echo ""
echo "==> Done! Distributable DMG: $DMG_PATH"
echo "    Size: $(du -h "$DMG_PATH" | cut -f1)"
