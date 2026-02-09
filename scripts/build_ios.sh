#!/bin/bash
set -e

# CFS iOS Build Script
# This builds the Rust core as a static library for iOS.
# Includes llama.cpp for local AI inference.

BUILD_TARGET="${1:-sim}"

case "$BUILD_TARGET" in
    sim|simulator)
        echo "Building cfs-mobile for iOS Simulator (aarch64)..."
        rustup target add aarch64-apple-ios-sim
        cargo build -p cfs-mobile --target aarch64-apple-ios-sim --release
        LIB_PATH="target/aarch64-apple-ios-sim/release/libcfs_mobile.a"
        ;;
    device|ios)
        echo "Building cfs-mobile for iOS Device (aarch64)..."
        rustup target add aarch64-apple-ios
        cargo build -p cfs-mobile --target aarch64-apple-ios --release
        LIB_PATH="target/aarch64-apple-ios/release/libcfs_mobile.a"
        ;;
    *)
        echo "Usage: $0 [sim|device]"
        echo "  sim    - Build for iOS Simulator (default)"
        echo "  device - Build for iOS Device"
        exit 1
        ;;
esac

echo "Build complete."

# Auto-copy to apps/ios/CFSMobile if it exists
if [ -d "apps/ios/CFSMobile" ]; then
    echo "Updating local copy in apps/ios/CFSMobile/..."
    rsync -av "$LIB_PATH" apps/ios/CFSMobile/

    if [ -f "apps/ios/CFSMobile/libcfs_mobile.a" ]; then
        echo "Successfully verified libcfs_mobile.a in apps/ios/CFSMobile/"
        ls -l apps/ios/CFSMobile/libcfs_mobile.a
    else
        echo "ERROR: Failed to find libcfs_mobile.a in target directory after sync!"
        exit 1
    fi
fi

echo ""
echo "The static library is at: $LIB_PATH"
echo ""
echo "Next steps in Xcode:"
echo "1. Open apps/ios/CFSMobile.xcodeproj in Xcode"
echo "2. Run on an iPhone Simulator (or device if you built for device)!"
echo ""
echo "For AI features, you need a GGUF model file:"
echo "  - Download SmolLM2-135M-Instruct (recommended for mobile):"
echo "    https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF"
echo "  - Place 'smollm2.gguf' in the app's Documents folder or bundle it"
