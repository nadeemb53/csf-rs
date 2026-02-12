#!/bin/bash
# CFS Clean Script - Removes all cached databases

CFS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Cleaning all CFS databases..."

# Relay
rm -f "$CFS_ROOT/cfs_relay.db" && echo "  - Relay DB"

# macOS
rm -rf "$CFS_ROOT/apps/macos/src-tauri/.cfs" && echo "  - macOS .cfs"
rm -rf "$CFS_ROOT/.cfs" && echo "  - Root .cfs"

# iOS Simulator
find ~/Library/Developer/CoreSimulator/Devices -name "mobile_graph.db" -delete 2>/dev/null
find ~/Library/Developer/CoreSimulator/Devices -type d -name ".cfs" -exec rm -rf {} + 2>/dev/null
echo "  - iOS Simulator databases"

echo "Done! All databases cleared."
