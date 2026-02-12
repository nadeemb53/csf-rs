#!/bin/bash
set -e

# CFS Development Script
# Clears all cached data and starts fresh development environment

CFS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CFS_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CFS Development Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
BUILD_IOS=false
START_RELAY=true
START_MACOS=true
CLEAN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-ios)
            BUILD_IOS=true
            shift
            ;;
        --no-relay)
            START_RELAY=false
            shift
            ;;
        --no-macos)
            START_MACOS=false
            shift
            ;;
        --clean-only)
            CLEAN_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --build-ios   Build iOS library before starting"
            echo "  --no-relay    Don't start the relay server"
            echo "  --no-macos    Don't start the macOS app"
            echo "  --clean-only  Only clean databases, don't start anything"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Clean all databases
echo -e "${YELLOW}[1/5] Cleaning all databases...${NC}"

# Clean relay database
if [ -f "$CFS_ROOT/cfs_relay.db" ]; then
    rm -f "$CFS_ROOT/cfs_relay.db"
    echo "  - Deleted relay database"
else
    echo "  - Relay database not found (clean)"
fi

# Clean macOS app database
if [ -d "$CFS_ROOT/apps/macos/src-tauri/.cfs" ]; then
    rm -rf "$CFS_ROOT/apps/macos/src-tauri/.cfs"
    echo "  - Deleted macOS .cfs directory"
else
    echo "  - macOS .cfs not found (clean)"
fi

# Clean any root .cfs directory
if [ -d "$CFS_ROOT/.cfs" ]; then
    rm -rf "$CFS_ROOT/.cfs"
    echo "  - Deleted root .cfs directory"
fi

# Clean iOS simulator databases
echo "  - Cleaning iOS Simulator databases..."
IOS_CLEANED=0
while IFS= read -r -d '' db; do
    rm -f "$db"
    ((IOS_CLEANED++))
done < <(find ~/Library/Developer/CoreSimulator/Devices -name "mobile_graph.db" -print0 2>/dev/null)

while IFS= read -r -d '' dir; do
    rm -rf "$dir"
    ((IOS_CLEANED++))
done < <(find ~/Library/Developer/CoreSimulator/Devices -type d -name ".cfs" -print0 2>/dev/null)

if [ $IOS_CLEANED -gt 0 ]; then
    echo "  - Cleaned $IOS_CLEANED iOS Simulator items"
else
    echo "  - iOS Simulator databases not found (clean)"
fi

echo -e "${GREEN}  All databases cleaned!${NC}"
echo ""

if [ "$CLEAN_ONLY" = true ]; then
    echo -e "${GREEN}Clean complete. Exiting (--clean-only specified).${NC}"
    exit 0
fi

# Step 2: Build iOS if requested
if [ "$BUILD_IOS" = true ]; then
    echo -e "${YELLOW}[2/5] Building iOS library...${NC}"
    "$CFS_ROOT/scripts/build_ios.sh" sim
    echo ""
else
    echo -e "${BLUE}[2/5] Skipping iOS build (use --build-ios to enable)${NC}"
    echo ""
fi

# Step 3: Build macOS app
echo -e "${YELLOW}[3/5] Building macOS app...${NC}"
cd "$CFS_ROOT/apps/macos"
cargo build --release 2>&1 | grep -E "(Compiling|Finished|error)" || true
cd "$CFS_ROOT"
echo -e "${GREEN}  macOS app built!${NC}"
echo ""

# Step 4: Start relay server in background
if [ "$START_RELAY" = true ]; then
    echo -e "${YELLOW}[4/5] Starting relay server...${NC}"

    # Kill any existing relay server
    pkill -f "cfs-relay-server" 2>/dev/null || true
    sleep 1

    # Start relay server
    cd "$CFS_ROOT/relay/cfs-relay-server"
    cargo run --release &
    RELAY_PID=$!
    cd "$CFS_ROOT"

    # Wait for relay to start
    sleep 2
    if kill -0 $RELAY_PID 2>/dev/null; then
        echo -e "${GREEN}  Relay server started (PID: $RELAY_PID)${NC}"
    else
        echo -e "${RED}  Failed to start relay server${NC}"
    fi
    echo ""
else
    echo -e "${BLUE}[4/5] Skipping relay server (--no-relay specified)${NC}"
    echo ""
fi

# Step 5: Start macOS app
if [ "$START_MACOS" = true ]; then
    echo -e "${YELLOW}[5/5] Starting macOS app...${NC}"

    # Kill any existing Tauri app
    pkill -f "macos-app" 2>/dev/null || true
    sleep 1

    cd "$CFS_ROOT/apps/macos"
    cargo tauri dev &
    MACOS_PID=$!
    cd "$CFS_ROOT"

    echo -e "${GREEN}  macOS app starting (PID: $MACOS_PID)${NC}"
    echo ""
else
    echo -e "${BLUE}[5/5] Skipping macOS app (--no-macos specified)${NC}"
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Development environment ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. In macOS app: Select test_corpus folder to watch"
echo "  2. In Xcode: Run CFSMobile on iOS Simulator"
echo "  3. In iOS app: Tap 'Sync' to pull data from relay"
echo ""
echo "To stop all processes:"
echo "  pkill -f 'cfs-relay-server'; pkill -f 'macos-app'"
echo ""

# Keep script running to show logs
if [ "$START_RELAY" = true ] || [ "$START_MACOS" = true ]; then
    echo "Press Ctrl+C to stop..."
    wait
fi
