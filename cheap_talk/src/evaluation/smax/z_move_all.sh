#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the parent folder as one level up from the script directory
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Create the destination folder "z_all" in the script's directory
DEST_DIR="$SCRIPT_DIR/z_all"
mkdir -p "$DEST_DIR"

# Recursively find *.jpg and *.pdf files in the child folders of the parent folder.
# Using -mindepth 2 ensures we only look inside subfolders (child folders), not in the parent folder itself.
find "$PARENT_DIR" -mindepth 2 -type f \( -iname "*.jpg" -o -iname "*.pdf" \) -exec cp {} "$DEST_DIR" \;

echo "Files copied to $DEST_DIR"