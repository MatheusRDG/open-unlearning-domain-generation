#!/bin/bash
# Restore missing config files from git

set -e

echo "==================================="
echo "Restoring Config Files from Git"
echo "==================================="
echo ""

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    echo "Please run this script from the repository root"
    exit 1
fi

echo "Step 1: Checking current status..."
echo "Files in configs/:"
ls -la configs/ 2>&1 || echo "configs/ directory not found or empty"
echo ""

echo "Step 2: Restoring config files from git..."
git checkout HEAD -- configs/

echo ""
echo "Step 3: Verifying restoration..."
echo ""

if [ -f "configs/unlearn.yaml" ]; then
    echo "✓ configs/unlearn.yaml restored"
else
    echo "✗ configs/unlearn.yaml still missing"
fi

if [ -f "configs/train.yaml" ]; then
    echo "✓ configs/train.yaml restored"
else
    echo "✗ configs/train.yaml still missing"
fi

if [ -f "configs/eval.yaml" ]; then
    echo "✓ configs/eval.yaml restored"
else
    echo "✗ configs/eval.yaml still missing"
fi

echo ""
echo "All config files in configs/:"
ls -la configs/*.yaml 2>&1 || echo "No yaml files found"

echo ""
echo "Config subdirectories:"
ls -d configs/*/ 2>&1 || echo "No subdirectories found"

echo ""
echo "==================================="
echo "Restoration Complete!"
echo "==================================="
