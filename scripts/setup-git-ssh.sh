#!/bin/bash

##############################################################################
# Git SSH Setup for RunPod
#
# This script configures SSH for git operations on RunPod instances.
# Run this ONCE when you start a new RunPod instance.
#
# Usage:
#   bash scripts/setup-git-ssh.sh
##############################################################################

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Git SSH Setup for RunPod                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Project: ${PROJECT_ROOT}"
echo ""

##############################################################################
# Step 1: Setup SSH Directory
##############################################################################

echo "Setting up SSH directory..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Copy project SSH key to user SSH directory
if [ -f ".ssh/runpod_key" ]; then
    cp .ssh/runpod_key ~/.ssh/runpod_unlearning
    cp .ssh/runpod_key.pub ~/.ssh/runpod_unlearning.pub
    chmod 600 ~/.ssh/runpod_unlearning
    chmod 644 ~/.ssh/runpod_unlearning.pub
    echo "✓ SSH keys copied to ~/.ssh/"
else
    echo "✗ SSH key not found in .ssh/runpod_key"
    echo "  Please make sure you've cloned the repo with the SSH key"
    exit 1
fi

##############################################################################
# Step 2: Configure SSH for GitHub
##############################################################################

echo ""
echo "Configuring SSH for GitHub..."

# Create or update SSH config
SSH_CONFIG=~/.ssh/config

# Check if GitHub entry already exists
if grep -q "Host github.com-unlearning" "$SSH_CONFIG" 2>/dev/null; then
    echo "⚠️  SSH config for github.com-unlearning already exists, skipping..."
else
    cat >> "$SSH_CONFIG" << 'EOF'

# RunPod Unlearning Project
Host github.com-unlearning
    HostName github.com
    User git
    IdentityFile ~/.ssh/runpod_unlearning
    IdentitiesOnly yes
EOF
    chmod 600 "$SSH_CONFIG"
    echo "✓ SSH config updated"
fi

##############################################################################
# Step 3: Start SSH Agent and Add Key
##############################################################################

echo ""
echo "Adding SSH key to agent..."

# Start ssh-agent if not running
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)"
fi

# Add the key
ssh-add ~/.ssh/runpod_unlearning
echo "✓ SSH key added to agent"

##############################################################################
# Step 4: Configure Git Remote
##############################################################################

echo ""
echo "Configuring git remote..."

# Get current remote URL
CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")

if [[ "$CURRENT_REMOTE" == *"github.com-unlearning"* ]]; then
    echo "✓ Git remote already configured for SSH"
elif [[ "$CURRENT_REMOTE" == *"https://github.com"* ]]; then
    # Convert HTTPS to SSH using our custom host
    NEW_REMOTE=$(echo "$CURRENT_REMOTE" | sed 's|https://github.com/|git@github.com-unlearning:|')
    git remote set-url origin "$NEW_REMOTE"
    echo "✓ Git remote updated from HTTPS to SSH"
    echo "  New remote: $NEW_REMOTE"
elif [[ "$CURRENT_REMOTE" == *"git@github.com:"* ]]; then
    # Convert standard SSH to custom host
    NEW_REMOTE=$(echo "$CURRENT_REMOTE" | sed 's|git@github.com:|git@github.com-unlearning:|')
    git remote set-url origin "$NEW_REMOTE"
    echo "✓ Git remote updated to use custom SSH host"
    echo "  New remote: $NEW_REMOTE"
else
    echo "⚠️  Unexpected remote format: $CURRENT_REMOTE"
    echo "  Please configure manually"
fi

##############################################################################
# Step 5: Test Connection
##############################################################################

echo ""
echo "Testing SSH connection to GitHub..."
if ssh -T git@github.com-unlearning 2>&1 | grep -q "successfully authenticated"; then
    echo "✓ SSH connection successful!"
else
    echo ""
    echo "Connection test output:"
    ssh -T git@github.com-unlearning 2>&1 || true
    echo ""
    echo "⚠️  If you see 'Permission denied', make sure you've added the SSH key to GitHub:"
    echo "   https://github.com/settings/keys"
fi

##############################################################################
# Step 6: Configure Git User (Optional)
##############################################################################

echo ""
echo "Configuring git user for this repository..."

# Check if git user is already set globally
GLOBAL_USER=$(git config --global user.name 2>/dev/null || echo "")
GLOBAL_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [ -z "$GLOBAL_USER" ]; then
    read -p "Enter your Git username: " GIT_USER
    git config user.name "$GIT_USER"
    echo "✓ Git user.name set to: $GIT_USER (local)"
else
    echo "✓ Using global git user: $GLOBAL_USER"
fi

if [ -z "$GLOBAL_EMAIL" ]; then
    read -p "Enter your Git email: " GIT_EMAIL
    git config user.email "$GIT_EMAIL"
    echo "✓ Git user.email set to: $GIT_EMAIL (local)"
else
    echo "✓ Using global git email: $GLOBAL_EMAIL"
fi

##############################################################################
# Summary
##############################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now use git commands:"
echo "  git pull"
echo "  git add ."
echo "  git commit -m 'Your message'"
echo "  git push"
echo ""
echo "Current git remote:"
git remote -v
echo ""
