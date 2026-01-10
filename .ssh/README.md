# SSH Key for RunPod

This directory contains the SSH key pair for pushing to GitHub from RunPod instances.

## ⚠️ SECURITY NOTE

**DO NOT commit private keys to public repositories!**

This key is for this specific project only and should be:
- Added to `.gitignore` if the repo is public
- Rotated regularly
- Removed from GitHub when no longer needed

## Setup Instructions

### 1. Add Public Key to GitHub

Copy the public key:
```bash
cat .ssh/runpod_key.pub
```

Then add it to GitHub:
1. Go to https://github.com/settings/keys
2. Click "New SSH key"
3. Title: `RunPod - Unlearning Project`
4. Key: Paste the public key from above
5. Click "Add SSH key"

### 2. On RunPod Instance

After cloning the repository on RunPod, run:

```bash
bash scripts/setup-git-ssh.sh
```

This will:
- Copy SSH keys to `~/.ssh/`
- Configure SSH for GitHub
- Add key to SSH agent
- Update git remote to use SSH
- Test the connection

### 3. Verify Setup

Test SSH connection:
```bash
ssh -T git@github.com-unlearning
```

You should see:
```
Hi YourUsername! You've successfully authenticated, but GitHub does not provide shell access.
```

### 4. Use Git Normally

```bash
git pull
git add .
git commit -m "Update from RunPod"
git push
```

## Files

- `runpod_key` - Private key (keep secret!)
- `runpod_key.pub` - Public key (safe to share)

## Troubleshooting

### Permission Denied

If you get "Permission denied (publickey)":
1. Make sure you've added the public key to GitHub
2. Check key permissions: `ls -la ~/.ssh/runpod_unlearning`
3. Should be `-rw-------` (600)

### Wrong Remote URL

Check current remote:
```bash
git remote -v
```

Should show:
```
origin  git@github.com-unlearning:MatheusRDG/open-unlearning-domain-generation.git
```

Fix if needed:
```bash
git remote set-url origin git@github.com-unlearning:MatheusRDG/open-unlearning-domain-generation.git
```

### Key Not Found

Make sure to run the setup script:
```bash
bash scripts/setup-git-ssh.sh
```
