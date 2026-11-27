#!/bin/bash
# Runs once, when the image is built. No secrets or /data here: anything that
# needs those belongs in a @modal.enter().
set -euo pipefail

# Node 20 so the agent can `npm ci && npm run build` to verify
# sudarshana-gateway changes; its tooling needs newer Node than apt ships.
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
# NodeSource's script exits 0 when it fails, and apt then installs Debian's older Node.
node --version | grep -q '^v20\.'

# Fixed commit identity: a self-set noreply email once linked agent commits to
# an unrelated real GitHub user. .invalid never resolves.
git config --global user.name "Sudarshana (Rinkesh's autonomous agent)"
git config --global user.email sudarshana@agent.invalid

# Plain git commands log in through gh, which reads $GITHUB_TOKEN when git asks.
# Without this the agent put the token in remote URLs, which saves its value
# in .git/config on the Volume.
# This is what `gh auth setup-git` writes; that command refuses to run without a
# token, and there is none at build time.
git config --global credential.https://github.com.helper '!gh auth git-credential'
