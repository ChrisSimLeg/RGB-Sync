#!/usr/bin/env bash
# Deploy sync_kde_rgb.py to ~/.local/bin and the service unit, then restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/sync_kde_rgb.py"
DEST="$HOME/.local/bin/sync_kde_rgb.py"
SERVICE_SRC="$SCRIPT_DIR/sync-kde-rgb.service"
SERVICE_DEST="$HOME/.config/systemd/user/sync-kde-rgb.service"
SERVICE="sync-kde-rgb.service"

if [[ ! -f "$SRC" ]]; then
    echo "Error: $SRC not found." >&2
    exit 1
fi

# Syntax check before deploying
echo "Checking syntax..."
python3 -m py_compile "$SRC" || { echo "Syntax error — aborting." >&2; exit 1; }

# Deploy script
echo "Copying $SRC → $DEST"
mkdir -p "$(dirname "$DEST")"
cp "$SRC" "$DEST"
chmod +x "$DEST"

# Deploy service unit if present
if [[ -f "$SERVICE_SRC" ]]; then
    echo "Copying $SERVICE_SRC → $SERVICE_DEST"
    mkdir -p "$(dirname "$SERVICE_DEST")"
    cp "$SERVICE_SRC" "$SERVICE_DEST"
    systemctl --user daemon-reload
fi

# Restart service
echo "Restarting $SERVICE..."
systemctl --user restart "$SERVICE"

# Brief wait then show status
sleep 2
systemctl --user status "$SERVICE" --no-pager | head -12
echo ""
echo "Deploy complete."
