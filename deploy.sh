#!/usr/bin/env bash
# Deploy sync_kde_rgb.py to ~/.local/bin and restart the systemd user service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/sync_kde_rgb.py"
DEST="$HOME/.local/bin/sync_kde_rgb.py"
SERVICE="sync-kde-rgb.service"

if [[ ! -f "$SRC" ]]; then
    echo "Error: $SRC not found." >&2
    exit 1
fi

# Syntax check before deploying
echo "Checking syntax..."
python3 -m py_compile "$SRC" || { echo "Syntax error — aborting." >&2; exit 1; }

# Deploy
echo "Copying $SRC → $DEST"
cp "$SRC" "$DEST"
chmod +x "$DEST"

# Restart service
echo "Restarting $SERVICE..."
systemctl --user restart "$SERVICE"

# Brief wait then show status
sleep 2
systemctl --user status "$SERVICE" --no-pager | head -12
echo ""
echo "Deploy complete."
