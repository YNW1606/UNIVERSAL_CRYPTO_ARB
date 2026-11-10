#!/bin/bash
set -e

echo "═══════════════════════════════════════════════════════════"
echo "  GLOBAL QUANT SYSTEM — Oracle Cloud ARM Setup"
echo "  Free Tier: 4 OCPU, 24GB RAM — runs forever"
echo "═══════════════════════════════════════════════════════════"

# 1. System packages
echo "[1/5] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y python3-pip python3-venv git screen htop

# 2. Python environment
echo "[2/5] Setting up Python virtual environment..."
cd ~/UNIVERSAL_CRYPTO_ARB
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install numpy requests xgboost joblib scipy -q

# 3. Test Python imports
echo "[3/5] Testing imports..."
python3 -c "import numpy, requests, xgboost, joblib, scipy; print('  All imports OK')"

# 4. Create cache directories
echo "[4/5] Creating directories..."
mkdir -p accuracy_db alpha_cache stat_arb_cache factor_cache risk_cache bt_cache

# 5. Create systemd service
echo "[5/5] Creating and enabling systemd service..."
sudo tee /etc/systemd/system/crypto_scanner.service > /dev/null << 'EOF'
[Unit]
Description=Global Quant System — Crypto Scanner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/UNIVERSAL_CRYPTO_ARB
ExecStart=/home/ubuntu/UNIVERSAL_CRYPTO_ARB/venv/bin/python signal_accuracy_tracker.py --watch 1800
Restart=always
RestartSec=30
StandardOutput=append:/home/ubuntu/UNIVERSAL_CRYPTO_ARB/scanner.log
StandardError=append:/home/ubuntu/UNIVERSAL_CRYPTO_ARB/scanner.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable crypto_scanner
sudo systemctl start crypto_scanner

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE — scanner is running 24/7"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Useful commands:"
echo "  sudo systemctl status crypto_scanner         # is it running?"
echo "  sudo systemctl restart crypto_scanner        # restart"
echo "  sudo systemctl stop crypto_scanner           # stop"
echo "  tail -f ~/UNIVERSAL_CRYPTO_ARB/scanner.log   # live logs"
echo ""
echo "  Accuracy report (run anytime after a few hours):"
echo "  cd ~/UNIVERSAL_CRYPTO_ARB && source venv/bin/activate"
echo "  python signal_accuracy_tracker.py --report"
echo ""
echo "  Scans every 30 min. Accuracy data builds in accuracy_db/signals.db"
echo "═══════════════════════════════════════════════════════════"
