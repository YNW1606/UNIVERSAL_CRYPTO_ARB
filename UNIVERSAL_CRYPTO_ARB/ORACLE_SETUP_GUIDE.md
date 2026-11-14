# Oracle Cloud Free Tier Setup — Crypto Scanner 24/7
4-core ARM VM, 24 GB RAM, $0.00 forever (no expiry, no auto-charges)

---

## Step 1 — Create Oracle Cloud Account (5 min)

1. Go to **https://cloud.oracle.com/free**
2. Sign up — credit card required for verification but **Never Charged** on Always Free
3. Choose home region: **ap-mumbai-1** (Mumbai) for India — lowest latency to Binance

---

## Step 2 — Launch Free ARM VM (3 min)

1. Dashboard → **Create a VM Instance**
2. Name: `crypto-scanner`
3. **Image**: Ubuntu 22.04 (Canonical)
4. **Shape**: Click "Change Shape"
   - Select **Ampere** tab
   - Shape: `VM.Standard.A1.Flex`
   - OCPU count: **4** (free allowance)
   - Memory: **24 GB** (free allowance)
5. **SSH Keys**: Upload your public key OR click "Generate a key pair" and download both files
6. Click **Create**

Wait ~2 minutes for status to show **Running**.

---

## Step 3 — Push Code to GitHub First (do this on your laptop)

```bash
cd C:\Users\Yash\UNIVERSAL_CRYPTO_ARB

# If not already a git repo with remote:
git remote add origin https://github.com/kiranmerchant6/UNIVERSAL_CRYPTO_ARB.git

git add .
git commit -m "Global quant system — 5 engines live"
git push -u origin master
```

---

## Step 4 — Connect to VM and Deploy (5 min)

Copy the **Public IP** from your VM details page, then:

```bash
# From your laptop terminal (Git Bash / PowerShell):
ssh ubuntu@<YOUR_VM_PUBLIC_IP>

# Now you're on the Oracle VM. Run:
git clone https://github.com/kiranmerchant6/UNIVERSAL_CRYPTO_ARB.git
cd UNIVERSAL_CRYPTO_ARB
bash setup_oracle.sh
```

That's it. The scanner starts immediately as a systemd service.

---

## Step 5 — Verify It's Running

```bash
sudo systemctl status crypto_scanner
# Should show: active (running)

tail -f ~/UNIVERSAL_CRYPTO_ARB/scanner.log
# Shows live scan output every 30 min
```

---

## Step 6 — View Accuracy Report (after a few hours)

```bash
ssh ubuntu@<YOUR_VM_PUBLIC_IP>
cd ~/UNIVERSAL_CRYPTO_ARB
source venv/bin/activate
python signal_accuracy_tracker.py --report
```

---

## Daily Commands Cheatsheet

```bash
# Check scanner health
sudo systemctl status crypto_scanner

# See live logs
tail -f ~/UNIVERSAL_CRYPTO_ARB/scanner.log

# Accuracy report
source ~/UNIVERSAL_CRYPTO_ARB/venv/bin/activate
python ~/UNIVERSAL_CRYPTO_ARB/signal_accuracy_tracker.py --report

# Manual single scan
python ~/UNIVERSAL_CRYPTO_ARB/global_quant_system.py

# Restart scanner
sudo systemctl restart crypto_scanner

# Stop scanner
sudo systemctl stop crypto_scanner

# Pull latest code updates
cd ~/UNIVERSAL_CRYPTO_ARB && git pull && sudo systemctl restart crypto_scanner
```

---

## What It Logs

Every 30 minutes the scanner:
1. Checks outcomes of signals from 4 hours ago (WIN/LOSS)
2. Runs all 5 engines on 8 symbols
3. Logs any active signals (score ≥ 35) to `accuracy_db/signals.db`
4. Prints a summary to `scanner.log`

After 1 week you'll have 300+ signal records to calculate:
- Win rate by score tier (PERFECT / HIGH / MODERATE)
- Information Coefficient per engine
- Profit Factor across all signals
- Best/worst performing symbols

---

## Always Free Tier Limits (Oracle)

| Resource | Always Free Allowance |
|---|---|
| ARM VMs (Ampere A1) | 4 OCPU + 24 GB RAM total |
| Boot volume | 200 GB |
| Outbound data | 10 TB/month |
| Uptime | Unlimited — never paused |
| Expiry | Never |

The crypto scanner uses < 5% CPU and < 500 MB RAM — well within limits.
