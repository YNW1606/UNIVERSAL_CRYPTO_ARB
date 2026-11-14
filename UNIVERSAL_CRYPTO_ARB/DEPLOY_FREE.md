# Free 24/7 Deployment Options

## Option 1: GitHub Actions (FREE — 2000 min/month on free plan)

Best for: accuracy tracking only (no server needed, fully free)

1. Push this folder to a GitHub repo:
   ```
   cd C:\Users\Yash\UNIVERSAL_CRYPTO_ARB
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/universal-crypto-arb.git
   git push -u origin main
   ```
2. Go to repo Settings → Actions → General → Allow all actions
3. `.github/workflows/crypto_scan.yml` is already included — it runs every 30 min
4. Each run: scans all 8 symbols, logs signals to `accuracy_db/signals.db`, commits to repo
5. View results: check `accuracy_db/signals.db` in repo, or add a step to print the report

Limit: GitHub Actions gives 2000 free minutes/month. Each scan ≈ 5 min → ~400 scans/month → fine.

---

## Option 2: Oracle Cloud Free Tier (BEST — forever free, real server)

Best for: 24/7 always-on with no limits

1. Sign up at https://cloud.oracle.com (free, requires credit card but NOT charged)
2. Launch an Always Free ARM VM:
   - Shape: VM.Standard.A1.Flex — 4 OCPU, 24 GB RAM (completely free forever)
   - OS: Ubuntu 22.04
3. SSH in and set up:
   ```bash
   sudo apt update && sudo apt install python3-pip git screen -y
   git clone https://github.com/YOUR_USERNAME/universal-crypto-arb.git
   cd universal-crypto-arb
   pip3 install -r requirements.txt
   ```
4. Run forever in background with screen:
   ```bash
   screen -S quant
   python3 signal_accuracy_tracker.py --watch 1800
   # Ctrl+A then D to detach (keeps running after logout)
   ```
5. Auto-restart on reboot:
   ```bash
   crontab -e
   # Add this line:
   @reboot cd /home/ubuntu/universal-crypto-arb && screen -dmS quant python3 signal_accuracy_tracker.py --watch 1800
   ```
6. Check logs anytime: `screen -r quant`

---

## Option 3: Railway (~$5/month — you already have an account)

Best for: easy deploy, you already use Railway for ATRION

```bash
cd C:\Users\Yash\UNIVERSAL_CRYPTO_ARB
railway login
railway init
railway up
```

`Procfile` and `railway.toml` are already included — no extra config needed.
Cost: ~$5/month (CPU usage is minimal for a scanner).

---

## Checking accuracy results

Once running (any option), view results:
```bash
python signal_accuracy_tracker.py --report
```

Or on the server:
```bash
python3 signal_accuracy_tracker.py --report
```
