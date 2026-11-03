"""Hourly CSV-folder monitor with optional Telegram alerts.

This does NOT fetch MCX data by itself. Place fresh, timestamped CSV files in
incoming/ (options_latest.csv and futures_history.csv) using a permitted data
source/export. Run this process on an always-on PC/VPS for hourly checks.
"""
import os, time, json, hashlib
from pathlib import Path
import pandas as pd
import requests
from risk_engine import analyse

ROOT = Path(__file__).resolve().parent
INCOMING = ROOT / "incoming"
STATE = ROOT / "state.json"
INCOMING.mkdir(exist_ok=True)


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=15)
    r.raise_for_status()
    return True


def main():
    interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "3600"))
    while True:
        try:
            op = INCOMING / "options_latest.csv"
            fu = INCOMING / "futures_history.csv"
            if op.exists() and fu.exists():
                options = pd.read_csv(op); futures = pd.read_csv(fu)
                prior = json.loads(STATE.read_text()) if STATE.exists() else {}
                result = analyse(options, futures, prior)
                snap = result["snapshot"]
                message = (f"MCX Natural Gas risk scan\nOverall: {result['overall_score']}/100 ({result['overall_label']})"
                           f"\nPut seller: {result['put_score']}/100 ({result['put_label']})"
                           f"\nCall seller: {result['call_score']}/100 ({result['call_label']})"
                           f"\nTrend: {result['trend']} | PCR: {result['pcr']}"
                           f"\nWarnings: " + "; ".join(result["warnings"]))
                # Notify each hourly run when risk is elevated; avoid sending repeated low-risk noise.
                if max(result["overall_score"], result["put_score"], result["call_score"]) >= 30:
                    send_telegram(message)
                STATE.write_text(json.dumps(snap, indent=2))
                print(message, flush=True)
            else:
                print("Waiting for incoming/options_latest.csv and incoming/futures_history.csv", flush=True)
        except Exception as e:
            print(f"Scan error: {e}", flush=True)
        time.sleep(interval)

if __name__ == "__main__": main()
