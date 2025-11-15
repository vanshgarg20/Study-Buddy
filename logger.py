# logger.py
import os
import json
import time
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def log_event(event_type: str, payload: dict):
    """
    Save a log entry to logs/log.jsonl with a timestamp.
    """
    entry = {
        "ts": time.time(),
        "event": event_type,
        "payload": payload
    }
    filepath = LOG_DIR / "log.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
