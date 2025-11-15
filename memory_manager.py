# memory_manager.py
import json
from pathlib import Path
from typing import Dict, Any

MEMORY_FILE = Path("memory.json")

def load_memory() -> Dict[str, Any]:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    # default structure
    return {"users": {}}

def save_memory(mem: Dict[str, Any]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

def get_user_state(user_id: str) -> Dict[str, Any]:
    mem = load_memory()
    return mem.get("users", {}).get(user_id, {"plans": [], "sessions": []})

def save_user_state(user_id: str, state: Dict[str, Any]):
    mem = load_memory()
    users = mem.setdefault("users", {})
    users[user_id] = state
    save_memory(mem)
