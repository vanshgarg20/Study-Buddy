# debug_runner.py
import traceback, json, os, sys, time
from pathlib import Path

# show which python and env var preview
print("Python:", sys.executable)
print("OPENAI_API_KEY set?:", bool(os.getenv("OPENAI_API_KEY")))
print("STUDY_BUDDY_MODEL:", os.getenv("STUDY_BUDDY_MODEL"))

# try to import and run demo flow with detailed exception capture
try:
    # Import here so we catch import-time errors
    from agent import create_study_plan
    print("\nImported agent.create_study_plan successfully.")
    print("Calling create_study_plan('debug_user', 'Test Topic', weeks=1)...\n")
    rec = create_study_plan("debug_user", "Test Topic", weeks=1)
    print("Function returned. Summary:")
    # Pretty print limited output
    print("Topic:", rec.get("topic"))
    print("Created at:", rec.get("created_at"))
    print("Plan type:", type(rec.get("plan")))
    # If plan is dict, show keys / small preview
    p = rec.get("plan")
    if isinstance(p, dict):
        print("Plan keys:", list(p.keys()))
        try:
            print("Plan preview:", json.dumps(p, indent=2)[:2000])
        except Exception:
            print("Plan preview (raw):", str(p)[:1000])
    else:
        print("Plan content (first 1000 chars):", str(p)[:1000])
except Exception as e:
    print("\n--- Exception traceback ---")
    traceback.print_exc()
    print("--- end traceback ---\n")

# show memory.json
mem_path = Path("memory.json")
if mem_path.exists():
    try:
        print("\n--- memory.json (preview) ---")
        with mem_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # show debug_user entry if present
            u = data.get("users", {}).get("debug_user")
            print(json.dumps(u, indent=2, ensure_ascii=False)[:2000])
    except Exception:
        print("Could not read memory.json (parsing error). Raw file:")
        print(mem_path.read_text()[:2000])
else:
    print("\nmemory.json not found.")

# show last 40 lines of logs/log.jsonl if exists
logs = Path("logs/log.jsonl")
if logs.exists():
    print("\n--- last lines of logs/log.jsonl ---")
    with logs.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[-40:]:
            # try pretty printing small entries
            try:
                obj = json.loads(line)
                print(json.dumps(obj, indent=None, ensure_ascii=False))
            except Exception:
                print(line.strip())
else:
    print("\nlogs/log.jsonl not found.")
