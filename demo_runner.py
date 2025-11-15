# demo_runner.py
import json
from agent import create_study_plan
from memory_manager import load_memory
from logger import log_event

def pretty_print_plan(record):
    print("=== Study Plan Summary ===")
    print("Topic:", record["topic"])
    print("Weeks:", record["weeks"])
    print("Answers:", record["answers"])
    plan = record.get("plan")
    if isinstance(plan, dict):
        # print keys nicely
        for k, v in plan.items():
            print(f"\n--- {k} ---")
            print(json.dumps(v, indent=2, ensure_ascii=False) if not isinstance(v, str) else v)
    else:
        print("Plan:", plan)
    print("\n--- Quiz (sample) ---")
    quiz = record.get("quiz", [])
    if quiz:
        for i, q in enumerate(quiz[:3]):
            print(f"{i+1}. {q.get('question')}")
            opts = q.get('options') or []
            for j, opt in enumerate(opts):
                print(f"   {j+1}. {opt}")
    else:
        print("No quiz generated.")

if __name__ == "__main__":
    user_id = "demo_user"
    topic = "Introduction to Machine Learning"
    record = create_study_plan(user_id, topic, weeks=2)
    pretty_print_plan(record)
    # show memory.json content
    print("\nMemory sample:")
    mem = load_memory()
    print(json.dumps(mem.get("users", {}).get(user_id, {}), indent=2, ensure_ascii=False))
    print("\nLogs saved in logs/log.jsonl")
