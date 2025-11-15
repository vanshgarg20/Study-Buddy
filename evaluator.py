# evaluator.py
import json
import time
from agent import create_study_plan
from memory_manager import get_user_state
from logger import log_event

# Simple evaluation harness:
# - Define several test prompts (simulated users)
# - For each, produce plan and score against a simple rubric:
#   completeness (0-5), clarity (0-5), resources (0-5)
# The scoring is manual-heuristic to produce evaluation numbers for the writeup.

TEST_CASES = [
    {"user_id": "eval_user_1", "topic": "Linear Algebra basics", "weeks": 2},
    {"user_id": "eval_user_2", "topic": "Intro to Python", "weeks": 1},
    {"user_id": "eval_user_3", "topic": "Probability for ML", "weeks": 3}
]

def score_plan(plan_record):
    # Heuristic scoring: prefer plans with explicit "weeks" array and resources
    plan = plan_record.get("plan", {})
    s_comp = 0
    s_clarity = 0
    s_resources = 0
    if isinstance(plan, dict):
        if plan.get("weeks"):
            s_comp = min(5, max(2, len(plan.get("weeks"))))  # if present
        else:
            s_comp = 1
        # clarity: presence of daily_template improves clarity
        s_clarity = 5 if plan.get("daily_template") else 2
        s_resources = 5 if plan.get("resources") else 1
    else:
        s_comp = 1
        s_clarity = 1
        s_resources = 1
    return {"completeness": s_comp, "clarity": s_clarity, "resources": s_resources}

def run_evaluation():
    results = []
    for tc in TEST_CASES:
        rec = create_study_plan(tc["user_id"], tc["topic"], weeks=tc["weeks"])
        scores = score_plan(rec)
        results.append({"user_id": tc["user_id"], "topic": tc["topic"], "scores": scores})
        time.sleep(1)  # small pause to avoid rate bursts
    # Aggregate
    agg = {"count": len(results), "avg_completeness": 0, "avg_clarity": 0, "avg_resources": 0}
    for r in results:
        agg["avg_completeness"] += r["scores"]["completeness"]
        agg["avg_clarity"] += r["scores"]["clarity"]
        agg["avg_resources"] += r["scores"]["resources"]
    agg["avg_completeness"] /= agg["count"]
    agg["avg_clarity"] /= agg["count"]
    agg["avg_resources"] /= agg["count"]
    # Save results
    fpath = f"evaluation_result_{int(time.time())}.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump({"results": results, "aggregate": agg}, f, indent=2, ensure_ascii=False)
    print("Evaluation complete. Results saved to", fpath)
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    log_event("evaluation.run", {"aggregate": agg})
    return agg

if __name__ == "__main__":
    run_evaluation()
