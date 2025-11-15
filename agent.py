# agent.py
import os
import json
import time
from typing import Dict, Any, List
from groq import Groq

from memory_manager import load_memory, save_memory, get_user_state, save_user_state
from tools import web_search, generate_quiz_prompt, parse_quiz_output_to_list
from logger import log_event

# -------------------------
# Configuration
# -------------------------
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set. Set it before running.")
client = Groq(api_key=GROQ_KEY)

MODEL_NAME = os.getenv("STUDY_BUDDY_MODEL", "llama-3.1-70b-versatile")

# -------------------------
# LLM wrapper using Groq
# -------------------------
def ask_llm(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 500) -> str:
    """
    messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
    Groq chat completion wrapper. Safe logging for usage objects.
    """
    groq_messages = []
    for m in messages:
        groq_messages.append({
            "role": m.get("role", "user"),
            "content": m.get("content", "")
        })

    log_event("llm_call.request", {"model": MODEL_NAME, "messages": groq_messages})

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        # Friendly message for model errors (decommissioned / not found)
        err_msg = str(e)
        log_event("llm_call.error", {"model": MODEL_NAME, "error": err_msg})
        print("LLM request failed. Error:", err_msg)
        print("Check that the model name in $STUDY_BUDDY_MODEL is correct and available in the Groq console.")
        raise

    # Safely extract text
    text = ""
    try:
        text = resp.choices[0].message.content
    except Exception:
        text = str(resp)

    # Make usage JSON-serializable before logging
    usage_info = getattr(resp, "usage", None)
    usage_safe = None
    try:
        # If usage_info behaves like a dict or has .to_dict() method, try converting
        if usage_info is None:
            usage_safe = None
        else:
            # attempt best-effort conversion
            try:
                usage_safe = dict(usage_info)
            except Exception:
                try:
                    usage_safe = usage_info.to_dict()
                except Exception:
                    usage_safe = str(usage_info)
    except Exception:
        usage_safe = str(usage_info)

    log_event("llm_call.response", {"model": MODEL_NAME, "usage": usage_safe})
    return text.strip()
# -------------------------
# Agent flows (same logical flow as before)
# -------------------------
def build_followup_questions(topic: str) -> List[str]:
    system = {
        "role": "system",
        "content": "You are a concise assistant that asks three follow-up questions in JSON to clarify the user's study requirements."
    }
    user = {
        "role": "user",
        "content": (
            f"User wants a study plan for the topic: '{topic}'. "
            "Return exactly a JSON object with keys: q1, q2, q3 (each a short question)."
        )
    }
    out = ask_llm([system, user], temperature=0.0, max_tokens=200)
    # Try to extract JSON
    try:
        parsed = json.loads(out)
        return [parsed.get("q1",""), parsed.get("q2",""), parsed.get("q3","")]
    except Exception:
        # fallback simple questions
        return [
            "What is your current skill level in this topic (beginner/intermediate/advanced)?",
            "How many hours per day can you dedicate to studying?",
            "What is your primary goal (e.g., exam, project, basics)?"
        ]

def synthesize_plan(topic: str, answers: Dict[str, Any], weeks: int = 2) -> Dict[str, Any]:
    # get references from web_search tool
    references = web_search(topic)
    prompt = (
        f"Create a {weeks}-week study plan for topic '{topic}' for a {answers.get('skill_level')} "
        f"learner with {answers.get('hours_per_day')} hours/day. Goal: {answers.get('goal')}. "
        f"Use these reference notes: {references}. Output JSON with keys: 'weeks' (array of per-week plans), "
        "'daily_template' (string), 'resources' (array). Keep it concise."
    )
    messages = [
        {"role": "system", "content": "You are a helpful study planner. Output valid JSON."},
        {"role": "user", "content": prompt}
    ]
    out = ask_llm(messages, temperature=0.2, max_tokens=1200)
    # parse JSON
    try:
        plan_json = json.loads(out)
    except Exception:
        # If parsing fails, store raw text in 'raw_plan'
        plan_json = {"raw_plan": out}
    return plan_json

def generate_quiz(topic: str, n: int = 5) -> List[Dict[str, Any]]:
    prompt = generate_quiz_prompt(topic, n)
    messages = [
        {"role": "system", "content": "You are a quiz generator. Output a JSON array."},
        {"role": "user", "content": prompt}
    ]
    out = ask_llm(messages, temperature=0.3, max_tokens=600)
    parsed = parse_possible_json_array(out)
    if parsed:
        return parsed
    return []

def parse_possible_json_array(text: str):
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        # try to salvage JSON-like substring
        import re
        match = re.search(r"(\[.*\])", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                return None
    return None

# -------------------------
# High-level API
# -------------------------
def create_study_plan(user_id: str, topic: str, weeks: int = 2, answers: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Orchestrates follow-up questions (if answers not provided), design plan, generate quiz, save to memory.
    Returns the saved plan record.
    """
    log_event("agent.create_study_plan.start", {"user_id": user_id, "topic": topic})
    # load user state
    state = get_user_state(user_id)
    # Step A: if no answers, create follow-up questions (we simulate user reply in demo)
    if answers is None:
        questions = build_followup_questions(topic)
        # For real app: present questions to user and collect answers.
        # For demo: choose default/simulated answers.
        answers = {
            "skill_level": "beginner",
            "hours_per_day": 2,
            "goal": f"Understand core {topic} concepts and solve beginner problems"
        }
        log_event("agent.followup_questions", {"questions": questions, "simulated_answers": answers})
    # Step B: plan synthesis
    plan = synthesize_plan(topic, answers, weeks=weeks)
    # Step C: generate quiz
    quiz = generate_quiz(topic, n=5)
    # Step D: store plan record
    plan_record = {
        "topic": topic,
        "weeks": weeks,
        "answers": answers,
        "plan": plan,
        "quiz": quiz,
        "created_at": time.time()
    }
    state_plans = state.get("plans", [])
    state_plans.append(plan_record)
    state["plans"] = state_plans
    save_user_state(user_id, state)
    log_event("agent.create_study_plan.finish", {"user_id": user_id, "topic": topic})
    return plan_record
