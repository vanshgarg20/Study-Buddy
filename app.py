# app.py - Final Study Buddy (Groq LLM integration + local fallback)
# Requirements: pip install streamlit requests reportlab
import re
import time
import io
import base64
import hashlib
import json
from typing import Dict, Any, List, Optional
from html import escape

import streamlit as st
import streamlit.components.v1 as components
import requests

# Optional PDF support
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# -------------------------
# session state for plans
if "demo_state" not in st.session_state:
    st.session_state["demo_state"] = {"plans": []}

def get_user_state(uid: str) -> Dict[str, Any]:
    return st.session_state["demo_state"]

def save_user_state(uid: str, state: Dict[str, Any]) -> None:
    st.session_state["demo_state"] = state

# -------------------------
# helpers
STOPWORDS = {"and","the","of","in","for","to","with","a","an","on","by","&","&amp;","is","this"}

def clean_tokens(topic: str) -> List[str]:
    parts = re.split(r'[^0-9A-Za-z]+', topic.lower())
    toks = [p for p in parts if p and p not in STOPWORDS]
    return toks or [topic.lower()]

def deterministic_seed(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def slugify(s: Optional[str]) -> str:
    if s is None:
        return "item"
    s = str(s).strip().lower()
    s = re.sub(r'[^0-9a-z]+', '_', s)
    s = re.sub(r'__+', '_', s).strip('_')
    return s or "item"

def plan_to_text(record: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Study Buddy — Plan: {record.get('topic','')}")
    created_ts = int(record.get("created_at", time.time()))
    lines.append(f"Saved: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_ts))}")
    lines.append(f"Source: {record.get('source','local')} • Weeks: {record.get('weeks')}")
    lines.append("")
    answers = record.get("answers", {})
    lines.append(f"Weeks: {record.get('weeks','—')}  •  Skill: {answers.get('skill_level','—')}  •  Hours/day: {answers.get('hours_per_day','—')}")
    if answers.get("goal"):
        lines.append(f"Goal: {answers.get('goal')}")
    lines.append("")
    plan = record.get("plan", {})
    weeks_list = plan.get("weeks", [])
    for w_i, week in enumerate(weeks_list, start=1):
        lines.append(f"Week {w_i}")
        for d in week.get("days", []):
            day_name = d.get("day", "")
            topics = d.get("topics", [])
            topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)
            lines.append(f"  - {day_name}: {topics_str}")
        lines.append("")
    if plan.get("daily_template"):
        lines.append("Daily Template:")
        lines.append(f"  {plan.get('daily_template')}")
        lines.append("")
    if plan.get("resources"):
        lines.append("Recommended Resources:")
        for r in plan.get("resources", []):
            lines.append(f"  - {r}")
    return "\n".join(lines)

# -------------------------
# PDF helper (optional)
def generate_pdf_bytes_platypus(title: str, record: Dict[str, Any]) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not installed")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18, leading=22, alignment=0, spaceAfter=12)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, textColor=colors.grey, spaceAfter=8)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading3"], fontSize=12, leading=14, spaceBefore=8, spaceAfter=6)
    normal_style = ParagraphStyle("NormalCustom", parent=styles["Normal"], fontSize=11, leading=14, spaceAfter=4)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=11, leftIndent=12, leading=14, spaceAfter=2)
    story = []
    story.append(Paragraph(escape(title), title_style))
    created_ts = int(record.get("created_at", time.time()))
    meta_text = f"Saved: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_ts))} • Source: {escape(str(record.get('source','local')))} • Weeks: {record.get('weeks')}"
    story.append(Paragraph(meta_text, meta_style))
    answers = record.get("answers", {})
    summary = f"Weeks: {record.get('weeks','—')}  •  Skill: {escape(str(answers.get('skill_level','—')))}  •  Hours/day: {escape(str(answers.get('hours_per_day','—')))}"
    story.append(Paragraph(summary, normal_style))
    if answers.get("goal"):
        story.append(Paragraph(f"<b>Goal:</b> {escape(str(answers.get('goal')))}", normal_style))
    story.append(Spacer(1, 8))
    plan = record.get("plan", {})
    weeks_list = plan.get("weeks", [])
    for w_i, week in enumerate(weeks_list, start=1):
        story.append(Paragraph(f"Week {w_i}", heading_style))
        for d in week.get("days", []):
            day_name = escape(d.get("day", ""))
            topics = d.get("topics", [])
            topics_str = escape(", ".join(topics) if isinstance(topics, list) else str(topics))
            story.append(Paragraph(f"• <b>{day_name}:</b> {topics_str}", bullet_style))
        story.append(Spacer(1, 6))
    daily_template = plan.get("daily_template")
    if daily_template:
        story.append(Paragraph("<b>Daily Template:</b>", heading_style))
        story.append(Paragraph(escape(daily_template), normal_style))
        story.append(Spacer(1, 6))
    resources = plan.get("resources", [])
    if resources:
        story.append(Paragraph("<b>Recommended Resources:</b>", heading_style))
        for r in resources:
            story.append(Paragraph(f"• {escape(str(r))}", bullet_style))
        story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# -------------------------
# Domain modules (career included)
DOMAIN_MODULES = {
    "dsa": [
        ["Arrays","Strings","Two Pointers"],
        ["Linked Lists","Stacks","Queues"],
        ["Trees","BST","Trie"],
        ["Graphs","BFS/DFS","Shortest Paths"],
        ["Sorting","Searching","Hashing"],
        ["Dynamic Programming","Greedy","Backtracking"]
    ],
    "ml": [
        ["Math & Linear Algebra","Probability","Statistics"],
        ["Supervised Learning","Regression","Classification"],
        ["Neural Networks","CNNs","RNNs"],
        ["Optimization","Loss Functions","Regularization"],
        ["Deployment","Model Serving","Monitoring"],
        ["Advanced Topics","Transformers","Self-supervised"]
    ],
    "web": [
        ["HTML & CSS","DOM","Accessibility"],
        ["JavaScript Basics","ES6+","DOM Manipulation"],
        ["Frontend Framework","React/Vue","State Management"],
        ["Backend Basics","APIs","Databases"],
        ["Auth & Security","Testing","Deployment"],
        ["Performance","Caching","Scaling"]
    ],
    "db": [
        ["RDBMS Basics","SQL Queries","Joins"],
        ["Indexes","Query Optimization","Transactions"],
        ["NoSQL Basics","Document Stores","Key-Value DB"],
        ["Data Modeling","Normalization","Denormalization"],
        ["Replication","Sharding","Backup/Restore"],
        ["Analytics","OLAP","Data Warehousing"]
    ],
    "cv": [
        ["Image Processing","Filters","Transforms"],
        ["Classical CV","Features","SIFT/ORB"],
        ["Deep CV","CNNs","Object Detection"],
        ["Segmentation","Keypoints","Pose Estimation"],
        ["Data Augmentation","Training Tricks","Transfer Learning"],
        ["Deployment","Edge Models","Optimization"]
    ],
    "career": [
        ["Resume & Profile","Formatting","Keywords"],
        ["Behavioral Questions","STAR Method","Story Crafting"],
        ["Mock Interviews","Problem Solving","Pair Practice"],
        ["Company Research","Role Mapping","Expectations"],
        ["System Design / Case Study","High-level Thinking","Trade-offs"],
        ["Negotiation & HR","Offer Review","Salary Discussion"]
    ],
    "default": [
        ["Introduction","Core Concepts","Motivation"],
        ["Tools & Setup","Libraries","Environment"],
        ["Core Algorithms","Patterns","Practice"],
        ["Mini-project","Integration","Testing"],
        ["Optimization","Scaling","Evaluation"],
        ["Advanced Concepts","Papers","Further Reading"]
    ]
}

def detect_domain(tokens: List[str]) -> str:
    tset = set(tokens)
    # career/soft-skills heuristics
    if any(x in tset for x in ("interview","interviews","resume","cv","cvresume","behavioural","behavioral","behavior","soft","skill","skills","communication","story","portfolio","networking","negotiation","salary","hr","offer","mock")):
        return "career"
    if any(x in tset for x in ("dsa","ds","data","structure","structures","algorithm","algorithms","algo")):
        return "dsa"
    if any(x in tset for x in ("ml","machine","learning","neural","deep","model","models")):
        return "ml"
    if any(x in tset for x in ("web","frontend","backend","react","vue","javascript","html","css","node")):
        return "web"
    if any(x in tset for x in ("db","database","sql","nosql","postgres","mysql","mongodb")):
        return "db"
    if any(x in tset for x in ("cv","vision","image","opencv","cnn","segmentation","detection")):
        return "cv"
    return "default"

# build topic-aware day list using modules
def build_topic_daylist(topic: str, weeks: int, answers: Dict[str, Any]) -> Dict[str, Any]:
    weeks = max(1, int(weeks))
    tokens = clean_tokens(topic)
    domain = detect_domain(tokens)
    seed = deterministic_seed(topic + str(answers.get("skill_level","")))
    modules = DOMAIN_MODULES.get(domain, DOMAIN_MODULES["default"])
    base_days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    weeks_list = []
    for w in range(weeks):
        day_entries = []
        for i, day in enumerate(base_days):
            row_idx = (seed + w*3 + i) % len(modules)
            row = modules[row_idx]
            primary = row[i % len(row)]
            secondary = row[(i+1) % len(row)]
            token = tokens[(seed + i + w) % len(tokens)]
            if domain == "career":
                if i % 3 == 0:
                    topics = [primary, f"Practice: {token} (mock)"]
                elif i % 3 == 1:
                    topics = [secondary, "STAR story refinement", f"Action: draft 1 resume bullet about {token}"]
                else:
                    topics = ["Mock interview (30-45 min)", "Feedback & notes", f"Company research: {token}"]
            elif domain == "dsa":
                topics = [primary, secondary, f"LeetCode practice: {token}"]
            elif domain == "ml":
                topics = [primary, secondary, f"Mini experiment: {token}"]
            elif domain == "web":
                topics = [primary, secondary, f"Build: {token} feature"]
            elif domain == "db":
                topics = [primary, secondary, f"Design exercise: model {token}"]
            elif domain == "cv":
                topics = [primary, secondary, f"Notebook: try task on {token}"]
            else:
                topics = [primary, secondary, f"Practice: {token}"]
            day_entries.append({"day": day, "topics": topics})
        weeks_list.append({"days": day_entries})
    # domain-tuned daily templates & resources
    hours = answers.get("hours_per_day", 2)
    skill = answers.get("skill_level", "beginner")
    if domain == "career":
        daily_template = f"{hours} hours/day: 40% practice (mock + problems) • 30% story & resume work • 30% research & interviews"
        resources = [
            "Resume templates & review guides",
            "Common behavioral questions + STAR examples",
            "Platforms for mock interviews (Pramp, InterviewBuddy, peers)",
            "System design primers & curated interview lists"
        ]
    elif domain == "dsa":
        daily_template = f"{hours} hours/day: 60% problem practice + 30% concept review + 10% mock tests"
        resources = [
            "Top LeetCode lists",
            "CLRS / EPI chapters (selected)",
            "Interactive practice: Codeforces/HackerRank"
        ]
    elif domain == "ml":
        daily_template = f"{hours} hours/day: theory + experiments + reading (adjust by skill={skill})"
        resources = [
            "Practical ML course (Coursera/fast.ai)",
            "Hands-on books & Kaggle notebooks",
            "Model deployment tutorials"
        ]
    else:
        daily_template = f"{hours} hours/day: mix of theory + practice + mini project"
        resources = [
            f"Intro course about {topic}",
            "YouTube playlists & sample projects",
            "Documentation & official guides"
        ]
    plan = {"weeks": weeks_list, "daily_template": daily_template, "resources": resources}
    return plan

def local_create_plan(uid: str, topic: str, weeks: int, answers: Dict[str, Any]) -> Dict[str, Any]:
    created = int(time.time())
    rec = {
        "topic": topic.strip() or "Topic",
        "weeks": int(max(1, weeks)),
        "answers": answers,
        "plan": build_topic_daylist(topic.strip() or "Topic", int(weeks), answers),
        "quiz": [],
        "created_at": created,
        "source": "local"
    }
    return rec

# -------------------------
# Robust remote API wrapper (Groq OpenAI-compatible)
def generate_plan_via_api(uid: str, topic: str, weeks: int, answers: Dict[str, Any]) -> Dict[str, Any]:
    api_key = st.secrets.get("API_KEY")
    api_url = st.secrets.get("API_URL")
    if not api_key or not api_url:
        st.info("No API secrets found — using local generator.")
        return local_create_plan(uid, topic, weeks, answers)

    model_name = st.secrets.get("GROQ_MODEL", "llama-3.1-chat")

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that produces structured study plans. "
            "Always reply with valid JSON only, using the exact schema specified below. "
            "Do NOT include any extra explanation or markdown, only JSON."
        )
    }

    user_instructions = {
        "role": "user",
        "content": (
            f"Generate a study plan for topic: \"{topic}\". Weeks: {weeks}. Skill: {answers.get('skill_level')}. "
            f"Hours per day: {answers.get('hours_per_day')}. Goal: {answers.get('goal')}. "
            "Output MUST be valid JSON with this structure:\n\n"
            '{\n'
            '  "topic": string,\n'
            '  "plan": {\n'
            '    "weeks": [\n'
            '      { "days": [ { "day": "Mon|Tue|...", "topics": ["topic1","topic2", ...] }, ... ] },\n'
            '      ...\n'
            '    ],\n'
            '    "daily_template": string,\n'
            '    "resources": [ "resource1", "resource2", ... ]\n'
            '  }\n'
            '}\n\n'
            "Make the topics concrete and domain-specific (not generic words). Keep strings short."
        )
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [system_msg, user_instructions],
        "max_tokens": 1200,
        "temperature": 0.2
    }

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # parse OpenAI-like choices or direct JSON
        plan_obj = None
        topic_out = topic

        if isinstance(data, dict) and "choices" in data:
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                assistant_text = choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
            else:
                assistant_text = ""
            # try parse assistant_text as JSON
            try:
                parsed = json.loads(assistant_text)
                plan_obj = parsed.get("plan") if isinstance(parsed, dict) else None
                topic_out = parsed.get("topic", topic) if isinstance(parsed, dict) else topic
            except Exception:
                # extract json substring
                import re
                m = re.search(r'\{.*\}', assistant_text, flags=re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        plan_obj = parsed.get("plan") if isinstance(parsed, dict) else None
                        topic_out = parsed.get("topic", topic) if isinstance(parsed, dict) else topic
                    except Exception:
                        plan_obj = None
                        topic_out = topic
                else:
                    plan_obj = None
                    topic_out = topic
        else:
            # direct structured response
            plan_obj = data.get("plan") if isinstance(data, dict) else None
            topic_out = data.get("topic", topic) if isinstance(data, dict) else topic

        if plan_obj and isinstance(plan_obj, dict):
            rec = {
                "topic": topic_out,
                "weeks": int(max(1, weeks)),
                "answers": answers,
                "plan": plan_obj,
                "quiz": data.get("quiz", []) if isinstance(data, dict) else [],
                "created_at": int(time.time()),
                "source": "remote"
            }
            return rec

        # Debug output if couldn't parse structured plan
        st.warning("Remote API returned response but could not parse structured plan. Showing response for debugging.")
        try:
            st.code(json.dumps(data, indent=2), language="json")
        except Exception:
            st.text("Raw response:")
            st.text(resp.text)
    except requests.HTTPError as e:
        try:
            st.error(f"Remote API HTTP error: {e} — response: {resp.text}")
        except Exception:
            st.error(f"Remote API HTTP error: {e}")
    except Exception as e:
        st.error(f"Remote API request failed: {e}")

    st.info("Falling back to local generator.")
    return local_create_plan(uid, topic, weeks, answers)

# -------------------------
# UI + styles
st.set_page_config(page_title="Study Buddy", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.sb-page{display:flex;justify-content:center;padding:20px 0 40px}
.sb-container{width:100%;max-width:980px;padding:0 20px}
.sb-title{font-size:48px;font-weight:900;margin:4px 0 6px 0;background:linear-gradient(90deg,#00d4ff,#5b7cff,#c86dd7,#ff7abd);background-size:400% 400%;-webkit-background-clip:text;color:transparent;animation:floatGradient 12s ease-in-out infinite;text-shadow:0 6px 18px rgba(0,0,0,0.45)}
@keyframes floatGradient{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.sb-sub{color:#9aa3b2;margin-bottom:18px;font-size:14px}
.sb-card{background:rgba(255,255,255,0.02);border-radius:12px;padding:18px 20px;margin-bottom:18px;box-shadow:0 6px 18px rgba(0,0,0,0.45)}
.sb-topic{font-size:20px;font-weight:700;margin-bottom:6px;color:#fff}
.sb-meta{color:#9aa3b2;margin-bottom:10px;font-size:13px}
.sb-week{font-size:15px;font-weight:700;margin-top:12px;margin-bottom:6px}
.sb-day{margin-left:14px;margin-bottom:4px;color:#e6eef8}
.actions-wrapper{display:flex;align-items:center;gap:10px;margin-top:6px}
.actions-pill{background:rgba(255,255,255,0.03);color:#f4f7fb;border-radius:999px;padding:8px 12px;border:1px solid rgba(255,255,255,0.04);cursor:pointer;font-weight:700;display:inline-flex;align-items:center;gap:8px}
.actions-dropdown{position:absolute;top:44px;left:0;background:rgba(20,24,28,0.98);border-radius:10px;padding:8px;min-width:200px;box-shadow:0 10px 30px rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.03);z-index:9999;display:none;flex-direction:column;gap:6px}
.actions-dropdown.show{display:flex}
.actions-item{background:transparent;color:#e6eef8;border-radius:8px;padding:8px 12px;cursor:pointer;font-weight:600;display:flex;gap:8px;align-items:center}
.actions-item:hover{background:rgba(255,255,255,0.02);transform:translateY(-1px)}
.stButton>button,.stButton>div>button{border-radius:10px !important;padding:8px 12px !important;font-weight:700 !important}
@media (max-width:720px){.sb-title{font-size:36px}.actions-dropdown{left:auto;right:0}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar form (session-state safe)
with st.sidebar:
    st.header("Create Plan")
    if "topic_input" not in st.session_state:
        st.session_state["topic_input"] = "machine learning"
    if "weeks_input" not in st.session_state:
        st.session_state["weeks_input"] = 2
    if "skill_input" not in st.session_state:
        st.session_state["skill_input"] = "beginner"
    if "hours_input" not in st.session_state:
        st.session_state["hours_input"] = 2
    if "goal_input" not in st.session_state:
        st.session_state["goal_input"] = "Build a small project"
    if "username_input" not in st.session_state:
        st.session_state["username_input"] = ""

    with st.form(key="create_form"):
        topic = st.text_area("Topic", height=38, key="topic_input")
        weeks = st.number_input("Weeks", min_value=1, max_value=52, step=1, key="weeks_input")
        skill_level = st.selectbox("Skill level", ["beginner", "intermediate", "advanced"], key="skill_input")
        hours_per_day = st.slider("Hours per day", 1, 8, key="hours_input")
        goal = st.text_area("Goal (short)", height=64, key="goal_input")
        username = st.text_input("Username (optional)", key="username_input", help="Optional: use same name to keep plans consistent across sessions later.")
        submit = st.form_submit_button("Generate")

# Show small status: remote available?
if "API_KEY" in st.secrets and "API_URL" in st.secrets:
    st.info("Remote generation: ENABLED (using API_KEY + API_URL from secrets).")
else:
    st.info("Remote generation: NOT enabled — app will use local generator (no secrets found).")

# -------------------------
# Generate (use remote if possible, else local)
status_saved = False
uid = (st.session_state.get("username_input","").strip() or "session_user")
if submit:
    if not st.session_state.get("topic_input","").strip():
        st.error("Please enter a topic.")
    else:
        answers = {"skill_level": st.session_state.get("skill_input"), "hours_per_day": st.session_state.get("hours_input"), "goal": st.session_state.get("goal_input")}
        rec = generate_plan_via_api(uid, st.session_state["topic_input"].strip(), int(st.session_state["weeks_input"]), answers)
        state = get_user_state(uid)
        plans = state.get("plans", [])
        # Replace existing plan at index 0 (so only one saved plan per user)
        if plans:
            plans[0] = rec
        else:
            plans.insert(0, rec)
        state["plans"] = plans
        save_user_state(uid, state)
        st.success(f"Plan created and saved ✅ (source: {rec.get('source')})")
        status_saved = True

# -------------------------
# Header + render plans (copy & download only)
st.markdown("<div class='sb-page'><div class='sb-container'>", unsafe_allow_html=True)
st.markdown("<div class='sb-title'>Study Buddy</div>", unsafe_allow_html=True)
st.markdown("<div class='sb-sub'>Topic-aware study plans — remote API used when available.</div>", unsafe_allow_html=True)
if status_saved:
    st.success("Saved to memory.")
st.markdown("---")

if st.session_state.get("username_input","").strip():
    st.markdown(f"<div class='sb-meta sb-muted'>Logged as: <strong>{st.session_state.get('username_input')}</strong></div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='sb-meta sb-muted'>No username provided — plans saved to this browser session only.</div>", unsafe_allow_html=True)

state = get_user_state(uid)
plans = state.get("plans", [])

if not plans:
    st.info("No saved plans yet. Create one from the sidebar.")
else:
    for idx, p in enumerate(plans):
        topic_local = p.get("topic", "Plan")
        created = int(p.get("created_at", time.time()))
        answers = p.get("answers", {})
        plan_data = p.get("plan", {})
        source = p.get("source", "local")
        topic_slug = slugify(topic_local)
        key_base = f"{uid}_{topic_slug}_{created}_{idx}"

        st.markdown("<div class='sb-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='sb-topic'>{topic_local}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sb-meta muted'>saved {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created))} • source: {source} • weeks: {p.get('weeks')}</div>", unsafe_allow_html=True)

        # meta & goal
        st.markdown(f"- **Weeks:** {p.get('weeks','—')}  •  **Skill:** {answers.get('skill_level','—')}  •  **Hours/day:** {answers.get('hours_per_day','—')}")
        if answers.get('goal'):
            goal_raw = answers.get('goal')
            goal_display = goal_raw.replace("\n", "  \n")
            st.markdown(f"- **Goal:** {goal_display}")
        st.markdown("")

        # weeks
        weeks_list = plan_data.get("weeks", [])
        for w_i, week in enumerate(weeks_list, start=1):
            st.markdown(f"<div class='sb-week'>Week {w_i}</div>", unsafe_allow_html=True)
            for d in week.get("days", []):
                day_name = d.get("day", "")
                topics = d.get("topics", [])
                topics_str = ", ".join(topics) if isinstance(topics, list) else str(topics)
                st.markdown(f"<div class='sb-day'>• <strong>{day_name}:</strong> {topics_str}</div>", unsafe_allow_html=True)
            st.markdown("")

        # daily template & resources
        if plan_data.get("daily_template"):
            st.markdown("**Daily Template:**")
            st.markdown(f"- {plan_data.get('daily_template')}")
        resources = plan_data.get("resources", [])
        if resources:
            st.markdown("**Recommended Resources:**")
            for r in resources:
                st.markdown(f"- {r}")

        # Prepare PDF data URI (if available)
        pdf_data_uri = None
        if REPORTLAB_AVAILABLE:
            try:
                pdf_bytes = generate_pdf_bytes_platypus(f"Study Buddy — {topic_local}", p)
                b64 = base64.b64encode(pdf_bytes).decode("ascii")
                pdf_data_uri = f"data:application/pdf;base64,{b64}"
            except Exception:
                pdf_data_uri = None

        plan_text = plan_to_text(p)
        plan_js_safe = escape(plan_text).replace("\\", "\\\\").replace("`", "\\`").replace("\n","\\n").replace("\r","")

        # Actions: copy + download (no delete)
        col_left, col_right = st.columns([0.92, 0.08])
        with col_left:
            actions_html = f"""
            <div class="actions-wrapper" style="position:relative">
              <div class="actions-menu">
                <div class="actions-pill" id="pill_{key_base}">Actions ▾</div>
                <div class="actions-dropdown" id="menu_{key_base}">
                  <button class="actions-item" id="menu_download_{key_base}"><span class="icon">📄</span> Download PDF</button>
                  <button class="actions-item" id="menu_copy_{key_base}"><span class="icon">📋</span> Copy Plan</button>
                </div>
              </div>
            </div>

            <script>
            (function(){{
              const pill = document.getElementById("pill_{key_base}");
              const menu = document.getElementById("menu_{key_base}");
              pill.addEventListener("click", (e) => {{ e.stopPropagation(); menu.classList.toggle("show"); }});
              document.addEventListener("click", () => {{ menu.classList.remove("show"); }});
              menu.addEventListener("click", (e) => {{ e.stopPropagation(); }});

              // Download
              const dl = document.getElementById("menu_download_{key_base}");
              dl.addEventListener("click", () => {{
                const uri = { ('"' + pdf_data_uri + '"') if pdf_data_uri else 'null' };
                if (uri) {{
                    const a = document.createElement('a');
                    a.href = uri;
                    a.download = "{topic_slug}_plan_{created}.pdf";
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                }} else {{
                    alert('PDF not available. Ensure reportlab is installed.');
                }}
                menu.classList.remove('show');
              }});

              // Copy
              const copyBtn = document.getElementById("menu_copy_{key_base}");
              copyBtn.addEventListener("click", async () => {{
                try {{
                    await navigator.clipboard.writeText(`{plan_js_safe}`);
                    copyBtn.innerText = 'Copied ✓';
                    setTimeout(()=>{{ copyBtn.innerText = 'Copy Plan'; }}, 1400);
                }} catch(e) {{
                    window.prompt('Copy plan text (Ctrl/Cmd+C):', `{plan_js_safe}`);
                }}
                menu.classList.remove('show');
              }});
            }})();
            </script>
            """
            components.html(actions_html, height=80, scrolling=False)
        with col_right:
            st.write("")  # alignment
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

# footer
st.markdown("<div class='sb-muted'>Tip: add a username to persist plans across sessions (future feature).</div>", unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)
