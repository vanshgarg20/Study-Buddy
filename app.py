# app.py - Final Study Buddy (resources as plain text, tighter spacing, Plan ready ✅)
# Requirements: pip install streamlit requests reportlab
import re
import time
import io
import base64
import hashlib
import json
import ast
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

# ---------- session state ----------
if "demo_state" not in st.session_state:
    st.session_state["demo_state"] = {"plans": []}

def get_user_state(uid: str) -> Dict[str, Any]:
    return st.session_state["demo_state"]

def save_user_state(uid: str, state: Dict[str, Any]) -> None:
    st.session_state["demo_state"] = state

# ---------- helpers ----------
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

def safe_topics(t):
    """
    Robust formatter for 'topics' that may be:
      - list of strings
      - dict
      - string
      - None
      - other types
    Returns a plain string safe to display.
    """
    if isinstance(t, list):
        return ", ".join(str(x) for x in t)
    if isinstance(t, dict):
        return ", ".join(f"{k}: {v}" for k, v in t.items())
    if t is None:
        return "—"
    return str(t)

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
        # ensure Mon-Sun order in text version too
        desired_days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        day_map = {}
        for d in week.get("days", []):
            name = (d.get("day") or "").strip()
            nm = name[:3].title() if name else ""
            day_map[nm] = d.get("topics", [])
        for dd in desired_days:
            topics = day_map.get(dd)
            topics_str = safe_topics(topics) if topics is not None else "Rest / Catch-up / Self-study"
            lines.append(f"  - {dd}: {topics_str}")
        lines.append("")
    if plan.get("daily_template"):
        lines.append("Daily Template:")
        lines.append(f"  {plan.get('daily_template')}")
        lines.append("")
    if plan.get("resources"):
        lines.append("Recommended Resources:")
        for r in plan.get("resources", []):
            # reuse the same safe-formatting used in UI (title only)
            if isinstance(r, dict):
                title = r.get("title") or r.get("name") or str(r)
                lines.append(f"  - {title}")
            else:
                lines.append(f"  - {r}")
    return "\n".join(lines)

# ---------- PDF helper (clean - no UI calls) ----------
def generate_pdf_bytes_platypus(title: str, record: Dict[str, Any]) -> bytes:
    """
    Clean, professional PDF generator — Helvetica, forces Mon->Sun,
    uses safe_topics formatting and does NOT contain any UI (st.markdown).
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not installed")

    buffer = io.BytesIO()

    # Professional margins / A4
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Clean, professional styles (Sans-serif)
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leftIndent=12,
        leading=15,
        spaceAfter=2
    )

    def safe_topics_pdf(t):
        """Robust joining for topics used in PDF (lists, dicts, strings, None)."""
        if isinstance(t, list):
            return ", ".join(str(x) for x in t)
        if isinstance(t, dict):
            return ", ".join(f"{k}: {v}" for k, v in t.items())
        if t is None:
            return "—"
        return str(t)

    story = []

    # Title + meta
    story.append(Paragraph(escape(title), title_style))
    created_ts = int(record.get("created_at", time.time()))
    meta_txt = (
        f"Saved: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_ts))}  |  "
        f"Source: {escape(str(record.get('source','local')))}  |  "
        f"Weeks: {record.get('weeks')}"
    )
    story.append(Paragraph(meta_txt, body_style))
    story.append(Spacer(1, 8))

    # Summary
    answers = record.get("answers", {})
    story.append(Paragraph("<b>Summary</b>", heading_style))
    story.append(Paragraph(
        f"Weeks: {record.get('weeks','—')}<br/>"
        f"Skill level: {escape(str(answers.get('skill_level','—')))}<br/>"
        f"Hours/day: {escape(str(answers.get('hours_per_day','—')))}",
        body_style
    ))
    if answers.get("goal"):
        story.append(Paragraph(f"<b>Goal:</b> {escape(str(answers.get('goal')))}", body_style))
    story.append(Spacer(1, 10))

    # Study Plan (force Mon-Sun)
    story.append(Paragraph("<b>Study Plan</b>", heading_style))
    plan = record.get("plan", {})
    week_list = plan.get("weeks", [])
    desired_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for w_i, week in enumerate(week_list, start=1):
        story.append(Paragraph(f"Week {w_i}", heading_style))

        # Build map short-day -> topics
        day_map = {}
        for d in week.get("days", []):
            name = (d.get("day") or "").strip()
            nm = name[:3].title() if name else ""
            day_map[nm] = d.get("topics", [])

        for dd in desired_days:
            topics = day_map.get(dd)
            if topics:
                topics_str = escape(safe_topics_pdf(topics))
                story.append(Paragraph(f"• <b>{dd}:</b> {topics_str}", bullet_style))
            else:
                story.append(Paragraph(f"• <b>{dd}:</b> Rest / Catch-up / Self-study", bullet_style))

        story.append(Spacer(1, 8))

    # Daily template
    daily_template = plan.get("daily_template")
    if daily_template:
        story.append(Paragraph("<b>Daily Template</b>", heading_style))
        story.append(Paragraph(escape(daily_template), body_style))
        story.append(Spacer(1, 8))

    # Resources (titles only)
    resources = plan.get("resources", [])
    if resources:
        story.append(Paragraph("<b>Recommended Resources</b>", heading_style))
        for r in resources:
            if isinstance(r, dict):
                title = r.get("title") or r.get("name") or str(r)
            else:
                title = str(r)
            story.append(Paragraph(f"• {escape(title)}", bullet_style))
        story.append(Spacer(1, 10))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ---------- domain modules ----------
DOMAIN_MODULES = {
    "dsa":[["Arrays","Strings","Two Pointers"],["Linked Lists","Stacks","Queues"],["Trees","BST","Trie"],["Graphs","BFS/DFS","Shortest Paths"],["Sorting","Searching","Hashing"],["Dynamic Programming","Greedy","Backtracking"]],
    "ml":[["Math & Linear Algebra","Probability","Statistics"],["Supervised Learning","Regression","Classification"],["Neural Networks","CNNs","RNNs"],["Optimization","Loss Functions","Regularization"],["Deployment","Model Serving","Monitoring"],["Advanced Topics","Transformers","Self-supervised"]],
    "web":[["HTML & CSS","DOM","Accessibility"],["JavaScript Basics","ES6+","DOM Manipulation"],["Frontend Framework","React/Vue","State Management"],["Backend Basics","APIs","Databases"],["Auth & Security","Testing","Deployment"],["Performance","Caching","Scaling"]],
    "db":[["RDBMS Basics","SQL Queries","Joins"],["Indexes","Query Optimization","Transactions"],["NoSQL Basics","Document Stores","Key-Value DB"],["Data Modeling","Normalization","Denormalization"],["Replication","Sharding","Backup/Restore"],["Analytics","OLAP","Data Warehousing"]],
    "cv":[["Image Processing","Filters","Transforms"],["Classical CV","Features","SIFT/ORB"],["Deep CV","CNNs","Object Detection"],["Segmentation","Keypoints","Pose Estimation"],["Data Augmentation","Training Tricks","Transfer Learning"],["Deployment","Edge Models","Optimization"]],
    "career":[["Resume & Profile","Formatting","Keywords"],["Behavioral Questions","STAR Method","Story Crafting"],["Mock Interviews","Problem Solving","Pair Practice"],["Company Research","Role Mapping","Expectations"],["System Design / Case Study","High-level Thinking","Trade-offs"],["Negotiation & HR","Offer Review","Salary Discussion"]],
    "default":[["Introduction","Core Concepts","Motivation"],["Tools & Setup","Libraries","Environment"],["Core Algorithms","Patterns","Practice"],["Mini-project","Integration","Testing"],["Optimization","Scaling","Evaluation"],["Advanced Concepts","Papers","Further Reading"]]
}

def detect_domain(tokens: List[str]) -> str:
    tset = set(tokens)
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
    hours = answers.get("hours_per_day", 2)
    skill = answers.get("skill_level", "beginner")
    if domain == "career":
        daily_template = f"{hours} hours/day: 40% practice (mock + problems) • 30% story & resume work • 30% research & interviews"
        resources = [{"name":"Resume templates & review guides","link": "https://example.com/resume-templates"},
                     {"name":"Common behavioral Q&A (STAR examples)","link":"https://example.com/star-examples"},
                     {"name":"Mock interview platforms (Pramp, InterviewBuddy)","link":"https://www.pramp.com/"}]
    elif domain == "dsa":
        daily_template = f"{hours} hours/day: 60% problem practice + 30% concept review + 10% mock tests"
        resources = [{"name":"Top LeetCode lists","link":"https://leetcode.com/explore/featured/card/top-interview-questions/"},
                     {"name":"CLRS / EPI chapters (selected)","link":"https://en.wikipedia.org/wiki/Introduction_to_Algorithms"},
                     {"name":"Interactive practice: Codeforces","link":"https://codeforces.com/"}]
    elif domain == "ml":
        daily_template = f"{hours} hours/day: theory + experiments + reading (adjust by skill={skill})"
        resources = [{"name":"Practical ML course (Coursera/fast.ai)","link":"https://www.coursera.org/"},
                     {"name":"Kaggle notebooks & tutorials","link":"https://www.kaggle.com/"},
                     {"name":"Model deployment tutorials","link":"https://www.tensorflow.org/serving"}]
    else:
        daily_template = f"{hours} hours/day: mix of theory + practice + mini project"
        resources = [f"Intro course about {topic}", f"YouTube playlists for {topic}", "Documentation & official guides"]
    plan = {"weeks": weeks_list, "daily_template": daily_template, "resources": resources}
    return plan

def local_create_plan(uid: str, topic: str, weeks: int, answers: Dict[str, Any]) -> Dict[str, Any]:
    created = int(time.time())
    rec = {"topic": topic.strip() or "Topic","weeks": int(max(1, weeks)),"answers": answers,"plan": build_topic_daylist(topic.strip() or "Topic", int(weeks), answers),"quiz": [],"created_at": created,"source": "local"}
    return rec

# ---------- improved remote wrapper with retries (silent) ----------
def generate_plan_via_api(uid: str, topic: str, weeks: int, answers: Dict[str, Any]) -> Dict[str, Any]:
    api_key = st.secrets.get("API_KEY")
    api_url = st.secrets.get("API_URL")
    if not api_key or not api_url:
        return local_create_plan(uid, topic, weeks, answers)

    preferred = st.secrets.get("GROQ_MODEL")
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += [
        "llama-3.1-70b-versatile",
        "mixtral-8x7b",
        "llama-3.1-instruct",
        "mixtral-instruct",
        "llama-2-13b-chat",
    ]
    seen = set()
    models = [m for m in candidates if not (m in seen or seen.add(m))]

    system_msg = {"role":"system","content":"You are a helpful assistant that produces structured study plans. Reply with valid JSON only using the schema: {\"topic\":string, \"plan\": {\"weeks\": [{\"days\":[{\"day\":\"Mon\",\"topics\":[...]}, ... ]}, ...], \"daily_template\":string, \"resources\":[...]}}. No extra text."}
    user_instructions = {"role":"user","content": f"Generate a study plan for topic: \"{topic}\". Weeks: {weeks}. Skill: {answers.get('skill_level')}. Hours per day: {answers.get('hours_per_day')}. Goal: {answers.get('goal')}. Output MUST be JSON matching the schema."}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload_base = {"messages": [system_msg, user_instructions], "max_tokens":1200, "temperature":0.2}

    for model_name in models:
        payload = payload_base.copy()
        payload["model"] = model_name
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=20)
            if resp.status_code >= 400:
                try:
                    err = resp.json()
                    msg = err.get("error", {}).get("message") if isinstance(err, dict) else str(err)
                    if msg and "model" in msg and ("does not exist" in msg or "not found" in msg or "not have access" in msg):
                        continue
                except Exception:
                    if "model" in resp.text and ("does not exist" in resp.text or "not found" in resp.text or "not have access" in resp.text):
                        continue
                break

            data = resp.json()
            plan_obj = None; topic_out = topic
            if isinstance(data, dict) and "choices" in data:
                choices = data.get("choices", [])
                assistant_text = ""
                if choices and isinstance(choices[0], dict):
                    assistant_text = choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
                try:
                    parsed = json.loads(assistant_text)
                    plan_obj = parsed.get("plan") if isinstance(parsed, dict) else None
                    topic_out = parsed.get("topic", topic) if isinstance(parsed, dict) else topic
                except Exception:
                    m = re.search(r'\{.*\}', assistant_text, flags=re.S)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                            plan_obj = parsed.get("plan") if isinstance(parsed, dict) else None
                            topic_out = parsed.get("topic", topic) if isinstance(parsed, dict) else topic
                        except Exception:
                            plan_obj = None
            else:
                plan_obj = data.get("plan") if isinstance(data, dict) else None
                topic_out = data.get("topic", topic) if isinstance(data, dict) else topic

            if plan_obj and isinstance(plan_obj, dict):
                rec = {"topic": topic_out, "weeks": int(max(1, weeks)), "answers": answers, "plan": plan_obj, "quiz": data.get("quiz", [] ) if isinstance(data, dict) else [], "created_at": int(time.time()), "source": f"remote ({model_name})"}
                return rec
            continue

        except Exception:
            continue

    return local_create_plan(uid, topic, weeks, answers)

# ---------- UI (styles + form + rendering) ----------
st.set_page_config(page_title="Study Buddy", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
/* tightened spacing */
.sb-page{display:flex;justify-content:center;padding:12px 0 20px}
.sb-container{width:100%;max-width:980px;padding:0 16px}
.sb-title{font-size:44px;font-weight:900;margin:2px 0 8px 0;background:linear-gradient(90deg,#00d4ff,#5b7cff,#c86dd7,#ff7abd);background-size:400% 400%;-webkit-background-clip:text;color:transparent;animation:floatGradient 12s ease-in-out infinite;text-shadow:0 6px 18px rgba(0,0,0,0.45)}
@keyframes floatGradient{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.sb-sub{color:#9aa3b2;margin-bottom:12px;font-size:14px}
.sb-card{background:rgba(255,255,255,0.02);border-radius:10px;padding:14px 16px;margin-bottom:12px;box-shadow:0 6px 18px rgba(0,0,0,0.45)}
.sb-topic{font-size:18px;font-weight:800;margin-bottom:6px;color:#fff}
.sb-meta{color:#9aa3b2;margin-bottom:8px;font-size:13px}
.sb-week{font-size:14px;font-weight:700;margin-top:10px;margin-bottom:6px}
.sb-day{margin-left:12px;margin-bottom:4px;color:#e6eef8}
.actions-wrapper{display:flex;align-items:center;gap:8px;margin-top:6px}
.actions-pill{background:rgba(255,255,255,0.03);color:#f4f7fb;border-radius:999px;padding:6px 10px;border:1px solid rgba(255,255,255,0.04);cursor:pointer;font-weight:700;display:inline-flex;align-items:center;gap:8px}
.actions-dropdown{position:absolute;top:40px;left:0;background:rgba(20,24,28,0.98);border-radius:10px;padding:6px;min-width:180px;box-shadow:0 10px 30px rgba(0,0,0,0.5);border:1px solid rgba(255,255,255,0.03);z-index:9999;display:none;flex-direction:column;gap:6px}
.actions-dropdown.show{display:flex}.actions-item{background:transparent;color:#e6eef8;border-radius:8px;padding:8px 12px;cursor:pointer;font-weight:600;display:flex;gap:8px;align-items:center}.actions-item:hover{background:rgba(255,255,255,0.02);transform:translateY(-1px)}.stButton>button,.stButton>div>button{border-radius:10px !important;padding:8px 12px !important;font-weight:700 !important}
@media (max-width:720px){.sb-title{font-size:32px}.actions-dropdown{left:auto;right:0}}
</style>""", unsafe_allow_html=True)

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
        if plans:
            plans[0] = rec
        else:
            plans.insert(0, rec)
        state["plans"] = plans
        save_user_state(uid, state)
        # changed here: use emoji ✅ as requested
        st.success("Plan ready ✅")
        status_saved = True

st.markdown("<div class='sb-page'><div class='sb-container'>", unsafe_allow_html=True)
st.markdown("<div class='sb-title'>Study Buddy</div>", unsafe_allow_html=True)
st.markdown("<div class='sb-sub'>Personalized study plans tuned to your topic — remote AI boosts suggestions when accessible.</div>", unsafe_allow_html=True)
if status_saved:
    st.success("Saved to memory.")
st.markdown("---")

if st.session_state.get("username_input","").strip():
    st.markdown(f"<div class='sb-meta sb-muted'>Logged as: <strong>{st.session_state.get('username_input')}</strong></div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='sb-meta sb-muted'>No username provided — plans saved to this browser session only.</div>", unsafe_allow_html=True)

state = get_user_state(uid)
plans = state.get("plans", [])

# ---------- helper to normalize resource items ----------
def normalize_resource(item):
    """
    Accepts:
      - dicts like {'title':..,'url':..} or {'name':..,'link':..}
      - strings (plain URL or title)
      - stringified dicts ("{'name':'X','link':'http...'}")
    Returns (title, url_or_none)
    """
    # dict case
    if isinstance(item, dict):
        title = item.get("title") or item.get("name") or item.get("label") or str(item)
        url = item.get("url") or item.get("link") or item.get("href") or None
        return title, url
    # string case - try to parse stringified dict
    if isinstance(item, str):
        s = item.strip()
        # attempt safe literal_eval if it looks like a dict
        if (s.startswith("{") and s.endswith("}")) or ("'link'" in s or '"link"' in s or "'url'" in s):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    title = parsed.get("title") or parsed.get("name") or str(parsed)
                    url = parsed.get("url") or parsed.get("link") or None
                    return title, url
            except Exception:
                pass
        # if the string contains http(s) use it as url
        m = re.search(r"(https?://[^\s'\"<>]+)", s)
        if m:
            url = m.group(1)
            # try to extract a human title from before URL or just use the URL
            title_candidate = re.sub(r"https?://[^\s'\"<>]+", "", s).strip()
            title = title_candidate or url
            return title, url
        # otherwise treat it as a plain title and build a youtube search link (but we'll only use title in UI)
        title = s
        url = f"https://www.youtube.com/results?search_query={re.sub(r'\\s+', '+', title)}"
        return title, url
    # otherwise fallback
    return str(item), None

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

        st.markdown(f"- **Weeks:** {p.get('weeks','—')}  •  **Skill:** {answers.get('skill_level','—')}  •  **Hours/day:** {answers.get('hours_per_day','—')}")
        if answers.get('goal'):
            goal_raw = answers.get('goal')
            goal_display = goal_raw.replace("\n", "  \n")
            st.markdown(f"- **Goal:** {goal_display}")
        st.markdown("")

        weeks_list = plan_data.get("weeks", [])
        # UI: force Mon -> Sun
        for w_i, week in enumerate(weeks_list, start=1):
            st.markdown(f"<div class='sb-week'>Week {w_i}</div>", unsafe_allow_html=True)

            desired_days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            day_map_ui = {}
            for d in week.get("days", []):
                name = (d.get("day") or "").strip()
                nm = name[:3].title() if name else ""
                day_map_ui[nm] = d.get("topics", [])

            for dd in desired_days:
                topics = day_map_ui.get(dd)
                if topics:
                    topics_str = safe_topics(topics)
                    st.markdown(f"<div class='sb-day'>• <strong>{dd}:</strong> {escape(topics_str)}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='sb-day'>• <strong>{dd}:</strong> Rest / Catch-up / Self-study</div>", unsafe_allow_html=True)

            st.markdown("")

        if plan_data.get("daily_template"):
            st.markdown("**Daily Template:**")
            st.markdown(f"- {plan_data.get('daily_template')}")
        resources = plan_data.get("resources", [])
        if resources:
            st.markdown("**Recommended Resources:**")
            # show only titles (no clickable links) as requested
            for r in resources:
                title, url = normalize_resource(r)
                title = title or "Resource"
                safe_title = escape(str(title))
                st.markdown(f"- {safe_title}")

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
            components.html(actions_html, height=72, scrolling=False)
        with col_right:
            st.write("")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

st.markdown("<div class='sb-muted'>Tip: add a username to persist plans across sessions (future feature).</div>", unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)
