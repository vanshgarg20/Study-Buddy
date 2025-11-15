# app.py - Final Study Buddy (generate replaces previous plan; copy + download only)
# Requirements: pip install streamlit reportlab

import re
import time
import io
import base64
from typing import Dict, Any, List, Optional
from html import escape

import streamlit as st
import streamlit.components.v1 as components

# ReportLab (optional)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --------------------
# app state
if "demo_state" not in st.session_state:
    st.session_state["demo_state"] = {"plans": []}

def get_user_state(uid: str) -> Dict[str, Any]:
    return st.session_state["demo_state"]

def save_user_state(uid: str, state: Dict[str, Any]) -> None:
    st.session_state["demo_state"] = state

# --------------------
# page config + styles
st.set_page_config(page_title="Study Buddy", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .sb-page { display:flex; justify-content:center; padding:20px 0 40px; }
    .sb-container { width:100%; max-width:980px; padding:0 20px; }
    .sb-title { font-size:48px; font-weight:900; margin:4px 0 6px 0;
      background: linear-gradient(90deg,#00d4ff,#5b7cff,#c86dd7,#ff7abd);
      background-size: 400% 400%; -webkit-background-clip: text; color: transparent;
      animation: floatGradient 12s ease-in-out infinite; text-shadow: 0 6px 18px rgba(0,0,0,0.45);
    }
    @keyframes floatGradient { 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }
    .sb-sub { color:#9aa3b2; margin-bottom:18px; font-size:14px; }
    .sb-card { background: rgba(255,255,255,0.02); border-radius:12px; padding:18px 20px; margin-bottom:18px; box-shadow: 0 6px 18px rgba(0,0,0,0.45); }
    .sb-topic { font-size:20px; font-weight:700; margin-bottom:6px; color:#fff; }
    .sb-meta { color:#9aa3b2; margin-bottom:10px; font-size:13px; }
    .sb-week { font-size:15px; font-weight:700; margin-top:12px; margin-bottom:6px; }
    .sb-day { margin-left:14px; margin-bottom:4px; color:#e6eef8; }
    .actions-wrapper { display:flex; align-items:center; gap:10px; margin-top:6px; }
    .actions-pill { background: rgba(255,255,255,0.03); color: #f4f7fb; border-radius: 999px; padding: 8px 12px; border: 1px solid rgba(255,255,255,0.04); cursor: pointer; font-weight:700; display:inline-flex; align-items:center; gap:8px; }
    .actions-dropdown { position: absolute; top: 44px; left: 0; background: rgba(20,24,28,0.98); border-radius:10px; padding:8px; min-width:200px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.03); z-index: 9999; display:none; flex-direction:column; gap:6px; }
    .actions-dropdown.show { display:flex; }
    .actions-item { background: transparent; color: #e6eef8; border-radius:8px; padding:8px 12px; cursor: pointer; font-weight:600; display:flex; gap:8px; align-items:center; }
    .actions-item:hover { background: rgba(255,255,255,0.02); transform: translateY(-1px); }
    .stButton>button, .stButton>div>button { border-radius:10px !important; padding:8px 12px !important; font-weight:700 !important; }
    @media (max-width:720px) { .sb-title { font-size:36px; } .actions-dropdown { left: auto; right: 0; } }
    </style>
    """, unsafe_allow_html=True
)

# --------------------
# helpers
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

# PDF generator
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

# local plan generator
def local_create_plan(uid: str, topic: str, weeks: int, answers: Dict[str, Any]) -> Dict[str, Any]:
    weeks = max(1, int(weeks))
    created = int(time.time())
    topic_clean = (topic or "Topic").strip().title()
    base_days = [
        {"day": "Mon", "topics": [f"Introduction to {topic_clean}", "Types of learning"]},
        {"day": "Tue", "topics": ["Supervised Learning", "Unsupervised Learning"]},
        {"day": "Wed", "topics": ["Regression", "Classification"]},
        {"day": "Thu", "topics": ["Clustering", "Dimensionality Reduction"]},
        {"day": "Fri", "topics": ["Neural Networks", "Deep Learning basics"]},
        {"day": "Sat", "topics": ["Frameworks & Tools"]},
        {"day": "Sun", "topics": ["Practice", "Mini Project"]},
    ]
    weeks_list: List[Dict[str, Any]] = []
    for w in range(weeks):
        shift = w % len(base_days)
        week_days = []
        for d in base_days:
            topics = list(d["topics"])
            if shift:
                topics = topics[shift:] + topics[:shift]
            week_days.append({"day": d["day"], "topics": topics})
        weeks_list.append({"days": week_days})
    plan = {
        "weeks": weeks_list,
        "daily_template": f"{answers.get('hours_per_day',2)} hours/day: 1 hour theory + 1 hour practice",
        "resources": ["Intro course (Coursera)", "Book: Python Machine Learning", "YouTube playlists"]
    }
    rec = {
        "topic": topic,
        "weeks": weeks,
        "answers": answers,
        "plan": plan,
        "quiz": [],
        "created_at": created,
        "source": "local"
    }
    return rec

# --------------------
# Sidebar form - session_state-safe (initialize defaults once, then use key= only)
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

# generate handling — REPLACE existing plan (if any) for this uid
status_saved = False
uid = (st.session_state.get("username_input","").strip() or "session_user")
if submit:
    if not st.session_state.get("topic_input","").strip():
        st.error("Please enter a topic.")
    else:
        answers = {"skill_level": st.session_state.get("skill_input"), "hours_per_day": st.session_state.get("hours_input"), "goal": st.session_state.get("goal_input")}
        rec = local_create_plan(uid, st.session_state["topic_input"].strip(), int(st.session_state["weeks_input"]), answers)
        state = get_user_state(uid)
        plans = state.get("plans", [])
        # REPLACE logic: if there is an existing plan for this uid, replace the first one; otherwise insert
        if plans:
            plans[0] = rec
        else:
            plans.insert(0, rec)
        state["plans"] = plans
        save_user_state(uid, state)
        st.success("Plan created and saved ✅ (replaced previous plan)")
        status_saved = True

# header
st.markdown("<div class='sb-page'><div class='sb-container'>", unsafe_allow_html=True)
st.markdown("<div class='sb-title'>Study Buddy</div>", unsafe_allow_html=True)
st.markdown("<div class='sb-sub'>Simple, clean study plans — created instantly.</div>", unsafe_allow_html=True)
if status_saved:
    st.success("Saved to memory.")
st.markdown("---")

if st.session_state.get("username_input","").strip():
    st.markdown(f"<div class='sb-meta sb-muted'>Logged as: <strong>{st.session_state.get('username_input')}</strong></div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='sb-meta sb-muted'>No username provided — plans saved to this browser session only.</div>", unsafe_allow_html=True)

# render plans (only copy + download actions)
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

        # resources & template
        if plan_data.get("daily_template"):
            st.markdown("**Daily Template:**")
            st.markdown(f"- {plan_data.get('daily_template')}")
        resources = plan_data.get("resources", [])
        if resources:
            st.markdown("**Recommended Resources:**")
            for r in resources:
                st.markdown(f"- {r}")

        # Prepare PDF data URI
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

        # actions: only download + copy (no delete)
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
            components.html(actions_html, height=80, scrolling=False)
        with col_right:
            st.write("")  # keep layout alignment

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

# footer
st.markdown("<div class='sb-muted'>Tip: keep a username if you want consistent saved plans across sessions with future persistent storage.</div>", unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)
