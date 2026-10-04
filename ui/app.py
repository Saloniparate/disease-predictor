import base64
import html
import sys
import time
from pathlib import Path

# Put the project root on the path so `src` and `utils` can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.predict import predict_top_k
from utils.precautions import get_precautions
from utils.symptoms_list import symptom_labels

st.set_page_config(
    page_title="AI Disease Prediction System", layout="wide"
)

# ---------------------------------------------------------------- themes
LIGHT = {
    "bg_a": "#f8f9ff", "bg_b": "#f3f0ff", "bg_c": "#eef6ff",
    "glow_a": "#e0e7ff", "glow_b": "#ede9fe",
    "hero": "linear-gradient(120deg, #312e81 0%, #4f46e5 55%, #7c3aed 100%)",
    "card": "rgba(255,255,255,0.72)", "card_border": "rgba(255,255,255,0.9)",
    "card_shadow": "0 10px 40px -12px rgba(79,70,229,0.18)",
    "tint_a": "rgba(238,242,255,0.95)", "tint_b": "rgba(245,243,255,0.95)",
    "text": "#1e1b4b", "muted": "#64748b", "accent": "#4f46e5",
    "tile": "linear-gradient(135deg, #e0e7ff, #ede9fe)", "rank_bg": "#e0e7ff",
    "input_bg": "#ffffff", "input_border": "#e0e7ff", "menu_bg": "#ffffff",
    "track": "#e0e7ff",
    "prec_bg": "rgba(255,255,255,0.8)", "prec_border": "#e0e7ff",
    "btn2_bg": "#ffffff", "btn2_border": "#c7d2fe", "btn2_text": "#4f46e5", "btn2_hover": "#eef2ff",
    "quick_bg": "#eef2ff", "quick_border": "#e0e7ff",
    "info_bg": "#ecfeff", "info_border": "#a5f3fc", "info_text": "#155e75",
    "warn_bg": "#fffbeb", "warn_border": "#fde68a", "warn_text": "#92400e",
}

DARK = {
    "bg_a": "#0b0d17", "bg_b": "#0f1222", "bg_c": "#111426",
    "glow_a": "rgba(99,102,241,0.14)", "glow_b": "rgba(139,92,246,0.10)",
    "hero": "linear-gradient(120deg, #1e1b4b 0%, #3730a3 55%, #5b21b6 100%)",
    "card": "rgba(24,27,46,0.75)", "card_border": "rgba(255,255,255,0.08)",
    "card_shadow": "0 10px 40px -12px rgba(0,0,0,0.6)",
    "tint_a": "rgba(30,33,60,0.85)", "tint_b": "rgba(26,24,52,0.85)",
    "text": "#e8eaf6", "muted": "#94a3b8", "accent": "#a5b4fc",
    "tile": "linear-gradient(135deg, rgba(99,102,241,0.28), rgba(139,92,246,0.2))",
    "rank_bg": "rgba(99,102,241,0.25)",
    "input_bg": "#151932", "input_border": "rgba(255,255,255,0.1)", "menu_bg": "#181b2e",
    "track": "#262a45",
    "prec_bg": "rgba(255,255,255,0.04)", "prec_border": "rgba(255,255,255,0.08)",
    "btn2_bg": "rgba(255,255,255,0.05)", "btn2_border": "rgba(255,255,255,0.14)",
    "btn2_text": "#c7d2fe", "btn2_hover": "rgba(129,140,248,0.15)",
    "quick_bg": "rgba(255,255,255,0.05)", "quick_border": "rgba(255,255,255,0.1)",
    "info_bg": "rgba(34,211,238,0.10)", "info_border": "rgba(34,211,238,0.30)", "info_text": "#67e8f9",
    "warn_bg": "rgba(245,158,11,0.12)", "warn_border": "rgba(245,158,11,0.35)", "warn_text": "#fcd34d",
}

st.session_state.setdefault("dark_mode", False)
T = DARK if st.session_state["dark_mode"] else LIGHT

QUICK_PICKS = [
    q
    for q in ["High fever", "Cough", "Headache", "Fatigue",
              "Vomiting", "Nausea", "Chills", "Skin rash"]
    if q in symptom_labels
]

# ---------------------------------------------------------------- icons (Lucide)
# st.html strips inline <svg>, so every icon is a base64 <img>.
ICONS = {
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "stethoscope": '<path d="M11 2v2"/><path d="M5 2v2"/><path d="M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1"/><path d="M8 15a6 6 0 0 0 12 0v-3"/><circle cx="20" cy="10" r="2"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
}


def data_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def icon(name, color="#ffffff", size=20):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )
    return f'<img src="{data_uri(svg)}" width="{size}" height="{size}" alt="" style="display:block">'


# ---------------------------------------------------------------- html builders
def load_css():
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    root = ":root{" + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in T.items()) + "}"
    st.markdown(f"<style>{root}{css}</style>", unsafe_allow_html=True)


def hero_html():
    return f"""
    <div class="hero">
      <div class="hero-text">
        <div class="hero-badge">{icon("activity", "#ffffff", 14)} AI MEDICAL ASSISTANT</div>
        <div class="hero-title">AI Disease Prediction System</div>
        <div class="hero-sub">Select the symptoms you are experiencing and our machine
        learning model will suggest the most likely conditions, with a confidence score
        and recommended precautions.</div>
      </div>
      <div class="hero-art">{icon("heart", "#ffffff", 64)}</div>
    </div>
    """


def section_title(ico, title, sub):
    return (
        f'<div class="section-title"><span class="ico">{icon(ico, T["accent"], 20)}</span>{title}</div>'
        f'<div class="section-sub">{sub}</div>'
    )


def notice(text, kind="info"):
    name = "alert" if kind == "warn" else "info"
    color = T["warn_text"] if kind == "warn" else T["info_text"]
    return f'<div class="notice {kind}">{icon(name, color, 18)}<span>{html.escape(text)}</span></div>'


def gauge_html(pct):
    """Semi-circle gradient gauge, drawn as a base64 SVG image (animation runs inside it)."""
    pct = max(0, min(pct, 100))
    arc = "M 20 100 A 80 80 0 0 1 180 100"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 125" width="260" height="162">
<defs>
<linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#22d3ee"/><stop offset="0.5" stop-color="#6366f1"/><stop offset="1" stop-color="#8b5cf6"/>
</linearGradient>
<filter id="glow" x="-20%" y="-30%" width="140%" height="160%">
<feGaussianBlur stdDeviation="3" result="b"/>
<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter>
</defs>
<style>@keyframes fill{{from{{stroke-dashoffset:100}}}} .arc{{animation:fill 1.3s cubic-bezier(.22,1,.36,1)}}</style>
<path d="{arc}" pathLength="100" fill="none" stroke="{T['track']}" stroke-width="14" stroke-linecap="round"/>
<path class="arc" d="{arc}" pathLength="100" fill="none" stroke="url(#g)" stroke-width="14"
 stroke-linecap="round" stroke-dasharray="100" stroke-dashoffset="{100 - pct}" filter="url(#glow)"/>
<text x="100" y="90" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="34" font-weight="800" fill="{T['text']}">{pct:.0f}%</text>
<text x="100" y="110" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="10" letter-spacing="1.5" fill="{T['muted']}">CONFIDENCE</text>
</svg>"""
    return f'<div class="gauge-wrap"><img src="{data_uri(svg)}" width="260" height="162" alt="Confidence gauge"></div>'


def results_html(top3):
    disease, conf = top3[0]
    bars = ""
    for i, (name, score) in enumerate(top3, start=1):
        bars += f"""
        <div class="bar-row">
          <div class="bar-head">
            <span class="rank">{i}</span>
            <span class="bar-name">{html.escape(str(name))}</span>
            <span class="bar-pct">{score}%</span>
          </div>
          <div class="track"><div class="fill" style="width:{min(score, 100)}%"></div></div>
        </div>
        """
    low = ""
    if conf < 40:
        low = notice(
            "Confidence is low. Selecting more of your symptoms usually gives a clearer result.",
            "info",
        )
    return f"""
    <div class="card">
      {section_title("activity", "Prediction Result", "Most likely condition based on your symptoms")}
      {gauge_html(conf)}
      <div style="text-align:center">
        <span class="badge">{icon("stethoscope", "#ffffff", 20)} {html.escape(str(disease))}</span>
      </div>
      <div class="bars-title">Top 3 predictions</div>
      {bars}
      {low}
    </div>
    """


def empty_html():
    return f"""
    <div class="card">
      {section_title("activity", "Prediction Result", "Your analysis will appear here")}
      <div class="empty">
        <div class="ico">{icon("heart", T["accent"], 30)}</div>
        <b>No prediction yet</b>
        Pick your symptoms on the left and press Predict Disease.
      </div>
    </div>
    """


def precautions_html(disease, items):
    cards = "".join(
        f'<div class="prec"><span class="prec-ico">{icon("check", "#ffffff", 18)}</span>'
        f"<span>{html.escape(p)}</span></div>"
        for p in items
    )
    return f"""
    <div class="card tint">
      {section_title("shield", "Recommended Precautions", f"General guidance for {html.escape(str(disease))}")}
      <div class="prec-grid">{cards}</div>
    </div>
    """


def disclaimer_html():
    return (
        f'<div class="disclaimer">{icon("alert", T["warn_text"], 22)}'
        "<span><b>Medical disclaimer:</b> this is a student machine-learning project, "
        "not a medical diagnosis. Please consult a doctor for any health concern.</span></div>"
    )


# ---------------------------------------------------------------- callbacks
def add_symptom(label):
    current = list(st.session_state.get("selected", []))
    if label not in current:
        current.append(label)
    st.session_state["selected"] = current


def reset():
    st.session_state["selected"] = []
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)


# ---------------------------------------------------------------- page
load_css()

_, toggle_col = st.columns([5, 1.4])
with toggle_col:
    st.toggle(":material/dark_mode: Dark mode", key="dark_mode")

st.html(hero_html())

left, right = st.columns(2, gap="large")

# ----- LEFT: input
with left:
    with st.container(key="input_card"):
        st.html(section_title("search", "Select Your Symptoms",
                              "Search the list or tap a common symptom below"))
        st.multiselect(
            "Symptoms",
            options=sorted(symptom_labels),
            key="selected",
            placeholder="Search symptoms, e.g. fever, headache...",
            label_visibility="collapsed",
        )
        st.caption(f"{len(st.session_state.get('selected', []))} symptom(s) selected")

        if QUICK_PICKS:
            st.html('<div class="mini-label">Common symptoms</div>')
            with st.container(key="quick"):
                for start in range(0, len(QUICK_PICKS), 4):
                    cols = st.columns(4)
                    for col, label in zip(cols, QUICK_PICKS[start:start + 4]):
                        col.button(label, key=f"q_{label}",
                                   on_click=add_symptom, args=(label,))

        st.write("")
        b1, b2 = st.columns([2, 1])
        predict = b1.button("Predict Disease", type="primary",
                            help="Run the model on the selected symptoms")
        b2.button("Reset", on_click=reset)

        run = False
        if predict:
            if st.session_state.get("selected"):
                st.session_state.pop("error", None)
                run = True
            else:
                st.session_state["error"] = "Please select at least one symptom."
                st.session_state.pop("result", None)
        if "error" in st.session_state:
            st.html(notice(st.session_state["error"], "warn"))

# ----- RIGHT: results
with right:
    if run:
        with st.spinner("Analysing your symptoms..."):
            time.sleep(0.8)  # short pause so the loading state is visible
            symptoms = [symptom_labels[l] for l in st.session_state["selected"]]
            st.session_state["result"] = predict_top_k(symptoms, k=3)

    result = st.session_state.get("result")
    st.html(results_html(result) if result else empty_html())

# ----- Precautions (full width)
if result:
    disease = result[0][0]
    items = get_precautions(disease)
    if items:
        st.html('<div class="row-gap"></div>')
        st.html(precautions_html(disease, items))

st.html('<div class="row-gap"></div>')
st.html(disclaimer_html())