import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
import json
from datetime import date, datetime
import plotly.graph_objects as go
import os
import base64
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LungSight AI · Clinical Intelligence",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("lung_cancer_model.pkl")

model = load_model()
MODEL_ACCURACY = 0.85

# ─────────────────────────────────────────────
# PATIENT HISTORY
# ─────────────────────────────────────────────
HISTORY_FILE = "patient_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(records):
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)

def add_record(record):
    records = load_history()
    records.insert(0, record)
    save_history(records[:50])

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "auth": False, "username": "", "step": 0,
    "risk": None, "pred_css": "", "snap": None,
    "show_results": False,
    "age": 40, "gender": "Male",
    "smoking": 3, "alcohol_use": 2, "balanced_diet": 5,
    "obesity": 3, "snoring": 2,
    "air_pollution": 3, "passive_smoker": 2,
    "dust_allergy": 2, "occupational": 3,
    "genetic_risk": 3, "chronic_lung": 2, "frequent_cold": 2,
    "chest_pain": 2, "coughing_blood": 0, "fatigue": 3,
    "weight_loss": 1, "breath": 2, "wheezing": 1,
    "swallow": 0, "clubbing": 0, "dry_cough": 2,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:      #03080f;
    --s1:      #071220;
    --s2:      #0c1e35;
    --border:  rgba(30,160,255,0.13);
    --b2:      rgba(30,160,255,0.25);
    --accent:  #1ea0ff;
    --a2:      #0055cc;
    --teal:    #00d4aa;
    --danger:  #ff3355;
    --warn:    #ff9900;
    --safe:    #00e5a0;
    --text:    #ddeeff;
    --muted:   rgba(180,220,255,0.4);
    --mono:    'JetBrains Mono', monospace;
    --display: 'Outfit', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: var(--display) !important;
    background: var(--bg) !important;
    color: var(--text);
}
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 100% 60% at 20% -10%, rgba(0,80,200,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 80% 50% at 90% 90%,  rgba(0,200,160,0.07) 0%, transparent 55%);
    background-attachment: fixed;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
[data-testid="stSidebar"] { background: var(--s1) !important; border-right: 1px solid var(--border); }

.step-card {
    background: var(--s1); border: 1px solid var(--border);
    border-radius: 20px; padding: 28px 32px; margin-bottom: 18px;
    position: relative; overflow: hidden;
}
.step-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--a2), var(--accent)); opacity: 0;
}
.step-card.active { border-color: var(--b2); }
.step-card.active::before { opacity: 1; }
.step-card.done { border-color: rgba(0,212,170,0.2); }
.step-card.done::before { background: linear-gradient(90deg,#006644,var(--teal)); opacity:1; }
.step-card.locked { opacity: 0.35; pointer-events: none; }
.step-header { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.step-num {
    width:34px; height:34px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-size:12px; font-weight:700; flex-shrink:0;
}
.step-num.active { background:rgba(30,160,255,0.18); border:1.5px solid var(--accent); color:var(--accent); }
.step-num.done   { background:rgba(0,212,170,0.15);  border:1.5px solid var(--teal);   color:var(--teal); }
.step-num.locked { background:rgba(255,255,255,0.04); border:1.5px solid rgba(255,255,255,0.1); color:var(--muted); }
.step-title { font-size:17px; font-weight:700; color:var(--text); }
.step-desc  { font-size:12px; color:var(--muted); margin-left:48px; margin-bottom:24px; line-height:1.5; }

[data-testid="stSlider"] label {
    font-family:var(--mono) !important; font-size:10px !important;
    letter-spacing:1.5px !important; text-transform:uppercase !important; color:var(--muted) !important;
}
[data-testid="stSlider"] > div > div > div { background:linear-gradient(90deg,var(--a2),var(--accent)) !important; }
[data-testid="stSlider"] > div > div > div > div { background:#fff !important; box-shadow:0 0 10px var(--accent) !important; }

.stButton > button {
    background:linear-gradient(135deg,var(--a2),var(--accent)) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-family:var(--mono) !important; font-size:10px !important;
    letter-spacing:2.5px !important; font-weight:500 !important;
    padding:13px 28px !important; transition:all 0.2s !important;
    box-shadow:0 6px 24px rgba(0,100,255,0.28) !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 10px 32px rgba(30,160,255,0.38) !important; }

.stTextInput input, .stDateInput input {
    background:var(--s2) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important;
    font-family:var(--mono) !important; font-size:13px !important;
}
.stTextInput input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(30,160,255,0.12) !important; }
.stSelectbox > div > div {
    background:var(--s2) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important;
}
.stTextArea textarea {
    background:var(--s2) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important;
    font-family:var(--mono) !important; font-size:12px !important;
}

.risk-banner {
    border-radius:24px; padding:44px; text-align:center; margin:24px 0;
    position:relative; overflow:hidden;
}
.risk-low    { background:rgba(0,229,160,0.07);  border:1.5px solid var(--safe); }
.risk-medium { background:rgba(255,153,0,0.07);  border:1.5px solid var(--warn); }
.risk-high   { background:rgba(255,51,85,0.07);  border:1.5px solid var(--danger); }
.risk-eyebrow { font-family:var(--mono); font-size:10px; letter-spacing:4px;
    text-transform:uppercase; color:var(--muted); margin-bottom:12px; }
.risk-big { font-size:68px; font-weight:900; letter-spacing:-3px; line-height:1; }
.risk-low  .risk-big { color:var(--safe); }
.risk-medium .risk-big { color:var(--warn); }
.risk-high .risk-big { color:var(--danger); }
.risk-sub { font-size:14px; color:var(--muted); margin-top:12px; max-width:480px; margin-left:auto; margin-right:auto; line-height:1.6; }
.risk-action { display:inline-block; margin-top:18px; padding:9px 22px;
    border-radius:8px; font-family:var(--mono); font-size:10px; letter-spacing:2px; font-weight:600; text-transform:uppercase; }
.risk-low .risk-action    { background:rgba(0,229,160,0.12); color:var(--safe);   border:1px solid rgba(0,229,160,0.3); }
.risk-medium .risk-action { background:rgba(255,153,0,0.12);  color:var(--warn);   border:1px solid rgba(255,153,0,0.3); }
.risk-high .risk-action   { background:rgba(255,51,85,0.12);  color:var(--danger); border:1px solid rgba(255,51,85,0.3); }

.metric-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:22px 0; }
.metric-box {
    background:var(--s1); border:1px solid var(--border);
    border-radius:16px; padding:18px 20px;
}
.metric-box .mlabel { font-family:var(--mono); font-size:9px; letter-spacing:3px;
    text-transform:uppercase; color:var(--muted); margin-bottom:5px; }
.metric-box .mvalue { font-size:28px; font-weight:800; color:var(--accent); letter-spacing:-1px; }
.metric-box .msub   { font-size:10px; color:var(--muted); margin-top:2px; }

.hist-head { font-family:var(--mono); font-size:9px; letter-spacing:2px;
    text-transform:uppercase; color:var(--muted); padding:8px 20px 12px;
    display:grid; grid-template-columns:120px 1fr 60px 80px 120px; gap:16px; }
.hist-row {
    display:grid; grid-template-columns:120px 1fr 60px 80px 120px;
    gap:16px; align-items:center; padding:12px 20px;
    border-radius:10px; border:1px solid var(--border); background:var(--s1);
    margin-bottom:8px; font-size:13px; transition:all 0.15s;
}
.hist-row:hover { border-color:var(--b2); background:var(--s2); }
.badge { display:inline-block; padding:3px 12px; border-radius:999px;
    font-family:var(--mono); font-size:10px; letter-spacing:1px; font-weight:600; }
.badge-low    { background:rgba(0,229,160,0.12); color:var(--safe);   border:1px solid rgba(0,229,160,0.25); }
.badge-medium { background:rgba(255,153,0,0.12);  color:var(--warn);   border:1px solid rgba(255,153,0,0.25); }
.badge-high   { background:rgba(255,51,85,0.12);  color:var(--danger); border:1px solid rgba(255,51,85,0.25); }

.wordmark { font-size:26px; font-weight:900; letter-spacing:-1px; color:#fff; }
.wordmark em { color:var(--accent); font-style:normal; }
.slabel { font-family:var(--mono); font-size:9px; letter-spacing:3px;
    text-transform:uppercase; color:var(--accent); margin-bottom:4px; display:block; }
.appt-block { background:var(--s1); border:1px solid var(--border);
    border-radius:20px; padding:28px 32px; margin-top:14px; }

.sp-bar { display:flex; align-items:center; gap:0; margin-bottom:32px; }
.sp-dot {
    width:30px; height:30px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-size:10px; font-weight:700; flex-shrink:0; transition:all 0.3s;
}
.sp-dot.done   { background:var(--teal); color:#000; }
.sp-dot.active { background:var(--accent); color:#fff; box-shadow:0 0 16px rgba(30,160,255,0.5); }
.sp-dot.future { background:var(--s2); color:var(--muted); border:1px solid var(--border); }
.sp-line { flex:1; height:2px; background:var(--border); margin:0 4px; }
.sp-line.done { background:var(--teal); }

hr { border-color:rgba(30,160,255,0.1) !important; margin:36px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LUNG SVG — rendered via components.html to avoid Streamlit sanitization
# ─────────────────────────────────────────────
def lung_component(damage_pct: float, height: int = 480) -> None:
    """Render the lung SVG + radar chart using st.components.v1.html so the SVG is not stripped."""
    if damage_pct > 0.6:
        r, g, b = 255, 51, 85
        ring_color = "#ff3355"
    elif damage_pct > 0.3:
        r, g, b = 255, 153, 0
        ring_color = "#ff9900"
    else:
        r, g, b = 0, 229, 160
        ring_color = "#00e5a0"

    fill_offset = int(290 * (1 - damage_pct))
    pct_int = int(damage_pct * 100)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&family=JetBrains+Mono:wght@400;500&display=swap');
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      body {{
        background: transparent;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: {height}px;
        font-family: 'Outfit', sans-serif;
      }}
      .lung-wrap {{
        background: rgba(7,18,32,0.8);
        border: 1px solid rgba(30,160,255,0.13);
        border-radius: 20px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        position: relative;
        overflow: hidden;
      }}
      .lung-wrap::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, {ring_color}, transparent);
        opacity: 0.7;
      }}
      .lung-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: rgba(180,220,255,0.35);
        text-align: center;
        margin-top: 12px;
      }}
      .pulse-ring {{
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0; left: 0;
        pointer-events: none;
        overflow: hidden;
        border-radius: 20px;
      }}
      @keyframes pulseRing {{
        0%   {{ opacity: 0.4; transform: scale(1);   }}
        100% {{ opacity: 0;   transform: scale(1.15); }}
      }}
    </style>
    </head>
    <body>
    <div class="lung-wrap">
      <div class="pulse-ring">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="position:absolute;top:0;left:0;opacity:0.04;">
          <rect width="100%" height="100%" fill="rgb({r},{g},{b})"/>
        </svg>
      </div>

      <svg viewBox="0 0 420 370" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:380px;">
        <defs>
          <linearGradient id="fillG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgb({r},{g},{b})" stop-opacity="0.9"/>
            <stop offset="100%" stop-color="rgb({max(0,r-70)},{max(0,g-50)},{max(0,b-40)})" stop-opacity="1"/>
          </linearGradient>
          <linearGradient id="outG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgba(30,160,255,0.55)"/>
            <stop offset="100%" stop-color="rgba(30,160,255,0.12)"/>
          </linearGradient>
          <linearGradient id="bgLungL" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#0a1628"/>
            <stop offset="100%" stop-color="#040c18"/>
          </linearGradient>
          <linearGradient id="bgLungR" x1="1" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0a1628"/>
            <stop offset="100%" stop-color="#040c18"/>
          </linearGradient>
          <clipPath id="clipL">
            <path d="M 210 35 C 168 22,96 32,70 84 C 42 144,38 215,54 268
                     C 65 303,102 322,133 317 C 162 311,187 291,196 262
                     C 206 232,209 162,210 118 Z"/>
          </clipPath>
          <clipPath id="clipR">
            <path d="M 210 35 C 252 22,324 32,350 84 C 378 144,382 215,366 268
                     C 355 303,318 322,287 317 C 258 311,233 291,224 262
                     C 214 232,211 162,210 118 Z"/>
          </clipPath>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="softglow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="9" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="innerGlow">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
          </filter>
        </defs>

        <!-- Background lung shapes with gradient -->
        <path d="M 210 35 C 168 22,96 32,70 84 C 42 144,38 215,54 268
                 C 65 303,102 322,133 317 C 162 311,187 291,196 262
                 C 206 232,209 162,210 118 Z"
              fill="url(#bgLungL)" stroke="url(#outG)" stroke-width="1.5"/>
        <path d="M 210 35 C 252 22,324 32,350 84 C 378 144,382 215,366 268
                 C 355 303,318 322,287 317 C 258 311,233 291,224 262
                 C 214 232,211 162,210 118 Z"
              fill="url(#bgLungR)" stroke="url(#outG)" stroke-width="1.5"/>

        <!-- Inner grid lines for texture -->
        <g clip-path="url(#clipL)" opacity="0.04">
          <line x1="80" y1="40" x2="80" y2="320" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="110" y1="30" x2="110" y2="320" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="140" y1="25" x2="140" y2="315" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="170" y1="25" x2="170" y2="310" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="50" y1="100" x2="205" y2="100" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="42" y1="150" x2="205" y2="150" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="40" y1="200" x2="202" y2="200" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="46" y1="250" x2="198" y2="250" stroke="#1ea0ff" stroke-width="1"/>
        </g>
        <g clip-path="url(#clipR)" opacity="0.04">
          <line x1="230" y1="25" x2="230" y2="310" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="260" y1="25" x2="260" y2="315" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="290" y1="25" x2="290" y2="315" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="320" y1="30" x2="320" y2="320" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="215" y1="100" x2="370" y2="100" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="215" y1="150" x2="378" y2="150" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="218" y1="200" x2="380" y2="200" stroke="#1ea0ff" stroke-width="1"/>
          <line x1="222" y1="250" x2="374" y2="250" stroke="#1ea0ff" stroke-width="1"/>
        </g>

        <!-- DAMAGE FILL - Left -->
        <g clip-path="url(#clipL)">
          <rect x="35" y="{15 + fill_offset}" width="185" height="{310 - fill_offset}"
                fill="url(#fillG)" opacity="0.78"/>
          <!-- Shine highlight on fill -->
          <rect x="35" y="{15 + fill_offset}" width="30" height="{310 - fill_offset}"
                fill="rgba(255,255,255,0.06)" opacity="0.5"/>
        </g>
        <!-- DAMAGE FILL - Right -->
        <g clip-path="url(#clipR)">
          <rect x="210" y="{15 + fill_offset}" width="185" height="{310 - fill_offset}"
                fill="url(#fillG)" opacity="0.78"/>
          <rect x="340" y="{15 + fill_offset}" width="30" height="{310 - fill_offset}"
                fill="rgba(255,255,255,0.06)" opacity="0.5"/>
        </g>

        <!-- Bronchi network left -->
        <g opacity="0.22" stroke="rgba(30,160,255,1)" stroke-width="1.4" fill="none">
          <path d="M 210 85 C 183 105,160 135,150 175"/>
          <path d="M 150 175 C 138 200,124 220,118 255"/>
          <path d="M 150 175 C 157 202,160 224,155 258"/>
          <path d="M 210 85 C 196 115,192 155,196 205"/>
          <path d="M 118 255 C 108 270,100 280,95 295"/>
          <path d="M 155 258 C 150 272,148 284,145 300"/>
        </g>
        <!-- Bronchi network right -->
        <g opacity="0.22" stroke="rgba(30,160,255,1)" stroke-width="1.4" fill="none">
          <path d="M 210 85 C 237 105,260 135,270 175"/>
          <path d="M 270 175 C 282 200,296 220,302 255"/>
          <path d="M 270 175 C 263 202,260 224,265 258"/>
          <path d="M 210 85 C 224 115,228 155,224 205"/>
          <path d="M 302 255 C 312 270,320 280,325 295"/>
          <path d="M 265 258 C 270 272,272 284,275 300"/>
        </g>

        <!-- Bronchi nodes -->
        <g fill="rgba(30,160,255,0.35)">
          <circle cx="150" cy="175" r="2.5"/>
          <circle cx="118" cy="255" r="2"/>
          <circle cx="155" cy="258" r="2"/>
          <circle cx="270" cy="175" r="2.5"/>
          <circle cx="302" cy="255" r="2"/>
          <circle cx="265" cy="258" r="2"/>
        </g>

        <!-- Trachea -->
        <rect x="206" y="0" width="8" height="52" rx="4"
              fill="none" stroke="rgba(30,160,255,0.5)" stroke-width="1.5"/>
        <rect x="204" y="14" width="12" height="4" rx="2" fill="rgba(30,160,255,0.18)"/>
        <rect x="204" y="23" width="12" height="4" rx="2" fill="rgba(30,160,255,0.14)"/>
        <rect x="204" y="32" width="12" height="4" rx="2" fill="rgba(30,160,255,0.10)"/>
        <rect x="204" y="41" width="12" height="4" rx="2" fill="rgba(30,160,255,0.07)"/>

        <!-- Glow outline on top -->
        <path d="M 210 35 C 168 22,96 32,70 84 C 42 144,38 215,54 268
                 C 65 303,102 322,133 317 C 162 311,187 291,196 262
                 C 206 232,209 162,210 118 Z"
              fill="none" stroke="rgba({r},{g},{b},0.5)" stroke-width="1.5" filter="url(#glow)"/>
        <path d="M 210 35 C 252 22,324 32,350 84 C 378 144,382 215,366 268
                 C 355 303,318 322,287 317 C 258 311,233 291,224 262
                 C 214 232,211 162,210 118 Z"
              fill="none" stroke="rgba({r},{g},{b},0.5)" stroke-width="1.5" filter="url(#glow)"/>

        <!-- Percentage label -->
        <text x="210" y="352" text-anchor="middle"
              font-family="JetBrains Mono, monospace" font-size="10"
              fill="rgba(180,220,255,0.35)" letter-spacing="3">ESTIMATED DAMAGE</text>
        <text x="210" y="330" text-anchor="middle"
              font-family="Outfit, sans-serif" font-size="48" font-weight="900"
              fill="rgb({r},{g},{b})" letter-spacing="-2"
              filter="url(#softglow)">{pct_int}%</text>
      </svg>

      <p class="lung-label">Interpretive display — not diagnostic imaging</p>
    </div>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=False)


def calc_damage(snap: dict, risk: str) -> float:
    base = {"Low": 0.08, "Medium": 0.38, "High": 0.72}[risk]
    mod  = (snap.get("Smoking",0) + snap.get("Coughing Blood",0)*2 +
            snap.get("Chronic Lung",0)) / 30.0
    return min(0.97, max(0.03, base + mod * 0.15))


# ═══════════════════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════════════════
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.05, 1])
    with col:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(7,18,32,0.96);border:1px solid rgba(30,160,255,0.2);
                    border-radius:28px;padding:52px 48px;
                    box-shadow:0 40px 100px rgba(0,0,0,0.75),0 0 80px rgba(0,80,200,0.08);">
          <div class="wordmark">Lung<em>Sight</em>
            <span style="font-size:13px;font-weight:400;color:rgba(180,220,255,0.3);letter-spacing:2px;"> AI</span>
          </div>
          <p style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:4px;
                    text-transform:uppercase;color:rgba(180,220,255,0.3);margin:8px 0 38px;">
            Clinical Intelligence · v3.1
          </p>
        </div>
        """, unsafe_allow_html=True)

        uname = st.text_input("Clinician / Patient Name", placeholder="Dr. Mehta or Ramesh Patel")
        pwd   = st.text_input("Access Code", type="password", placeholder="••••••••")
        err   = st.empty()

        if st.button("ACCESS SYSTEM  ⟶", use_container_width=True):
            if not uname.strip() or not pwd.strip():
                err.error("Both fields are required.")
            else:
                with st.spinner("Authenticating…"):
                    time.sleep(0.5)
                st.session_state.auth     = True
                st.session_state.username = uname.strip()
                st.rerun()

        st.markdown("""
        <p style="font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:2px;
                  color:rgba(180,220,255,0.18);text-align:center;margin-top:28px;text-transform:uppercase;">
          Academic &amp; Clinical Research Use Only
        </p>
        """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════
# NAV BAR
# ═══════════════════════════════════════════════════
c1, _, c3 = st.columns([1, 3, 1])
with c1:
    st.markdown('<div class="wordmark" style="padding-top:6px;">Lung<em>Sight</em>'
                '<span style="font-size:11px;font-weight:400;color:rgba(180,220,255,0.28);letter-spacing:2px;"> AI</span></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f"<p style='text-align:right;color:var(--muted);font-size:11px;"
                f"font-family:JetBrains Mono,monospace;padding-top:10px;'>⬡ {st.session_state.username}</p>",
                unsafe_allow_html=True)
    if st.button("Logout"):
        for k in ["auth","risk","pred_css","snap","show_results","step"]:
            st.session_state[k] = False if k=="auth" else None if k in ["risk","pred_css","snap"] else False if k=="show_results" else 0
        st.rerun()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════
tab_assess, tab_history = st.tabs(["🫁  Risk Assessment", "📋  Patient History"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — ASSESSMENT
# ══════════════════════════════════════════════════════════════
with tab_assess:
    STEPS = [
        ("Patient Profile",        "Basic demographics of the patient."),
        ("Lifestyle Factors",      "Smoking, alcohol, diet, and daily habits."),
        ("Environmental Exposure", "Pollution, workplace, and passive exposure."),
        ("Medical History",        "Genetic risk and prior lung conditions."),
        ("Clinical Symptoms",      "Current symptoms reported by the patient."),
    ]
    step = st.session_state.step

    # Step progress dots
    prog_html = '<div class="sp-bar">'
    for i, (ttl, _) in enumerate(STEPS):
        state = "done" if i < step else "active" if i == step else "future"
        icon  = "✓" if i < step else str(i+1)
        prog_html += f'<div class="sp-dot {state}" title="{ttl}">{icon}</div>'
        if i < len(STEPS)-1:
            prog_html += f'<div class="sp-line {"done" if i < step else ""}"></div>'
    prog_html += '</div>'
    st.markdown(prog_html, unsafe_allow_html=True)

    # ── STEP 0 ────────────────────────────────────
    s0 = "active" if step==0 else "done" if step>0 else "locked"
    st.markdown(f"""
    <div class="step-card {s0}">
      <div class="step-header">
        <div class="step-num {s0}">{"✓" if step>0 else "1"}</div>
        <div class="step-title">Patient Profile</div>
      </div>
      <p class="step-desc">Basic demographics — age and gender.</p>
    </div>""", unsafe_allow_html=True)

    if step == 0:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.age = st.slider("Age (years)", 1, 100, st.session_state.age)
        with col2:
            st.session_state.gender = st.selectbox("Gender", ["Male","Female","Others"],
                index=["Male","Female","Others"].index(st.session_state.gender))
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _, btn, _ = st.columns([0.55, 0.3, 1])
        with btn:
            if st.button("NEXT → LIFESTYLE", use_container_width=True):
                st.session_state.step = 1; st.rerun()

    # ── STEP 1 ────────────────────────────────────
    s1 = "active" if step==1 else "done" if step>1 else "locked"
    st.markdown(f"""
    <div class="step-card {s1}">
      <div class="step-header">
        <div class="step-num {s1}">{"✓" if step>1 else "2"}</div>
        <div class="step-title">Lifestyle Factors</div>
      </div>
      <p class="step-desc">Daily habits that directly impact lung health.</p>
    </div>""", unsafe_allow_html=True)

    if step == 1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.smoking       = st.slider("Smoking",       0,10,st.session_state.smoking)
            st.session_state.alcohol_use   = st.slider("Alcohol Use",   0,10,st.session_state.alcohol_use)
        with c2:
            st.session_state.balanced_diet = st.slider("Balanced Diet", 0,10,st.session_state.balanced_diet)
            st.session_state.obesity       = st.slider("Obesity",       0,10,st.session_state.obesity)
        with c3:
            st.session_state.snoring       = st.slider("Snoring",       0,10,st.session_state.snoring)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([0.22, 0.28, 0.5])
        with b1:
            if st.button("← BACK", use_container_width=True):
                st.session_state.step = 0; st.rerun()
        with b2:
            if st.button("NEXT → ENVIRONMENT", use_container_width=True):
                st.session_state.step = 2; st.rerun()

    # ── STEP 2 ────────────────────────────────────
    s2 = "active" if step==2 else "done" if step>2 else "locked"
    st.markdown(f"""
    <div class="step-card {s2}">
      <div class="step-header">
        <div class="step-num {s2}">{"✓" if step>2 else "3"}</div>
        <div class="step-title">Environmental Exposure</div>
      </div>
      <p class="step-desc">Air quality, occupational, and second-hand exposure.</p>
    </div>""", unsafe_allow_html=True)

    if step == 2:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.air_pollution  = st.slider("Air Pollution",        0,10,st.session_state.air_pollution)
            st.session_state.passive_smoker = st.slider("Passive Smoker",       0,10,st.session_state.passive_smoker)
        with c2:
            st.session_state.dust_allergy   = st.slider("Dust Allergy",         0,10,st.session_state.dust_allergy)
            st.session_state.occupational   = st.slider("Occupational Hazards", 0,10,st.session_state.occupational)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([0.22, 0.3, 0.48])
        with b1:
            if st.button("← BACK ", use_container_width=True):
                st.session_state.step = 1; st.rerun()
        with b2:
            if st.button("NEXT → MEDICAL HISTORY", use_container_width=True):
                st.session_state.step = 3; st.rerun()

    # ── STEP 3 ────────────────────────────────────
    s3 = "active" if step==3 else "done" if step>3 else "locked"
    st.markdown(f"""
    <div class="step-card {s3}">
      <div class="step-header">
        <div class="step-num {s3}">{"✓" if step>3 else "4"}</div>
        <div class="step-title">Medical History</div>
      </div>
      <p class="step-desc">Genetic predisposition and prior lung conditions.</p>
    </div>""", unsafe_allow_html=True)

    if step == 3:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.genetic_risk  = st.slider("Genetic Risk",         0,10,st.session_state.genetic_risk)
        with c2:
            st.session_state.chronic_lung  = st.slider("Chronic Lung Disease", 0,10,st.session_state.chronic_lung)
        with c3:
            st.session_state.frequent_cold = st.slider("Frequent Cold",        0,10,st.session_state.frequent_cold)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([0.22, 0.26, 0.52])
        with b1:
            if st.button("← BACK  ", use_container_width=True):
                st.session_state.step = 2; st.rerun()
        with b2:
            if st.button("NEXT → SYMPTOMS", use_container_width=True):
                st.session_state.step = 4; st.rerun()

    # ── STEP 4 ────────────────────────────────────
    s4 = "active" if step==4 else "done" if step>4 else "locked"
    st.markdown(f"""
    <div class="step-card {s4}">
      <div class="step-header">
        <div class="step-num {s4}">{"✓" if step>4 else "5"}</div>
        <div class="step-title">Clinical Symptoms</div>
      </div>
      <p class="step-desc">Current symptoms observed or reported by the patient.</p>
    </div>""", unsafe_allow_html=True)

    if step == 4:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.chest_pain     = st.slider("Chest Pain",             0,10,st.session_state.chest_pain)
            st.session_state.coughing_blood = st.slider("Coughing Blood",         0,10,st.session_state.coughing_blood)
            st.session_state.fatigue        = st.slider("Fatigue",                0,10,st.session_state.fatigue)
        with c2:
            st.session_state.weight_loss    = st.slider("Weight Loss",            0,10,st.session_state.weight_loss)
            st.session_state.breath         = st.slider("Shortness of Breath",    0,10,st.session_state.breath)
            st.session_state.wheezing       = st.slider("Wheezing",               0,10,st.session_state.wheezing)
        with c3:
            st.session_state.swallow        = st.slider("Swallowing Difficulty",  0,10,st.session_state.swallow)
            st.session_state.clubbing       = st.slider("Clubbing of Fingernails",0,10,st.session_state.clubbing)
            st.session_state.dry_cough      = st.slider("Dry Cough",              0,10,st.session_state.dry_cough)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        b1, b2, b3, _ = st.columns([0.18, 0.20, 0.32, 0.30])
        with b1:
            if st.button("← BACK   ", use_container_width=True):
                st.session_state.step = 3; st.rerun()
        with b3:
            run = st.button("⚙  RUN ML ANALYSIS", use_container_width=True)

        if run:
            s = st.session_state
            gender_encoded = {"Male":0,"Female":1,"Others":2}[s.gender]
            features = np.array([[
                s.age, gender_encoded, s.air_pollution, s.alcohol_use,
                s.genetic_risk, s.chronic_lung, s.balanced_diet, s.obesity,
                s.smoking, s.passive_smoker, s.chest_pain, s.coughing_blood,
                s.fatigue, s.weight_loss, s.breath, s.wheezing,
                s.swallow, s.clubbing, s.frequent_cold, s.dry_cough,
                s.dust_allergy, s.occupational, s.snoring
            ]])

            with st.spinner("Running classifier…"):
                time.sleep(0.5)
                pred = model.predict(features)[0]

            labels = {0:("Low","low"), 1:("Medium","medium"), 2:("High","high")}
            risk_label, risk_css = labels.get(int(pred), ("Medium","medium"))

            snap = {
                "Smoking": s.smoking, "Air Pollution": s.air_pollution,
                "Genetic Risk": s.genetic_risk, "Chest Pain": s.chest_pain,
                "Fatigue": s.fatigue, "Wheezing": s.wheezing,
                "Dry Cough": s.dry_cough, "Alcohol Use": s.alcohol_use,
                "Balanced Diet": s.balanced_diet, "Obesity": s.obesity,
                "Passive Smoker": s.passive_smoker, "Chronic Lung": s.chronic_lung,
                "Frequent Cold": s.frequent_cold, "Coughing Blood": s.coughing_blood,
                "Weight Loss": s.weight_loss, "Shortness of Breath": s.breath,
                "Swallowing Difficulty": s.swallow, "Clubbing": s.clubbing,
                "Snoring": s.snoring, "Dust Allergy": s.dust_allergy,
                "Occupational": s.occupational,
            }

            st.session_state.risk         = risk_label
            st.session_state.pred_css     = risk_css
            st.session_state.snap         = snap
            st.session_state.show_results = True
            st.session_state.step         = 5

            add_record({
                "id":       datetime.now().strftime("%Y%m%d%H%M%S"),
                "datetime": datetime.now().strftime("%d %b %Y, %H:%M"),
                "patient":  s.username,
                "age":      s.age,
                "gender":   s.gender,
                "risk":     risk_label,
                "css":      risk_css,
                "smoking":  s.smoking,
                "air":      s.air_pollution,
                "genetic":  s.genetic_risk,
            })
            st.rerun()

    # ════════════════════════════════════════
    # RESULTS SECTION
    # ════════════════════════════════════════
    if st.session_state.show_results and st.session_state.risk:
        risk     = st.session_state.risk
        pred_css = st.session_state.pred_css
        snap     = st.session_state.snap

        color = {"low":"#00e5a0","medium":"#ff9900","high":"#ff3355"}[pred_css]
        advice = {
            "Low":    ("No Immediate Concern",      "Maintain healthy lifestyle. Annual check-up recommended."),
            "Medium": ("Specialist Consultation",   "Moderate risk. Imaging and Pulmonologist consult advised."),
            "High":   ("Immediate Evaluation Needed","High risk. Urgent referral to Lung Cancer Specialist."),
        }
        ttl, desc = advice[risk]

        st.markdown("---")
        st.markdown('<span class="slabel">Analysis Complete · ML Prediction</span>', unsafe_allow_html=True)

        # Risk banner
        st.markdown(f"""
        <div class="risk-banner risk-{pred_css}">
          <div class="risk-eyebrow">LungSight AI v3.1 · Trained Classifier Result</div>
          <div class="risk-big">{risk} Risk</div>
          <div class="risk-sub">{desc}</div>
          <div class="risk-action">{ttl}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metric strip
        damage    = calc_damage(snap, risk)
        top_fac   = max(snap, key=snap.get)
        st.markdown(f"""
        <div class="metric-strip">
          <div class="metric-box">
            <div class="mlabel">Lung Damage Est.</div>
            <div class="mvalue" style="color:{color};">{int(damage*100)}%</div>
            <div class="msub">Based on risk + key markers</div>
          </div>
          <div class="metric-box">
            <div class="mlabel">Model Accuracy</div>
            <div class="mvalue">{int(MODEL_ACCURACY*100)}%</div>
            <div class="msub">Classifier performance</div>
          </div>
          <div class="metric-box">
            <div class="mlabel">Top Risk Factor</div>
            <div class="mvalue" style="font-size:16px;padding-top:5px;letter-spacing:-0.5px;">{top_fac}</div>
            <div class="msub">Score: {snap[top_fac]}/10</div>
          </div>
          <div class="metric-box">
            <div class="mlabel">Patient</div>
            <div class="mvalue" style="font-size:16px;padding-top:5px;letter-spacing:-0.5px;">{st.session_state.username[:12]}</div>
            <div class="msub">{st.session_state.age} yrs · {st.session_state.gender}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── LUNG VIS + BAR CHART ──
        col_lung, col_bar = st.columns([1, 1.65])

        with col_lung:
            st.markdown('<span class="slabel" style="margin-bottom:10px;">Lung Damage Visualisation</span>',
                        unsafe_allow_html=True)
            # Use components.html so SVG is rendered faithfully
            lung_component(damage, height=480)

        with col_bar:
            st.markdown('<span class="slabel" style="margin-bottom:10px;">Factor Severity Breakdown</span>',
                        unsafe_allow_html=True)
            bar_keys = ["Smoking","Air Pollution","Genetic Risk","Chest Pain",
                        "Fatigue","Wheezing","Dry Cough","Coughing Blood",
                        "Shortness of Breath","Weight Loss"]
            bar_vals = [snap[k] for k in bar_keys]
            bar_cols = [color if v>=7 else "#0088ff" if v>=4 else "rgba(30,160,255,0.2)" for v in bar_vals]

            fig = go.Figure(go.Bar(
                x=bar_vals, y=bar_keys, orientation='h',
                marker=dict(color=bar_cols, line=dict(width=0)),
                text=[f"{v}/10" for v in bar_vals],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10, color="rgba(180,220,255,0.5)"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(range=[0,13], gridcolor="rgba(30,160,255,0.06)",
                           color="rgba(180,220,255,0.3)", tickfont=dict(size=9)),
                yaxis=dict(color="rgba(180,220,255,0.55)", tickfont=dict(size=11,family="Outfit")),
                height=370, bargap=0.38,
                font=dict(family="JetBrains Mono"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        # ── RADAR CHART ──
        st.markdown('<span class="slabel" style="margin-top:4px;">Risk Distribution — Radar Analysis</span>',
                    unsafe_allow_html=True)
        radar_keys = ["Smoking", "Genetic Risk", "Air Pollution", "Chest Pain", "Fatigue", "Passive Smoker"]
        radar_vals = [snap[k] for k in radar_keys]
        radar_vals_closed = radar_vals + [radar_vals[0]]
        radar_keys_closed = radar_keys + [radar_keys[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals_closed,
            theta=radar_keys_closed,
            fill='toself',
            fillcolor=f"rgba({','.join(str(int(c*255)) for c in [0,0.74,0.84])},0.12)",
            line=dict(color="#00bcd4", width=2),
            marker=dict(size=6, color="#00bcd4"),
            name="Risk Profile"
        ))
        fig_radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(7,18,32,0.6)",
                radialaxis=dict(
                    visible=True, range=[0,10],
                    gridcolor="rgba(30,160,255,0.08)",
                    color="rgba(180,220,255,0.3)",
                    tickfont=dict(size=9)
                ),
                angularaxis=dict(
                    gridcolor="rgba(30,160,255,0.08)",
                    color="rgba(180,220,255,0.45)",
                    tickfont=dict(size=11, family="Outfit")
                )
            ),
            showlegend=False,
            margin=dict(l=60, r=60, t=40, b=40),
            height=340,
            font=dict(family="JetBrains Mono", color="rgba(180,220,255,0.5)")
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar":False})

        # ── FULL INTENSITY HEATMAP ──
        st.markdown('<span class="slabel" style="margin-top:4px;">All Factors — Intensity Heatmap</span>',
                    unsafe_allow_html=True)
        all_k = list(snap.keys())
        all_v = [snap[k] for k in all_k]
        fig2 = go.Figure(go.Bar(
            x=all_k, y=all_v,
            marker=dict(
                color=all_v,
                colorscale=[[0,"#071220"],[0.3,"#0055cc"],[0.65,"#ff9900"],[1,"#ff3355"]],
                showscale=True,
                colorbar=dict(thickness=8, len=0.5,
                              tickfont=dict(size=8, color="rgba(180,220,255,0.3)"),
                              outlinecolor="rgba(0,0,0,0)")
            ),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=8, b=65),
            yaxis=dict(range=[0,12], gridcolor="rgba(30,160,255,0.05)",
                       color="rgba(180,220,255,0.3)", tickfont=dict(size=9)),
            xaxis=dict(color="rgba(180,220,255,0.4)", tickfont=dict(size=9,family="Outfit"), tickangle=-38),
            height=255, font=dict(family="JetBrains Mono"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

        # ── APPOINTMENT ──
        st.markdown("---")
        st.markdown('<span class="slabel">Referral &amp; Scheduling</span>', unsafe_allow_html=True)
        st.markdown("<h2 style='font-size:24px;font-weight:800;margin:4px 0 18px;'>Book Appointment</h2>",
                    unsafe_allow_html=True)

        doc = {"Low":"General Physician","Medium":"Pulmonologist","High":"Lung Cancer Specialist"}[risk]
        urg_label, urg_color = {"Low":("Routine","#00e5a0"),"Medium":("Priority","#ff9900"),"High":("URGENT","#ff3355")}[risk]

        st.markdown(f"""
        <div class="appt-block">
          <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div>
              <span class="slabel">Recommended Specialist</span>
              <p style="font-size:20px;font-weight:800;color:#fff;margin:4px 0 0;">{doc}</p>
            </div>
            <div style="margin-left:auto;">
              <span style="background:rgba(255,255,255,0.04);border:1.5px solid {urg_color};
                           border-radius:8px;padding:7px 16px;font-family:'JetBrains Mono',monospace;
                           font-size:10px;letter-spacing:2px;color:{urg_color};">{urg_label}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        with ac1:
            phone = st.text_input("Patient Phone", placeholder="+91 XXXXX XXXXX")
        with ac2:
            appt_date = st.date_input("Appointment Date", min_value=date.today())
        notes = st.text_area("Clinical Notes (optional)", placeholder="Additional observations…", height=76)

        _, conf_col, _ = st.columns([0.55, 0.38, 0.55])
        with conf_col:
            confirm = st.button("✓  CONFIRM APPOINTMENT", use_container_width=True)

        if confirm:
            if not phone.strip():
                st.error("Phone number is required.")
            else:
                st.success("✓  Appointment Confirmed")
                note_row = (f"<tr><td style='color:var(--muted);padding:9px 0;font-family:JetBrains Mono,monospace;"
                            f"font-size:9px;letter-spacing:1px;text-transform:uppercase;'>Notes</td>"
                            f"<td style='color:var(--muted);font-size:12px;'>{notes}</td></tr>") if notes.strip() else ""
                st.markdown(f"""
                <div class="appt-block" style="margin-top:14px;">
                  <span class="slabel">Appointment Summary</span>
                  <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:10px;">
                    <tr><td style="color:var(--muted);padding:9px 0;width:160px;font-family:JetBrains Mono,monospace;
                                   font-size:9px;letter-spacing:1px;text-transform:uppercase;">Patient</td>
                        <td style="color:var(--text);font-weight:600;">{st.session_state.username}</td></tr>
                    <tr><td style="color:var(--muted);padding:9px 0;font-family:JetBrains Mono,monospace;
                                   font-size:9px;letter-spacing:1px;text-transform:uppercase;">Specialist</td>
                        <td style="color:var(--text);font-weight:600;">{doc}</td></tr>
                    <tr><td style="color:var(--muted);padding:9px 0;font-family:JetBrains Mono,monospace;
                                   font-size:9px;letter-spacing:1px;text-transform:uppercase;">Date</td>
                        <td style="color:var(--text);font-weight:600;">{appt_date.strftime("%d %B %Y")}</td></tr>
                    <tr><td style="color:var(--muted);padding:9px 0;font-family:JetBrains Mono,monospace;
                                   font-size:9px;letter-spacing:1px;text-transform:uppercase;">Contact</td>
                        <td style="color:var(--accent);font-family:JetBrains Mono,monospace;">{phone}</td></tr>
                    <tr><td style="color:var(--muted);padding:9px 0;font-family:JetBrains Mono,monospace;
                                   font-size:9px;letter-spacing:1px;text-transform:uppercase;">Risk Level</td>
                        <td style="color:{color};font-weight:800;">{risk}</td></tr>
                    {note_row}
                  </table>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        _, rs_col, _ = st.columns([0.62, 0.32, 0.62])
        with rs_col:
            if st.button("↺  NEW ASSESSMENT", use_container_width=True):
                st.session_state.step         = 0
                st.session_state.risk         = None
                st.session_state.snap         = None
                st.session_state.show_results = False
                st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2 — PATIENT HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    records = load_history()

    st.markdown('<span class="slabel">Assessment Records</span>', unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:24px;font-weight:800;margin:4px 0 22px;'>Patient History</h2>",
                unsafe_allow_html=True)

    if not records:
        st.markdown("""
        <div style="background:var(--s1);border:1px solid var(--border);border-radius:20px;
                    padding:60px;text-align:center;">
          <p style="font-size:38px;margin-bottom:14px;">📋</p>
          <p style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;
                    text-transform:uppercase;color:var(--muted);">No records yet</p>
          <p style="font-size:12px;color:var(--muted);margin-top:8px;">Complete an assessment to see history here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        n_high = sum(1 for r in records if r["risk"]=="High")
        n_med  = sum(1 for r in records if r["risk"]=="Medium")
        n_low  = sum(1 for r in records if r["risk"]=="Low")

        st.markdown(f"""
        <div class="metric-strip">
          <div class="metric-box"><div class="mlabel">Total Records</div>
            <div class="mvalue">{len(records)}</div></div>
          <div class="metric-box"><div class="mlabel">High Risk</div>
            <div class="mvalue" style="color:#ff3355;">{n_high}</div></div>
          <div class="metric-box"><div class="mlabel">Medium Risk</div>
            <div class="mvalue" style="color:#ff9900;">{n_med}</div></div>
          <div class="metric-box"><div class="mlabel">Low Risk</div>
            <div class="mvalue" style="color:#00e5a0;">{n_low}</div></div>
        </div>
        """, unsafe_allow_html=True)

        if len(records) > 1:
            st.markdown('<span class="slabel" style="margin:18px 0 10px;display:block;">Risk Trend Over Time</span>',
                        unsafe_allow_html=True)
            df = pd.DataFrame(records[::-1])
            risk_map = {"Low":0,"Medium":1,"High":2}
            df["rn"] = df["risk"].map(risk_map)

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=df["datetime"], y=df["rn"],
                mode="lines+markers",
                line=dict(color="#1ea0ff", width=2),
                marker=dict(
                    color=[{"Low":"#00e5a0","Medium":"#ff9900","High":"#ff3355"}[r] for r in df["risk"]],
                    size=11, line=dict(color="#03080f", width=2)
                ),
                hovertemplate="<b>%{text}</b><br>%{x}<extra></extra>",
                text=df["risk"],
            ))
            fig_t.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis=dict(color="rgba(180,220,255,0.4)", tickfont=dict(size=9,family="Outfit"), showgrid=False),
                yaxis=dict(tickvals=[0,1,2], ticktext=["Low","Medium","High"],
                           color="rgba(180,220,255,0.4)", tickfont=dict(size=10),
                           gridcolor="rgba(30,160,255,0.06)"),
                height=200,
            )
            st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar":False})

        st.markdown('<span class="slabel" style="margin-top:20px;display:block;">All Records</span>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="hist-head">
          <span>Date</span><span>Patient</span><span>Age</span><span>Gender</span><span>Risk</span>
        </div>""", unsafe_allow_html=True)

        for r in records:
            st.markdown(f"""
            <div class="hist-row">
              <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);">{r['datetime']}</span>
              <span style="font-weight:600;">{r['patient']}</span>
              <span style="color:var(--muted);">{r['age']}</span>
              <span style="color:var(--muted);">{r['gender']}</span>
              <span><span class="badge badge-{r['css']}">{r['risk']}</span></span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _, del_col, _ = st.columns([0.7, 0.3, 0.7])
        with del_col:
            if st.button("🗑  Clear History", use_container_width=True):
                save_history([])
                st.rerun()

# ── Footer ──
st.markdown("""
<hr/>
<p style="font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:2px;
          text-transform:uppercase;color:rgba(180,220,255,0.14);text-align:center;padding-bottom:20px;">
  LungSight AI · Clinical Intelligence · v3.1 · Academic Research · Not a substitute for professional medical advice
</p>
""", unsafe_allow_html=True)