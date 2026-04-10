"""
AQI Forecasting Dashboard — TFT v3  ·  Advanced Research Console
=================================================================
Model  : TFTv3  (autoregressive decoder + teacher forcing + GRN)
Scaler : RobustScaler  (loaded from scalers_v3.pkl)
Output : Q10 / Q50 / Q90  ×  24 future hours
Charts : Plotly interactive — Dark Console OR White Research Paper mode
Theme  : Toggle in sidebar switches EVERYTHING simultaneously
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI · TFT v3 Research Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME STATE  — must happen before any CSS injection
# ─────────────────────────────────────────────────────────────
if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

# ─────────────────────────────────────────────────────────────
# THEME TOKENS
# ─────────────────────────────────────────────────────────────
DARK = dict(
    bg          = "#050810",
    surface     = "#0c1220",
    surface2    = "#101828",
    border      = "#1a2840",
    border2     = "#243650",
    accent      = "#00d4ff",
    accent2     = "#ff6b35",
    accent3     = "#00ff9d",
    warn        = "#ffd166",
    danger      = "#ff3b5c",
    text        = "#c8d8f0",
    text2       = "#8aa0c0",
    muted       = "#4a6080",
    info_label  = "#4a6080",
    info_value  = "#00d4ff",
    # plotly
    paper_bg    = "#050810",
    plot_bg     = "#0c1220",
    grid        = "#1a2840",
    font_color  = "#8aa0c0",
    legend_bg   = "#0c1220",
    legend_border = "#1a2840",
    hover_bg    = "#101828",
    hover_border= "#00d4ff",
    hover_font  = "#c8d8f0",
    title_color = "#00d4ff",
    bar_edge    = "#050810",
    # line colors for traces
    hist_line   = "#4a6080",
    raw_line    = "#00d4ff",
    roll6_line  = "#00ff9d",
    roll24_line = "#ff6b35",
    q50_line    = "#00d4ff",
    q10_line    = "#00ff9d",
    q90_line    = "#ff6b35",
    spread_line = "#ffd166",
    spread_fill = "rgba(255,209,102,0.15)",
    ribbon_fill = "rgba(0,212,255,0.10)",
    now_line    = "#ffd166",
    now_annot   = "#ffd166",
    trend_line  = "#ffd166",
    radar_fill1 = "rgba(0,212,255,0.12)",
    radar_fill2 = "rgba(255,209,102,0.08)",
    radar_grid  = "#1a2840",
    radar_bg    = "#0c1220",
    radar_ang   = "#8aa0c0",
    pie_edge    = "#050810",
    scatter_edge= "#050810",
    # section header
    sec_line    = "#00d4ff",
    sec_text    = "#00d4ff",
    sec_num     = "#4a6080",
    sec_border  = "#1a2840",
    # cards
    metric_bg   = "linear-gradient(135deg,#0c1220 0%,#101828 100%)",
    metric_border = "#1a2840",
    metric_hover_border = "#00d4ff",
    metric_label = "#4a6080",
    metric_value = "#00d4ff",
    # ticker
    ticker_bg   = "linear-gradient(90deg,transparent,rgba(0,212,255,0.05),rgba(0,212,255,0.08),rgba(0,212,255,0.05),transparent)",
    ticker_border = "rgba(0,212,255,0.15)",
    ticker_color= "#4a6080",
    # tip card
    tip_bg      = "#101828",
    tip_border  = "#1a2840",
    tip_text    = "#8aa0c0",
    tip_left    = "#00d4ff",
    # scanline
    scanline    = "rgba(0,212,255,0.012)",
    rangeslider_bg = "#0c1220",
    rangeselector_bg = "#0c1220",
    rangeselector_active = "#00d4ff",
    rangeselector_font = "#8aa0c0",
)

LIGHT = dict(
    bg          = "#f7f8fc",
    surface     = "#ffffff",
    surface2    = "#f0f2f8",
    border      = "#d0d8e8",
    border2     = "#b0bcd0",
    accent      = "#1565c0",
    accent2     = "#c0392b",
    accent3     = "#2e7d32",
    warn        = "#f9a825",
    danger      = "#c62828",
    text        = "#0d1117",
    text2       = "#1e2d3d",
    muted       = "#374151",
    info_label  = "#1e3a5f",
    info_value  = "#0d47a1",
    # plotly
    paper_bg    = "#ffffff",
    plot_bg     = "#f7f8fc",
    grid        = "#dde3ed",
    font_color  = "#0d1117",
    legend_bg   = "#ffffff",
    legend_border = "#d0d8e8",
    hover_bg    = "#ffffff",
    hover_border= "#1565c0",
    hover_font  = "#0d1117",
    title_color = "#1565c0",
    bar_edge    = "#ffffff",
    # line colors — research paper quality (saturated, accessible)
    hist_line   = "#78909c",
    raw_line    = "#1565c0",
    roll6_line  = "#2e7d32",
    roll24_line = "#c0392b",
    q50_line    = "#1565c0",
    q10_line    = "#2e7d32",
    q90_line    = "#c0392b",
    spread_line = "#f9a825",
    spread_fill = "rgba(249,168,37,0.12)",
    ribbon_fill = "rgba(21,101,192,0.10)",
    now_line    = "#f9a825",
    now_annot   = "#e65100",
    trend_line  = "#6a1b9a",
    radar_fill1 = "rgba(21,101,192,0.10)",
    radar_fill2 = "rgba(249,168,37,0.08)",
    radar_grid  = "#d0d8e8",
    radar_bg    = "#f7f8fc",
    radar_ang   = "#0d1117",
    pie_edge    = "#ffffff",
    scatter_edge= "#ffffff",
    # section header
    sec_line    = "#1565c0",
    sec_text    = "#1565c0",
    sec_num     = "#1e3a5f",
    sec_border  = "#d0d8e8",
    # cards
    metric_bg   = "linear-gradient(135deg,#ffffff 0%,#f0f2f8 100%)",
    metric_border = "#d0d8e8",
    metric_hover_border = "#1565c0",
    metric_label = "#1e3a5f",
    metric_value = "#1565c0",
    # ticker
    ticker_bg   = "linear-gradient(90deg,transparent,rgba(21,101,192,0.04),rgba(21,101,192,0.07),rgba(21,101,192,0.04),transparent)",
    ticker_border = "rgba(21,101,192,0.20)",
    ticker_color= "#1e3a5f",
    # tip card
    tip_bg      = "#f0f2f8",
    tip_border  = "#d0d8e8",
    tip_text    = "#0d1117",
    tip_left    = "#1565c0",
    # scanline (none in light mode)
    scanline    = "rgba(0,0,0,0)",
    rangeslider_bg = "#f0f2f8",
    rangeselector_bg = "#f0f2f8",
    rangeselector_active = "#1565c0",
    rangeselector_font = "#0d1117",
)

T = LIGHT if st.session_state.light_mode else DARK

# AQI band colors stay vivid in both themes (they encode meaning)
AQI_BANDS = [
    (50,  "GOOD",        "#27ae60", "Safe for all"),
    (100, "MODERATE",    "#f39c12", "Sensitive groups"),
    (150, "USG",         "#e67e22", "Unhealthy for SG"),
    (200, "UNHEALTHY",   "#e74c3c", "All affected"),
    (300, "VERY UNHLT.", "#8e44ad", "Serious effects"),
    (999, "HAZARDOUS",   "#c0392b", "Emergency"),
]

POLL_COLORS_DARK  = ["#00d4ff","#ffd166","#ff6b35","#00ff9d","#c77dff","#f15bb5","#00bbf9"]
POLL_COLORS_LIGHT = ["#1565c0","#e65100","#2e7d32","#6a1b9a","#ad1457","#00838f","#4527a0"]
POLL_COLORS = POLL_COLORS_LIGHT if st.session_state.light_mode else POLL_COLORS_DARK

# ─────────────────────────────────────────────────────────────
# CSS — dynamic, driven by T tokens
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Crimson+Pro:ital,wght@0,300;1,300&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

html,body,[class*="css"] {{
    font-family:'Share Tech Mono',monospace;
    background:{T['bg']}; color:{T['text']};
}}
.stApp {{ background:{T['bg']}; }}

/* scanline — hidden in light mode */
.stApp::before {{
    content:''; position:fixed; top:0;left:0;right:0;bottom:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,
        {T['scanline']} 2px,{T['scanline']} 4px);
    pointer-events:none; z-index:9999;
}}

/* sidebar */
section[data-testid="stSidebar"] {{
    background:{T['surface']} !important;
    border-right:1px solid {T['border']} !important;
}}
section[data-testid="stSidebar"] * {{ color:{T['text']} !important; }}
section[data-testid="stSidebar"] h2 {{
    font-family:'Orbitron',monospace !important;
    font-size:0.6rem !important; letter-spacing:4px;
    color:{T['accent']} !important; text-transform:uppercase;
}}

/* headings */
h1,h2,h3 {{ font-family:'Orbitron',monospace !important; }}
h1 {{ color:{T['accent']} !important; letter-spacing:2px;
      {'text-shadow:0 0 30px rgba(0,212,255,0.4);' if not st.session_state.light_mode else ''} }}
h2 {{ color:{T['text']} !important; font-size:0.85rem !important; letter-spacing:3px; text-transform:uppercase; }}
h3 {{ color:{T['accent']} !important; font-size:0.72rem !important; letter-spacing:2px; }}

/* metric cards */
[data-testid="metric-container"] {{
    background:{T['metric_bg']};
    border:1px solid {T['metric_border']};
    border-radius:4px; padding:1rem 1.2rem !important;
    position:relative; overflow:hidden; transition:all 0.3s ease;
}}
[data-testid="metric-container"]:hover {{
    border-color:{T['metric_hover_border']};
    box-shadow:0 4px 20px rgba(0,0,0,0.12);
    transform:translateY(-2px);
}}
[data-testid="metric-container"]::before {{
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,transparent,{T['accent']},transparent);
    animation:scanBar 4s linear infinite;
}}
@keyframes scanBar {{ 0%,100%{{opacity:0.4;}} 50%{{opacity:1;}} }}

[data-testid="stMetricLabel"] {{
    font-family:'Orbitron',monospace !important;
    font-size:0.55rem !important; letter-spacing:3px;
    text-transform:uppercase; color:{T['metric_label']} !important;
}}
[data-testid="stMetricValue"] {{
    font-family:'Orbitron',monospace !important;
    font-size:1.8rem !important; font-weight:700 !important;
    color:{T['metric_value']} !important;
}}
[data-testid="stMetricDelta"] {{
    font-family:'Share Tech Mono',monospace !important; font-size:0.7rem !important;
}}

/* tabs */
.stTabs [role="tablist"] {{
    gap:2px; background:{T['surface2']}; padding:4px;
    border-radius:4px; border:1px solid {T['border']};
}}
.stTabs [role="tab"] {{
    font-family:'Orbitron',monospace !important; font-size:0.58rem !important;
    letter-spacing:2px; text-transform:uppercase;
    background:transparent !important; border:1px solid transparent !important;
    border-radius:3px !important; color:{T['muted']} !important;
    padding:0.5rem 0.9rem !important; transition:all 0.2s;
}}
.stTabs [role="tab"]:hover {{ color:{T['accent']} !important; border-color:{T['border']} !important; }}
.stTabs [aria-selected="true"] {{
    background:{T['surface']} !important; color:{T['accent']} !important;
    border-color:{T['accent']} !important;
    box-shadow:0 0 8px rgba(0,0,0,0.08) !important;
}}

/* button */
.stButton>button {{
    font-family:'Orbitron',monospace !important;
    background:transparent !important; color:{T['accent']} !important;
    border:1px solid {T['accent']} !important; border-radius:3px !important;
    font-size:0.6rem !important; font-weight:700 !important;
    letter-spacing:3px !important; text-transform:uppercase !important;
    width:100% !important; padding:0.85rem !important; transition:all 0.3s !important;
}}
.stButton>button:hover {{
    background:{T['accent']}1a !important;
    box-shadow:0 4px 16px rgba(0,0,0,0.15) !important;
    transform:translateY(-1px) !important;
}}

/* selectbox */
.stSelectbox>div>div {{
    background:{T['surface']} !important; border:1px solid {T['border']} !important;
    border-radius:3px !important; color:{T['text']} !important;
    font-family:'Share Tech Mono',monospace !important;
}}
.stSelectbox>div>div:hover {{ border-color:{T['accent']} !important; }}

/* number input */
.stNumberInput>div>div>input {{
    background:{T['surface']} !important; border:1px solid {T['border']} !important;
    color:{T['text']} !important; font-family:'Share Tech Mono',monospace !important;
    border-radius:3px !important;
}}

/* checkbox / toggle */
.stCheckbox>label {{ font-family:'Share Tech Mono',monospace !important; font-size:0.78rem !important; }}

/* dataframe */
.stDataFrame {{ border:1px solid {T['border']} !important; border-radius:4px !important; }}

/* alerts */
.stSuccess {{ background:rgba(39,174,96,0.08)!important; border:1px solid rgba(39,174,96,0.35)!important; border-radius:3px!important; }}
.stError   {{ background:rgba(192,57,43,0.07)!important; border:1px solid rgba(192,57,43,0.35)!important; border-radius:3px!important; }}
.stInfo    {{ background:rgba(21,101,192,0.06)!important; border:1px solid rgba(21,101,192,0.3)!important; border-radius:3px!important; }}
.stWarning {{ background:rgba(249,168,37,0.07)!important; border:1px solid rgba(249,168,37,0.35)!important; border-radius:3px!important; }}

/* expander */
.streamlit-expanderHeader {{
    background:{T['surface']}!important; border:1px solid {T['border']}!important;
    border-radius:3px!important; font-family:'Orbitron',monospace!important;
    font-size:0.6rem!important; letter-spacing:2px!important; color:{T['text2']}!important;
}}

hr {{ border-color:{T['border']}!important; margin:1rem 0!important; }}

/* sidebar info rows */
.info-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:0.28rem 0; border-bottom:1px solid {T['border']}; font-size:0.7rem;
}}
.info-label {{ color:{T['info_label']}; font-size:0.62rem; letter-spacing:1px; }}
.info-value {{ color:{T['info_value']}; font-weight:bold; }}

/* section headers */
.sec-hdr {{
    display:flex; align-items:center; gap:10px;
    margin:1.4rem 0 0.8rem 0; padding-bottom:0.4rem;
    border-bottom:1px solid {T['sec_border']};
}}
.sec-hdr-txt {{
    font-family:'Orbitron',monospace; font-size:0.6rem;
    letter-spacing:4px; text-transform:uppercase; color:{T['sec_text']}; white-space:nowrap;
}}
.sec-hdr-num {{ font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:{T['sec_num']}; }}
.sec-hdr-line {{ flex:1; height:1px; background:linear-gradient(90deg,{T['sec_line']},transparent); opacity:0.3; }}

/* ticker band */
.ticker {{
    background:{T['ticker_bg']};
    border-top:1px solid {T['ticker_border']}; border-bottom:1px solid {T['ticker_border']};
    padding:0.18rem 0; margin-bottom:0.5rem;
    font-family:'Share Tech Mono',monospace; font-size:0.56rem; color:{T['ticker_color']};
    letter-spacing:3px; text-align:center;
}}

/* AQI scale ref cards */
.aqi-ref {{
    border-radius:3px; padding:0.55rem 0.5rem;
    text-align:center; border:1px solid; transition:all 0.25s;
    background:{T['surface']};
}}
.aqi-ref:hover {{ transform:translateY(-3px); box-shadow:0 4px 12px rgba(0,0,0,0.12); }}
.aqi-ref-lbl {{ font-family:'Orbitron',monospace; font-size:0.52rem; letter-spacing:2px; }}
.aqi-ref-range {{ font-family:'Share Tech Mono',monospace; font-size:0.64rem; color:{T['text']}; margin:2px 0; }}
.aqi-ref-desc {{ font-family:'Crimson Pro',serif; font-style:italic; font-size:0.65rem; color:{T['text2']}; }}

/* tooltip info card */
.tip-card {{
    background:{T['tip_bg']}; border:1px solid {T['tip_border']}; border-radius:4px;
    padding:0.8rem 1rem 0.8rem 1.2rem; font-size:0.72rem; color:{T['tip_text']};
    margin:0.5rem 0; border-left:3px solid {T['tip_left']};
}}

/* theme toggle label */
.theme-toggle {{
    font-family:'Orbitron',monospace; font-size:0.6rem;
    letter-spacing:3px; color:{T['accent']}; text-transform:uppercase;
    padding:0.4rem 0; display:block;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_PATH   = r"https://github.com/KaustubhSN12/TFT-AQI-Forecasting/blob/main/dataset/val_tft_realistic_continuous.csv"
SCALER_PATH = r"https://github.com/KaustubhSN12/TFT-AQI-Forecasting/blob/main/models/scalers_v3.pkl"
MODEL_PATH  = r"https://github.com/KaustubhSN12/TFT-AQI-Forecasting/blob/main/models/best_tft_v3_model_20.1.pth"


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class GRN(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d,d); self.fc2=nn.Linear(d,d)
        self.gate=nn.Linear(d,d); self.norm=nn.LayerNorm(d)
        self.drop=nn.Dropout(dropout); self.act=nn.ELU()
    def forward(self,x):
        h=self.drop(self.act(self.fc1(x))); h=self.fc2(h)
        return self.norm(x+torch.sigmoid(self.gate(x))*h)

class TFTv3(nn.Module):
    def __init__(self,input_size,hidden=128,pred_len=24,n_heads=4,n_layers=2,dropout=0.2):
        super().__init__()
        self.pred_len=pred_len; self.hidden=hidden
        self.input_proj=nn.Sequential(nn.Linear(input_size,hidden),nn.LayerNorm(hidden),nn.ELU(),nn.Dropout(dropout))
        self.encoder_lstm=nn.LSTM(hidden,hidden,n_layers,batch_first=True,dropout=dropout)
        self.encoder_grn=GRN(hidden,dropout)
        self.attn=nn.MultiheadAttention(hidden,n_heads,dropout=dropout,batch_first=True)
        self.attn_norm=nn.LayerNorm(hidden)
        self.decoder_input_proj=nn.Linear(1+hidden,hidden)
        self.decoder_lstm=nn.LSTM(hidden,hidden,n_layers,batch_first=True,dropout=dropout)
        self.decoder_grn=GRN(hidden,dropout)
        self.hour_embed=nn.Embedding(24,hidden)
        self.head_q10=nn.Linear(hidden,1); self.head_q50=nn.Linear(hidden,1); self.head_q90=nn.Linear(hidden,1)

    def encode(self,x):
        x=self.input_proj(x); enc_out,(h,c)=self.encoder_lstm(x)
        enc_out=self.encoder_grn(enc_out); attn_out,_=self.attn(enc_out,enc_out,enc_out)
        return enc_out,h,c

    def decode_step(self,prev_aqi,hour_idx,dh):
        hour_emb=self.hour_embed(hour_idx)
        inp=self.decoder_input_proj(torch.cat([prev_aqi,hour_emb],dim=-1)).unsqueeze(1)
        dec_out,(h,c)=self.decoder_lstm(inp,dh)
        dec_out=self.decoder_grn(dec_out.squeeze(1))
        return self.head_q10(dec_out).squeeze(-1),self.head_q50(dec_out).squeeze(-1),self.head_q90(dec_out).squeeze(-1),(h,c)

    def forward(self,x,teacher_forcing_ratio=0.0,target_hours=None):
        B=x.shape[0]; _,h,c=self.encode(x); prev_aqi=x[:,-1,0:1]
        q10_l,q50_l,q90_l=[],[],[]; dh=(h,c)
        for step in range(self.pred_len):
            hidx=torch.full((B,),step%24,dtype=torch.long,device=x.device)
            q10,q50,q90,dh=self.decode_step(prev_aqi,hidx,dh)
            q10_l.append(q10); q50_l.append(q50); q90_l.append(q90)
            prev_aqi=q50.unsqueeze(1).detach()
        return torch.stack(q10_l,1),torch.stack(q50_l,1),torch.stack(q90_l,1)


# ─────────────────────────────────────────────────────────────
# AQI HELPERS
# ─────────────────────────────────────────────────────────────
def aqi_cat(val):
    for thr,lbl,clr,desc in AQI_BANDS:
        if val<=thr: return lbl,clr,desc
    return "HAZARDOUS","#c0392b","Emergency"

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME FACTORY  — called fresh for every chart
# ─────────────────────────────────────────────────────────────
def PL():
    """Return base Plotly layout dict for current theme."""
    return dict(
        paper_bgcolor = T['paper_bg'],
        plot_bgcolor  = T['plot_bg'],
        font          = dict(family="Share Tech Mono, monospace",
                             color=T['font_color'], size=10),
        margin        = dict(l=55, r=25, t=45, b=55),
        xaxis         = dict(gridcolor=T['grid'], zerolinecolor=T['grid'],
                             showgrid=True, linecolor=T['border'],
                             tickfont=dict(color=T['font_color']),
                             title_font=dict(color=T['text'], size=10, family='Share Tech Mono, monospace')),
        yaxis         = dict(gridcolor=T['grid'], zerolinecolor=T['grid'],
                             showgrid=True, linecolor=T['border'],
                             tickfont=dict(color=T['font_color']),
                             title_font=dict(color=T['text'], size=10, family='Share Tech Mono, monospace')),
        legend        = dict(bgcolor=T['legend_bg'], bordercolor=T['legend_border'],
                             borderwidth=1, font=dict(size=9, color=T['font_color'])),
        hoverlabel    = dict(bgcolor=T['hover_bg'], bordercolor=T['hover_border'],
                             font=dict(family="Share Tech Mono, monospace",
                                       color=T['hover_font'], size=10)),
    )

def apply_layout(fig, title="", height=420):
    fig.update_layout(
        **PL(),
        title=dict(text=title,
                   font=dict(family="Orbitron, monospace",
                             color=T['title_color'], size=11),
                   x=0.01, xanchor="left"),
        height=height,
    )
    return fig

def colorbar_style(title_text):
    """Correct colorbar dict — compatible with all Plotly v5+."""
    return dict(
        title=dict(text=title_text,
                   font=dict(size=9, family="Orbitron, monospace",
                             color=T['font_color'])),
        tickfont=dict(size=8, color=T['font_color']),
        outlinecolor=T['border'],
        outlinewidth=1,
    )

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def sec_hdr(text, num=""):
    st.markdown(f"""<div class="sec-hdr">
      <span class="sec-hdr-num">{num}</span>
      <span class="sec-hdr-txt">{text}</span>
      <div class="sec-hdr-line"></div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="INITIALIZING SCALER MATRIX…")
def load_scalers():
    with open(SCALER_PATH,'rb') as f: return pickle.load(f)

@st.cache_resource(show_spinner="LOADING NEURAL ARCHITECTURE…")
def load_model_weights(n_feat,hidden,n_heads,n_layers,dropout,pred_len):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=TFTv3(input_size=n_feat,hidden=hidden,pred_len=pred_len,
             n_heads=n_heads,n_layers=n_layers,dropout=dropout)
    state=torch.load(MODEL_PATH,map_location=device)
    m.load_state_dict(state)
    return m.to(device).eval(),device

@st.cache_resource(show_spinner="PARSING SENSOR DATA STREAM…")
def load_data(feature_cols_tuple):
    feature_cols=list(feature_cols_tuple)
    df=pd.read_csv(DATA_PATH)
    df.columns=[c.replace('.','_') for c in df.columns]
    df['datetime']=pd.to_datetime(df['datetime'])
    df=df.sort_values(['city','station','datetime']).reset_index(drop=True)
    missing=[c for c in feature_cols if c not in df.columns]
    for col in missing: df[col]=0.0
    return df,feature_cols,missing


# ─────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────
boot_ok=False; boot_err=""; missing_cols=[]
try:
    pkg=load_scalers()
    scaler_X=pkg['scaler_X']; scaler_y=pkg['scaler_y']
    feat_cols=pkg['feature_cols']
    SEQ_LEN=pkg['seq_len']; PRED_LEN=pkg['pred_len']
    HIDDEN=pkg['hidden'];   N_FEAT=pkg['n_feat']
    N_HEADS=pkg['n_heads']; N_LAYERS=pkg['n_layers']; DROPOUT=pkg['dropout']
    model,device=load_model_weights(N_FEAT,HIDDEN,N_HEADS,N_LAYERS,DROPOUT,PRED_LEN)
    df,feat_cols,missing_cols=load_data(tuple(feat_cols))
    boot_ok=True
except Exception as e:
    boot_err=str(e)

pollutant_display=['PM2_5','PM10','NO2','SO2','CO','OZONE','NH3']


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""<div class="ticker">
  ◈ ATMOSPHERIC INTELLIGENCE PLATFORM &nbsp;·&nbsp; TFT v3 AUTOREGRESSIVE ENGINE &nbsp;·&nbsp;
  QUANTILE PREDICTION Q10/Q50/Q90 &nbsp;·&nbsp; 24-HOUR FORECAST HORIZON &nbsp;·&nbsp;
  BY KAUSTUBH NARAYANKAR ◈
</div>""", unsafe_allow_html=True)

col_hdr,col_hdr_r=st.columns([3,1])
with col_hdr:
    st.markdown(f"""
    <div style="padding:0.6rem 0 0.8rem 0;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:{T['muted']};
                  letter-spacing:4px;margin-bottom:4px;">RESEARCH CONSOLE · SYSTEM ACTIVE</div>
      <h1 style="margin:0;font-size:1.7rem;">🛰️ AQI FORECAST SYSTEM</h1>
      <div style="font-family:'{'EB Garamond' if st.session_state.light_mode else 'Crimson Pro'}',serif;
                  font-style:italic;font-size:1rem;color:{T['text2']};margin-top:4px;">
        Temporal Fusion Transformer v3 · Autoregressive Multi-Quantile Prediction
      </div>
    </div>""", unsafe_allow_html=True)
with col_hdr_r:
    st.markdown(f"""
    <div style="text-align:right;padding-top:1rem;">
      <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:{T['accent3']};">TFT v3 · ONLINE</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:{T['muted']};margin-top:4px;">
        KAUSTUBH NARAYANKAR
      </div>
    </div>""", unsafe_allow_html=True)

if not boot_ok:
    st.error(f"⚠️ SYSTEM BOOT FAILURE\n```\n{boot_err}\n```")
    st.stop()

if missing_cols:
    st.info(f"⚡ AUTO-FILL: {len(missing_cols)} feature(s) absent → padded with 0.0 — "
            f"`{'`, `'.join(missing_cols)}`")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # ── THEME TOGGLE ──────────────────────────────────────────
    st.markdown(f'<span class="theme-toggle">🎨 DISPLAY THEME</span>', unsafe_allow_html=True)
    light_toggle = st.toggle(
        "☀️ Research Paper (White)" if not st.session_state.light_mode else "🌙 Console (Dark)",
        value=st.session_state.light_mode,
        key="theme_toggle_widget",
        help="Switch between dark aerospace console and white research paper mode"
    )
    if light_toggle != st.session_state.light_mode:
        st.session_state.light_mode = light_toggle
        st.rerun()

    # Mode badge
    mode_label = "☀️ RESEARCH PAPER MODE" if st.session_state.light_mode else "🌙 CONSOLE MODE"
    mode_color = T['accent']
    st.markdown(f"""<div style="background:{T['surface2']};border:1px solid {T['border']};
        border-radius:3px;padding:0.3rem 0.6rem;text-align:center;margin-bottom:0.5rem;">
      <span style="font-family:'Orbitron',monospace;font-size:0.52rem;
                   letter-spacing:2px;color:{mode_color};">{mode_label}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📡 LOCATION TARGET")
    cities=sorted(df['city'].unique()); city=st.selectbox("CITY",cities)
    stations=sorted(df[df['city']==city]['station'].unique()); station=st.selectbox("STATION",stations)

    st.markdown("---")
    st.markdown("## ⚙️ MODEL PARAMS")
    st.markdown(f"""
    <div class="info-row"><span class="info-label">SEQ LEN</span><span class="info-value">{SEQ_LEN}h</span></div>
    <div class="info-row"><span class="info-label">HORIZON</span><span class="info-value">{PRED_LEN}h</span></div>
    <div class="info-row"><span class="info-label">FEATURES</span><span class="info-value">{N_FEAT}</span></div>
    <div class="info-row"><span class="info-label">HIDDEN DIM</span><span class="info-value">{HIDDEN}</span></div>
    <div class="info-row"><span class="info-label">ATTN HEADS</span><span class="info-value">{N_HEADS}</span></div>
    <div class="info-row"><span class="info-label">LSTM LAYERS</span><span class="info-value">{N_LAYERS}</span></div>
    <div class="info-row"><span class="info-label">DROPOUT</span><span class="info-value">{DROPOUT}</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🖥️ COMPUTE")
    if torch.cuda.is_available():
        st.markdown(f"""
        <div class="info-row"><span class="info-label">DEVICE</span>
          <span class="info-value" style="color:{T['accent3']};">GPU ✅</span></div>
        <div class="info-row"><span class="info-label">GPU</span>
          <span class="info-value" style="font-size:0.58rem;">{torch.cuda.get_device_name(0)[:22]}</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-row"><span class="info-label">DEVICE</span>'
                    f'<span class="info-value" style="color:{T["warn"]};">CPU</span></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎛️ INPUT MODE")
    use_manual=st.checkbox("✏️ Manual feature override",value=False)
    st.markdown("---")
    run_btn=st.button("⚡  EXECUTE FORECAST",use_container_width=True)


# ─────────────────────────────────────────────────────────────
# STATION DATA
# ─────────────────────────────────────────────────────────────
sdf=df[(df['city']==city)&(df['station']==station)].copy()
sdf=sdf.sort_values('datetime').reset_index(drop=True)

tab1,tab2,tab3,tab4=st.tabs([
    "◈  OVERVIEW","◈  FORECAST","◈  HISTORICAL","◈  POLLUTANTS"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    if len(sdf)==0:
        st.warning("NO DATA FOR SELECTED LOCATION"); st.stop()

    latest=sdf.iloc[-1]; aqi_now=float(latest['AQI'])
    cat,col,desc=aqi_cat(aqi_now)

    sec_hdr("LIVE STATION METRICS","01")
    k1,k2,k3,k4=st.columns(4)
    with k1: st.metric("CURRENT AQI",f"{aqi_now:.0f}",f"{cat}  ·  {desc}")
    with k2:
        a7=sdf['AQI'].tail(24*7).mean(); c2,_,_=aqi_cat(a7)
        st.metric("7-DAY AVG AQI",f"{a7:.0f}",c2)
    with k3:
        mx=sdf['AQI'].tail(24*7).max(); c3,_,_=aqi_cat(mx)
        st.metric("7-DAY PEAK AQI",f"{mx:.0f}",c3)
    with k4:
        st.metric("DATA RECORDS",f"{len(sdf):,}",
                  f"{sdf['datetime'].min().date()} → {sdf['datetime'].max().date()}")

    st.markdown("---")
    sec_hdr("AQI CATEGORY DISTRIBUTION & POLLUTANT SNAPSHOT","02")
    col_pie,col_snap=st.columns(2)

    with col_pie:
        recent=sdf['AQI'].tail(24*7); bc={}
        for v in recent:
            n,_,_=aqi_cat(v); bc[n]=bc.get(n,0)+1
        band_order=["GOOD","MODERATE","USG","UNHEALTHY","VERY UNHLT.","HAZARDOUS"]
        bclr_map={"GOOD":"#27ae60","MODERATE":"#f39c12","USG":"#e67e22",
                  "UNHEALTHY":"#e74c3c","VERY UNHLT.":"#8e44ad","HAZARDOUS":"#c0392b"}
        pb=[b for b in band_order if b in bc]
        pc=[bclr_map[b] for b in pb]; pv=[bc[b] for b in pb]
        fig_pie=go.Figure(go.Pie(
            labels=pb, values=pv, hole=0.55,
            marker=dict(colors=pc, line=dict(color=T['pie_edge'], width=2)),
            textfont=dict(family="Share Tech Mono, monospace", size=9, color=T['font_color']),
            hovertemplate="<b>%{label}</b><br>Hours: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"<b>{len(recent)}</b><br>HOURS",
                               x=0.5,y=0.5,showarrow=False,
                               font=dict(family="Orbitron, monospace",color=T['accent'],size=12))
        apply_layout(fig_pie,"AQI CATEGORY BREAKDOWN — LAST 7 DAYS",360)
        fig_pie.update_layout(showlegend=True,legend=dict(orientation="v",x=1.02,y=0.5))
        st.plotly_chart(fig_pie,use_container_width=True)

    with col_snap:
        dp=[p for p in pollutant_display if p in sdf.columns and p not in missing_cols]
        if dp:
            vals=[float(latest[p]) for p in dp]; names=[p.replace('_','') for p in dp]
            fig_bar=go.Figure(go.Bar(
                x=vals, y=names, orientation='h',
                marker=dict(color=POLL_COLORS[:len(dp)],line=dict(color=T['bar_edge'],width=0.5)),
                text=[f"{v:.1f}" for v in vals], textposition='outside',
                textfont=dict(size=9,color=T['font_color']),
                hovertemplate="<b>%{y}</b><br>Concentration: %{x:.2f}<extra></extra>",
            ))
            apply_layout(fig_bar,"LATEST SENSOR READINGS (µg/m³ or ppb)",360)
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar,use_container_width=True)

    sec_hdr("AQI SCALE REFERENCE","03")
    ref_cols=st.columns(6)
    for i,(thr,lbl,clr,desc) in enumerate(AQI_BANDS):
        lo=AQI_BANDS[i-1][0]+1 if i>0 else 0
        r,g,b=int(clr[1:3],16),int(clr[3:5],16),int(clr[5:7],16)
        with ref_cols[i]:
            st.markdown(f"""
            <div class="aqi-ref" style="background:rgba({r},{g},{b},0.12);border-color:{clr};">
              <div class="aqi-ref-lbl" style="color:{clr};">{lbl}</div>
              <div class="aqi-ref-range">{lo}–{thr if thr<999 else '500+'}</div>
              <div class="aqi-ref-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — FORECAST
# ══════════════════════════════════════════════════════════════
with tab2:
    sec_hdr("24-HOUR QUANTILE FORECAST ENGINE","01")
    if len(sdf)<SEQ_LEN:
        st.error(f"INSUFFICIENT DATA: Need ≥{SEQ_LEN} rows, found {len(sdf)}")
    else:
        if use_manual:
            with st.expander("⚙️ MANUAL FEATURE INPUT MATRIX",expanded=True):
                st.markdown(f'<div class="tip-card">Values replicated {SEQ_LEN}× as encoder input.</div>',
                            unsafe_allow_html=True)
                rep={}; c3=st.columns(3)
                for i,fc in enumerate(feat_cols):
                    with c3[i%3]: rep[fc]=st.number_input(fc,value=0.0,key=f"m_{fc}")
                input_array=np.array([[rep[c] for c in feat_cols]]*SEQ_LEN,dtype=np.float32)
        else:
            seq_df=sdf.tail(SEQ_LEN)
            input_array=seq_df[feat_cols].values.astype(np.float32)
            with st.expander("🔍 INPUT SEQUENCE PREVIEW  [LAST 6 ROWS]",expanded=False):
                prev=seq_df[['datetime']+feat_cols].tail(6).copy()
                prev['datetime']=prev['datetime'].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(prev,use_container_width=True,hide_index=True)

        if run_btn:
            with st.spinner("◈ RUNNING AUTOREGRESSIVE DECODE SEQUENCE…"):
                try:
                    assert input_array.shape==(SEQ_LEN,N_FEAT),\
                        f"Shape {input_array.shape} ≠ ({SEQ_LEN},{N_FEAT})"
                    inp_s=scaler_X.transform(input_array)
                    tensor=torch.FloatTensor(inp_s).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q10t,q50t,q90t=model(tensor,teacher_forcing_ratio=0.0)
                    def inv(t):
                        return scaler_y.inverse_transform(t.cpu().numpy().reshape(-1,1)).flatten()
                    q10=inv(q10t[0]); q50=inv(q50t[0]); q90=inv(q90t[0])
                    forecast=q50
                    last_ts=sdf['datetime'].iloc[-1]
                    future_ts=pd.date_range(last_ts+pd.Timedelta(hours=1),periods=PRED_LEN,freq='h')
                    hist_ts=sdf['datetime'].tail(SEQ_LEN)
                    hist_aqi=sdf['AQI'].tail(SEQ_LEN).values
                    hrs=[f"T+{i+1:02d}" for i in range(PRED_LEN)]
                    avg_f=forecast.mean(); max_f=forecast.max(); min_f=forecast.min()
                    ca,coa,da=aqi_cat(avg_f); cx,cox,_=aqi_cat(max_f); cn,con,_=aqi_cat(min_f)
                    spread_avg=(q90-q10).mean()
                    trend="📈 RISING" if forecast[-1]>forecast[0] else "📉 FALLING"

                    sec_hdr("QUANTILE FORECAST SUMMARY","02")
                    k1,k2,k3,k4=st.columns(4)
                    with k1: st.metric("AVG FORECAST · Q50",f"{avg_f:.0f}",f"{ca}  ·  {da}")
                    with k2: st.metric("PEAK FORECAST",f"{max_f:.0f}",cx)
                    with k3: st.metric("MIN FORECAST",f"{min_f:.0f}",cn)
                    with k4: st.metric("24H TREND",trend,f"±{spread_avg:.0f} AQI AVG SPREAD")
                    st.markdown("---")

                    # Chart 1 — main forecast
                    sec_hdr("TEMPORAL FORECAST VISUALIZATION","03")
                    fig1=go.Figure()
                    band_regions=[(0,50,"#27ae60"),(50,100,"#f39c12"),(100,150,"#e67e22"),
                                  (150,200,"#e74c3c"),(200,300,"#8e44ad"),(300,500,"#c0392b")]
                    for lo,hi,bc_clr in band_regions:
                        fig1.add_hrect(y0=lo,y1=hi,fillcolor=bc_clr,opacity=0.05,
                                       layer="below",line_width=0)
                    fig1.add_trace(go.Scatter(
                        x=hist_ts,y=hist_aqi,name=f"HISTORICAL ({SEQ_LEN}h)",
                        line=dict(color=T['hist_line'],width=1.5),
                        hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<extra>Historical</extra>",
                    ))
                    fig1.add_trace(go.Scatter(
                        x=list(future_ts)+list(future_ts)[::-1],
                        y=list(q90)+list(q10)[::-1],
                        fill='toself',fillcolor=T['ribbon_fill'],
                        line=dict(color='rgba(0,0,0,0)'),name="Q10–Q90 INTERVAL",hoverinfo='skip',
                    ))
                    fig1.add_trace(go.Scatter(
                        x=future_ts,y=q10,name="Q10 OPTIMISTIC",
                        line=dict(color=T['q10_line'],width=1.2,dash="dot"),
                        hovertemplate="T+%{pointNumber}h<br>Q10: %{y:.1f}<extra></extra>",
                    ))
                    fig1.add_trace(go.Scatter(
                        x=future_ts,y=q90,name="Q90 PESSIMISTIC",
                        line=dict(color=T['q90_line'],width=1.2,dash="dot"),
                        hovertemplate="T+%{pointNumber}h<br>Q90: %{y:.1f}<extra></extra>",
                    ))
                    fig1.add_trace(go.Scatter(
                        x=future_ts,y=q50,name="Q50 MEDIAN",
                        line=dict(color=T['q50_line'],width=3),
                        mode='lines+markers',
                        marker=dict(size=6,color=T['paper_bg'],
                                    line=dict(color=T['q50_line'],width=2)),
                        hovertemplate="<b>%{x}</b><br>Q50: %{y:.1f} AQI<extra>Forecast</extra>",
                    ))
                    fig1.add_vline(x=last_ts,line=dict(color=T['now_line'],width=1.2,dash="dash"))
                    fig1.add_annotation(x=last_ts,y=1,yref="paper",text=" T₀",
                        showarrow=False,font=dict(color=T['now_annot'],size=9),xanchor="left")
                    apply_layout(fig1,f"{city.upper()} / {station.upper()} — 24H MULTI-QUANTILE FORECAST",480)
                    fig1.update_layout(yaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),xaxis_title=dict(text="DATETIME", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),hovermode="x unified")
                    st.plotly_chart(fig1,use_container_width=True)

                    # Charts 2 + 3 side by side
                    sec_hdr("HOURLY DECOMPOSITION","04")
                    c_left,c_right=st.columns([3,2])
                    with c_left:
                        fig2=go.Figure()
                        fig2.add_trace(go.Bar(name="Q10 OPTIMISTIC",x=hrs,y=q10,
                            marker=dict(color=T['q10_line'],opacity=0.85,
                                        line=dict(color=T['bar_edge'],width=0.4)),
                            hovertemplate="<b>%{x}</b><br>Q10: %{y:.1f}<extra></extra>"))
                        fig2.add_trace(go.Bar(name="Q50 MEDIAN",x=hrs,y=q50,
                            marker=dict(color=T['q50_line'],opacity=0.85,
                                        line=dict(color=T['bar_edge'],width=0.4)),
                            hovertemplate="<b>%{x}</b><br>Q50: %{y:.1f}<extra></extra>"))
                        fig2.add_trace(go.Bar(name="Q90 PESSIMISTIC",x=hrs,y=q90,
                            marker=dict(color=T['q90_line'],opacity=0.85,
                                        line=dict(color=T['bar_edge'],width=0.4)),
                            hovertemplate="<b>%{x}</b><br>Q90: %{y:.1f}<extra></extra>"))
                        apply_layout(fig2,"Q10 / Q50 / Q90 PER FORECAST HOUR",380)
                        fig2.update_layout(barmode="group",yaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),
                                           bargap=0.15,bargroupgap=0.05)
                        st.plotly_chart(fig2,use_container_width=True)
                    with c_right:
                        spread=q90-q10
                        fig4=go.Figure()
                        fig4.add_trace(go.Scatter(x=hrs,y=spread,name="Q90−Q10 SPREAD",
                            fill='tozeroy',fillcolor=T['spread_fill'],
                            line=dict(color=T['spread_line'],width=2),mode='lines+markers',
                            marker=dict(size=5,color=T['paper_bg'],
                                        line=dict(color=T['spread_line'],width=1.5)),
                            hovertemplate="<b>%{x}</b><br>Spread: %{y:.1f} AQI<extra></extra>"))
                        fig4.add_hline(y=spread.mean(),
                            line=dict(color=T['spread_line'],width=1,dash="dash"),
                            annotation_text=f"MEAN {spread.mean():.0f}",
                            annotation_font_color=T['spread_line'],annotation_font_size=8)
                        apply_layout(fig4,"UNCERTAINTY WIDTH (Q90 − Q10)",380)
                        fig4.update_layout(yaxis_title=dict(text="AQI SPREAD", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),xaxis_title=dict(text="HOUR AHEAD", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")))
                        st.plotly_chart(fig4,use_container_width=True)

                    # Chart 4 — category timeline
                    sec_hdr("AQI CATEGORY TIMELINE · Q50 MEDIAN","05")
                    cat_colors=[aqi_cat(v)[1] for v in q50]
                    cat_labels=[aqi_cat(v)[0] for v in q50]
                    fig3=go.Figure(go.Bar(
                        x=hrs,y=q50,
                        marker=dict(color=cat_colors,line=dict(color=T['bar_edge'],width=0.5)),
                        text=[f"{v:.0f}" for v in q50],textposition='outside',
                        textfont=dict(size=8,color=T['font_color']),
                        customdata=cat_labels,
                        hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<br>Category: %{customdata}<extra></extra>",
                    ))
                    apply_layout(fig3,"HOUR-BY-HOUR FORECAST AQI (CLICK BARS TO COMPARE)",380)
                    fig3.update_layout(yaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),xaxis_title=dict(text="FORECAST HOUR", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")))
                    for thr,lbl,clr,_ in AQI_BANDS[:-1]:
                        fig3.add_hline(y=thr,line=dict(color=clr,width=0.7,dash="dot"),
                                       annotation_text=lbl,annotation_font_size=7,
                                       annotation_font_color=clr,annotation_position="right")
                    st.plotly_chart(fig3,use_container_width=True)

                    # Forecast table
                    sec_hdr("FULL HOURLY FORECAST TABLE","06")
                    fdf=pd.DataFrame({
                        "HOUR":hrs,"DATETIME":future_ts.strftime("%Y-%m-%d %H:%M"),
                        "Q10  BEST":[f"{v:.1f}" for v in q10],
                        "Q50  MEDIAN":[f"{v:.1f}" for v in q50],
                        "Q90  WORST":[f"{v:.1f}" for v in q90],
                        "SPREAD":[f"{(q90[i]-q10[i]):.1f}" for i in range(PRED_LEN)],
                        "CATEGORY":cat_labels,
                    })
                    st.dataframe(fdf,use_container_width=True,hide_index=True)

                    # Radar chart
                    dp_r=[p for p in pollutant_display if p in sdf.columns and p not in missing_cols]
                    if dp_r:
                        sec_hdr("POLLUTANT RADAR PROFILE · CURRENT vs 7-DAY AVG","07")
                        curr_vals=[float(sdf.iloc[-1][p]) for p in dp_r]
                        avg_vals=[float(sdf[p].tail(24*7).mean()) for p in dp_r]
                        pnames=[p.replace('_','') for p in dp_r]
                        fig_radar=go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=curr_vals+[curr_vals[0]],theta=pnames+[pnames[0]],
                            fill='toself',fillcolor=T['radar_fill1'],
                            line=dict(color=T['q50_line'],width=2),name="CURRENT",
                            hovertemplate="<b>%{theta}</b><br>%{r:.2f}<extra>Current</extra>"))
                        fig_radar.add_trace(go.Scatterpolar(
                            r=avg_vals+[avg_vals[0]],theta=pnames+[pnames[0]],
                            fill='toself',fillcolor=T['radar_fill2'],
                            line=dict(color=T['spread_line'],width=1.5,dash="dash"),name="7-DAY AVG",
                            hovertemplate="<b>%{theta}</b><br>%{r:.2f}<extra>7-Day Avg</extra>"))
                        fig_radar.update_layout(**PL(),height=380,
                            polar=dict(bgcolor=T['radar_bg'],
                                radialaxis=dict(visible=True,gridcolor=T['radar_grid'],
                                                color=T['font_color'],tickfont=dict(size=7)),
                                angularaxis=dict(gridcolor=T['radar_grid'],color=T['radar_ang'],
                                                 tickfont=dict(family="Share Tech Mono",size=9))),
                            title=dict(text="POLLUTANT RADAR",
                                       font=dict(family="Orbitron",color=T['title_color'],size=11),x=0.01))
                        st.plotly_chart(fig_radar,use_container_width=True)

                    st.success(f"✅ FORECAST COMPLETE · DEVICE: {str(device).upper()} · "
                               f"{PRED_LEN} STEPS DECODED · Q50 AVG: {avg_f:.1f} · CATEGORY: {ca}")

                except Exception as e:
                    st.error(f"⚠️ INFERENCE ERROR: {e}")
        else:
            st.markdown(f"""<div class="tip-card">
              SELECT CITY + STATION IN SIDEBAR, THEN PRESS
              <span style="color:{T['accent']};font-family:'Orbitron',monospace;">⚡ EXECUTE FORECAST</span>
              TO RUN THE AUTOREGRESSIVE DECODER.<br><br>
              ENCODER INPUT: LAST {SEQ_LEN} HOURS FROM SELECTED STATION.<br>
              OUTPUT: {PRED_LEN} FUTURE HOURS × Q10 / Q50 / Q90 QUANTILES.<br>
              ALL CHARTS ARE INTERACTIVE — HOVER, CLICK, ZOOM, PAN, TOGGLE SERIES.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL
# ══════════════════════════════════════════════════════════════
with tab3:
    sec_hdr("HISTORICAL AQI ANALYSIS","01")
    if len(sdf)==0:
        st.warning("NO DATA FOR SELECTED LOCATION")
    else:
        c_ctrl,_=st.columns([1,3])
        with c_ctrl:
            win=st.selectbox("WINDOW",[7,14,30,60,90],index=2,key="hist_win",
                             format_func=lambda x:f"LAST {x} DAYS")
        hist=sdf.tail(win*24).copy()
        hist['AQI_roll24']=hist['AQI'].rolling(24,min_periods=1).mean()
        hist['AQI_roll6']=hist['AQI'].rolling(6,min_periods=1).mean()

        k1,k2,k3,k4,k5=st.columns(5)
        for col_w,lbl,val in [(k1,"MEAN AQI",hist['AQI'].mean()),(k2,"STD DEV",hist['AQI'].std()),
                               (k3,"PEAK AQI",hist['AQI'].max()),(k4,"MIN AQI",hist['AQI'].min()),
                               (k5,"RECORDS",len(hist))]:
            with col_w:
                ca2,_,_=aqi_cat(val) if "AQI" in lbl else (f"OVER {win} DAYS","","")
                st.metric(lbl,f"{val:.0f}",ca2)

        st.markdown("---")
        sec_hdr("AQI TIME SERIES","02")
        fig_ts=go.Figure()
        fig_ts.add_trace(go.Scatter(x=hist['datetime'],y=hist['AQI'],name="RAW AQI",
            line=dict(color=T['raw_line'],width=0.8),opacity=0.5,
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>"))
        fig_ts.add_trace(go.Scatter(x=hist['datetime'],y=hist['AQI_roll6'],name="6H ROLLING",
            line=dict(color=T['roll6_line'],width=1.5),
            hovertemplate="<b>%{x}</b><br>6h Avg: %{y:.1f}<extra></extra>"))
        fig_ts.add_trace(go.Scatter(x=hist['datetime'],y=hist['AQI_roll24'],name="24H ROLLING",
            line=dict(color=T['roll24_line'],width=2.5),
            hovertemplate="<b>%{x}</b><br>24h Avg: %{y:.1f}<extra></extra>"))
        for thr,lbl,clr,_ in AQI_BANDS[:-1]:
            fig_ts.add_hline(y=thr,line=dict(color=clr,width=0.5,dash="dot"),
                             annotation_text=lbl,annotation_font_size=7,
                             annotation_font_color=clr,annotation_position="right")
        apply_layout(fig_ts,f"AQI TIME SERIES — LAST {win} DAYS  ({city}/{station})",450)
        fig_ts.update_layout(yaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),hovermode="x unified",
            xaxis=dict(
                rangeselector=dict(
                    buttons=[dict(count=1,label="1D",step="day",stepmode="backward"),
                             dict(count=7,label="7D",step="day",stepmode="backward"),
                             dict(count=14,label="14D",step="day",stepmode="backward"),
                             dict(step="all",label="ALL")],
                    bgcolor=T['rangeselector_bg'],
                    activecolor=T['rangeselector_active'],
                    font=dict(color=T['rangeselector_font'],size=9)),
                rangeslider=dict(visible=True,bgcolor=T['rangeslider_bg'],thickness=0.04),
                type="date"))
        st.plotly_chart(fig_ts,use_container_width=True)

        sec_hdr("DISTRIBUTION & DIURNAL PATTERN","03")
        c_dist,c_diurnal=st.columns(2)
        with c_dist:
            fig_h=go.Figure()
            bins_=np.arange(hist['AQI'].min(),hist['AQI'].max()+5,5)
            for lo_b,hi_b in zip(bins_[:-1],bins_[1:]):
                mask=(hist['AQI']>=lo_b)&(hist['AQI']<hi_b); count=mask.sum()
                if count==0: continue
                _,clr_b,_=aqi_cat((lo_b+hi_b)/2)
                fig_h.add_trace(go.Bar(x=[(lo_b+hi_b)/2],y=[count],width=hi_b-lo_b,
                    marker=dict(color=clr_b,opacity=0.8,line=dict(color=T['bar_edge'],width=0.3)),
                    name=f"{lo_b:.0f}–{hi_b:.0f}",showlegend=False,
                    hovertemplate=f"AQI {lo_b:.0f}–{hi_b:.0f}<br>Count: {count}<extra></extra>"))
            apply_layout(fig_h,"AQI FREQUENCY DISTRIBUTION",360)
            fig_h.update_layout(barmode="overlay",yaxis_title=dict(text="FREQUENCY", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),xaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),bargap=0)
            st.plotly_chart(fig_h,use_container_width=True)

        with c_diurnal:
            hist['hour']=hist['datetime'].dt.hour
            hourly=hist.groupby('hour')['AQI'].agg(['mean','std']).reset_index()
            fig_d=go.Figure()
            fig_d.add_trace(go.Scatter(
                x=list(hourly['hour'])+list(hourly['hour'])[::-1],
                y=list(hourly['mean']+hourly['std'])+list(hourly['mean']-hourly['std'])[::-1],
                fill='toself',fillcolor=T['ribbon_fill'],
                line=dict(color='rgba(0,0,0,0)'),name="±1 STD",hoverinfo='skip'))
            fig_d.add_trace(go.Bar(x=hourly['hour'],y=hourly['mean'],
                marker=dict(color=[aqi_cat(v)[1] for v in hourly['mean']],opacity=0.7,
                            line=dict(color=T['bar_edge'],width=0.3)),
                name="AVG AQI",
                hovertemplate="Hour %{x}:00<br>Avg AQI: %{y:.1f}<extra></extra>"))
            fig_d.add_trace(go.Scatter(x=hourly['hour'],y=hourly['mean'],name="TREND",
                line=dict(color=T['raw_line'],width=2.5),mode='lines+markers',
                marker=dict(size=5,color=T['paper_bg'],line=dict(color=T['raw_line'],width=1.5)),
                hovertemplate="Hour %{x}:00<br>Avg AQI: %{y:.1f}<extra></extra>"))
            apply_layout(fig_d,"AVG AQI BY HOUR OF DAY (DIURNAL PATTERN)",360)
            fig_d.update_layout(yaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),xaxis_title=dict(text="HOUR", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),
                                xaxis=dict(tickmode='array',tickvals=list(range(24)),
                                           ticktext=[str(h) for h in range(24)]),barmode="overlay")
            st.plotly_chart(fig_d,use_container_width=True)

        sec_hdr("DATE × HOUR HEATMAP","04")
        hist['date']=hist['datetime'].dt.date
        pivot=hist.pivot_table(values='AQI',index='date',columns='hour',aggfunc='mean')
        if not pivot.empty:
            fig_hm=go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(h) for h in pivot.columns],
                y=[str(d) for d in pivot.index],
                colorscale=[[0,"#27ae60"],[0.17,"#f39c12"],[0.33,"#e67e22"],
                             [0.5,"#e74c3c"],[0.67,"#8e44ad"],[1.0,"#c0392b"]],
                hovertemplate="Date: %{y}<br>Hour: %{x}:00<br>AQI: %{z:.1f}<extra></extra>",
                colorbar=colorbar_style("AQI"),
            ))
            apply_layout(fig_hm,"AQI INTENSITY HEATMAP — DATE × HOUR OF DAY",max(300,len(pivot)*14))
            fig_hm.update_layout(
                xaxis=dict(
                    title=dict(text="HOUR OF DAY",
                               font=dict(family="Orbitron, monospace",
                                         color=T["text"], size=11)),
                    tickfont=dict(size=8, color=T["text"]),
                ),
                yaxis=dict(
                    title=dict(text="DATE",
                               font=dict(family="Orbitron, monospace",
                                         color=T["text"], size=11)),
                    tickfont=dict(size=7, color=T["text"]),
                    autorange="reversed",
                ),
            )
            st.plotly_chart(fig_hm,use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — POLLUTANTS
# ══════════════════════════════════════════════════════════════
with tab4:
    sec_hdr("POLLUTANT CONCENTRATION ANALYSIS","01")
    dp=[p for p in pollutant_display if p in sdf.columns and p not in missing_cols]
    if len(sdf)==0 or not dp:
        st.warning("NO REAL POLLUTANT DATA AVAILABLE")
    else:
        c_ctrl2,_=st.columns([1,3])
        with c_ctrl2:
            days_p=st.selectbox("WINDOW",[3,7,14,30],index=1,key="poll_win",
                                format_func=lambda x:f"LAST {x} DAYS")
        hp=sdf.tail(days_p*24).copy()

        sec_hdr("POLLUTANT TREND OVERLAY  [CLICK LEGEND TO TOGGLE]","02")
        fig_all=go.Figure()
        for p,clr in zip(dp,POLL_COLORS):
            fig_all.add_trace(go.Scatter(
                x=hp['datetime'],y=hp[p],name=p.replace('_','').upper(),
                line=dict(color=clr,width=1.5),
                hovertemplate=f"<b>%{{x}}</b><br>{p.replace('_','')}: %{{y:.2f}}<extra></extra>"))
        apply_layout(fig_all,"ALL POLLUTANTS — INTERACTIVE OVERLAY (CLICK LEGEND TO ISOLATE)",400)
        fig_all.update_layout(yaxis_title=dict(text="CONCENTRATION", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),hovermode="x unified",
                              legend=dict(orientation="h",y=-0.15))
        st.plotly_chart(fig_all,use_container_width=True)

        sec_hdr("INDIVIDUAL POLLUTANT TRENDS","03")
        n_dp=len(dp); n_cols_grid=2; n_rows_grid=(n_dp+1)//n_cols_grid
        fig_grid=make_subplots(rows=n_rows_grid,cols=n_cols_grid,
                               subplot_titles=[p.replace('_','').upper() for p in dp],
                               vertical_spacing=0.12,horizontal_spacing=0.08)
        for idx,(p,clr) in enumerate(zip(dp,POLL_COLORS)):
            r,c=(idx//n_cols_grid)+1,(idx%n_cols_grid)+1
            fig_grid.add_trace(go.Scatter(
                x=hp['datetime'],y=hp[p],name=p.replace('_','').upper(),
                line=dict(color=clr,width=1.3),showlegend=False,
                fill='tozeroy',
                fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.10)",
                hovertemplate=f"{p.replace('_','')}: %{{y:.2f}}<extra></extra>"),row=r,col=c)
        fig_grid.update_layout(**PL(),height=280*n_rows_grid,
            title=dict(text="POLLUTANT GRID — CLICK TO ZOOM, DRAG TO PAN",
                       font=dict(family="Orbitron",color=T['title_color'],size=11),x=0.01))
        for ann in fig_grid.layout.annotations:
            ann.font=dict(family="Share Tech Mono",color=T['font_color'],size=9)
        st.plotly_chart(fig_grid,use_container_width=True)

        sec_hdr("CORRELATION & LAG ANALYSIS","04")
        c_corr,c_lag=st.columns([1,1.2])
        with c_corr:
            cc=dp+['AQI']
            corr=sdf[cc].tail(days_p*24).corr()
            fig_corr=go.Figure(go.Heatmap(
                z=corr.values,
                x=[c.replace('_','') for c in cc],
                y=[c.replace('_','') for c in cc],
                colorscale='RdBu_r',zmid=0,zmin=-1,zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=8,color=T['font_color']),
                hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
                colorbar=colorbar_style("r"),
            ))
            apply_layout(fig_corr,"PEARSON CORRELATION MATRIX",380)
            fig_corr.update_layout(
                xaxis=dict(tickfont=dict(size=8),tickangle=45),
                yaxis=dict(tickfont=dict(size=8),autorange="reversed"))
            st.plotly_chart(fig_corr,use_container_width=True)

        with c_lag:
            lag_cols=[c for c in ['AQI_lag_1h','AQI_lag_6h','AQI_lag_24h']
                      if c in sdf.columns and c not in missing_cols]
            if lag_cols:
                fig_lag=go.Figure()
                fig_lag.add_trace(go.Scatter(x=hp['datetime'],y=hp['AQI'],name="AQI (NOW)",
                    line=dict(color=T['raw_line'],width=2.5),
                    hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<extra>Now</extra>"))
                lag_clrs=["#f9a825","#e67e22","#e74c3c"]
                for lc,lclr in zip(lag_cols,lag_clrs):
                    fig_lag.add_trace(go.Scatter(x=hp['datetime'],y=hp[lc],
                        name=lc.replace('_',' ').upper(),
                        line=dict(color=lclr,width=1.5,dash="dash"),
                        hovertemplate=f"<b>%{{x}}</b><br>{lc}: %{{y:.1f}}<extra></extra>"))
                apply_layout(fig_lag,"AQI LAG FEATURES vs CURRENT",380)
                fig_lag.update_layout(yaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),hovermode="x unified")
                st.plotly_chart(fig_lag,use_container_width=True)

        if 'PM2_5' in sdf.columns and 'PM2_5' not in missing_cols:
            sec_hdr("PM2.5 vs AQI SCATTER ANALYSIS","05")
            sc_df=sdf[['PM2_5','AQI']].tail(days_p*24).dropna()
            sc_colors=[aqi_cat(v)[1] for v in sc_df['AQI']]
            fig_sc=go.Figure(go.Scatter(
                x=sc_df['PM2_5'],y=sc_df['AQI'],mode='markers',
                marker=dict(color=sc_colors,size=4,opacity=0.6,
                            line=dict(color=T['scatter_edge'],width=0.3)),
                hovertemplate="PM2.5: %{x:.2f}<br>AQI: %{y:.1f}<extra></extra>",showlegend=False))
            z=np.polyfit(sc_df['PM2_5'],sc_df['AQI'],1)
            xline=np.linspace(sc_df['PM2_5'].min(),sc_df['PM2_5'].max(),100)
            fig_sc.add_trace(go.Scatter(x=xline,y=np.polyval(z,xline),name="TREND",
                line=dict(color=T['trend_line'],width=2,dash="dash"),hoverinfo='skip'))
            apply_layout(fig_sc,"PM2.5 CONCENTRATION vs AQI  [COLOR = AQI CATEGORY]",380)
            fig_sc.update_layout(xaxis_title=dict(text="PM2.5 (µg/m³)", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")),yaxis_title=dict(text="AQI", font=dict(color=T['text'], size=10, family="Share Tech Mono, monospace")))
            st.plotly_chart(fig_sc,use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2.5rem;padding:0.8rem 0;border-top:1px solid {T['border']};">
  <div class="ticker">
    TFT v3 · AUTOREGRESSIVE DECODER · TEACHER FORCING · MIXED FP16 ·
    best_tft_v3_model_20.1.pth · scalers_v3.pkl · © KAUSTUBH NARAYANKAR
  </div>
</div>
""", unsafe_allow_html=True)
