"""
dashboard/app.py — AI Governance Framework Dashboard
Streamlit dashboard optimizado para Streamlit.io deployment
"""
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from scipy import stats as sc_stats
from scipy.stats import jarque_bera, binom, t as t_dist
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Governance Framework",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMetric"] {
    color: #000000 !important;
}
[data-testid="stMetricValue"] { font-size: 1.05rem; }
.sub { color: #6b7280; font-size: .82rem; margin-top: -10px; margin-bottom: 10px; }
.badge { 
    display: inline-block; padding: 2px 9px; border-radius: 10px;
    font-size: 11px; font-weight: 600; margin: 2px; 
}
.card { 
    background: #f8fafc; border-radius: 10px; padding: 14px 16px;
    border: 1px solid #e2e8f0; margin-bottom: 8px; 
}
/* CRITICAL: Force ALL Plotly text to be black */
.plotly .svg-container text { fill: black !important; }
.plotly .xaxislayer-above text { fill: black !important; }
.plotly .yaxislayer-above text { fill: black !important; }
.plotly .gtitle { fill: black !important; }
.plotly .xtitle { fill: black !important; }
.plotly .ytitle { fill: black !important; }
.plotly .legendtext { fill: black !important; }
.plotly .hovertext { fill: black !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════

TX = {
    "en": {
        "nav": "Navigation",
        "pages": ["🏛️ MRR Overview", "🔒 Policy Status", "💳 Credit Risk",
                  "📊 Market Risk", "⚖️ Comparative Metrics", "🌈 Fairness",
                  "⚠️ Stress Testing", "🇪🇺 EU AI Act", "🗂️ Audit Trail", "📑 Reports"],
        "no_data": "Showing synthetic demo data",
        "approved": "APPROVED", "review": "UNDER REVIEW", "retired": "RETIRED",
        "model_status": "Portfolio Status", "n_models": "Models", "n_approved": "Approved",
        "mrr_title": "Model Risk Register", "mrr_sub": "SR 11-7 · EU AI Act · BCBS 239",
        "mrr_timeline": "Validation timeline", "mrr_regs": "Regulations covered",
    },
    "es": {
        "nav": "Navegación",
        "pages": ["🏛️ MRR Overview", "🔒 Estado de Políticas", "💳 Riesgo de Crédito",
                  "📊 Riesgo de Mercado", "⚖️ Métricas Comparativas", "🌈 Fairness",
                  "⚠️ Stress Testing", "🇪🇺 EU AI Act", "🗂️ Audit Trail", "📑 Reportes"],
        "no_data": "Mostrando datos sintéticos",
        "approved": "APROBADO", "review": "EN REVISIÓN", "retired": "RETIRADO",
        "model_status": "Estado del Portfolio", "n_models": "Modelos", "n_approved": "Aprobados",
        "mrr_title": "Model Risk Register", "mrr_sub": "SR 11-7 · EU AI Act · BCBS 239",
        "mrr_timeline": "Cronograma de validaciones", "mrr_regs": "Regulaciones cubiertas",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def t(k):
    return TX[st.session_state.lang].get(k, TX["en"].get(k, k))

# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPER - Ensures all text is black
# ══════════════════════════════════════════════════════════════════════════════

def apply_black_theme(fig):
    """Apply black text to all Plotly figure elements"""
    fig.update_layout(
        font=dict(color="black", size=11, family="Arial"),
        title_font=dict(color="black", size=14),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        margin=dict(t=20, b=15, l=15, r=15),
        hoverlabel=dict(font=dict(color="black")),
    )
    fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"))
    fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"))
    fig.update_annotations(font=dict(color="black"))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def get_mrr_data():
    return {
        "n_models": 2, "n_approved": 2, "n_blocks": 0, "n_warns": 1,
        "models": [
            {"name": "Credit Scoring v2.1", "type": "Credit", "tier": "High", 
             "status": "approved", "sr117": 0.88, "psi": 0.06},
            {"name": "Market VaR TFT v1.0", "type": "Market", "tier": "High",
             "status": "approved", "sr117": 0.88, "psi": 0.08},
        ]
    }

def get_credit_data():
    np.random.seed(42)
    n = 8000
    y = np.random.binomial(1, 0.08, n)
    s = np.clip(y*np.random.beta(5,2,n)+(1-y)*np.random.beta(2,5,n), 0.01, 0.99)
    fpr, tpr, _ = roc_curve(y, s)
    gini = 2*roc_auc_score(y, s)-1
    
    return {
        "metrics": {"gini": 0.712, "auc": 0.856, "ks": 0.524, "psi": 0.06},
        "demo": {"y": y, "s": s, "fpr": fpr, "tpr": tpr, "gini": gini}
    }

def get_market_data():
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.today(), periods=500)
    returns = np.random.normal(-0.0003, 0.012, len(dates))
    
    return {
        "returns": pd.Series(returns, index=dates),
        "backtest": {"kupiec_pval": 0.38, "n_exceedances": 3, "zone": "green"}
    }

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MRR Overview
# ══════════════════════════════════════════════════════════════════════════════

def page_mrr():
    st.title(t("mrr_title"))
    st.markdown(f'<p class="sub">{t("mrr_sub")}</p>', unsafe_allow_html=True)
    
    mrr = get_mrr_data()
    c1, c2, c3 = st.columns(3)
    c1.metric(t("n_models"), f"{mrr['n_models']}")
    c2.metric("Policy Blocks", f"{'✅ 0' if mrr['n_blocks']==0 else '❌ '+str(mrr['n_blocks'])}")
    c3.metric("Policy Warns", f"{'✅ 0' if mrr['n_warns']==0 else '⚠️ '+str(mrr['n_warns'])}")
    
    st.divider()
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader(t("mrr_timeline"))
        MODELS = mrr['models']
        n = len(MODELS)
        fig = go.Figure()
        
        for i, m in enumerate(MODELS):
            last = pd.to_datetime("2025-01-15")
            nxt = pd.to_datetime("2025-07-15")
            col = "#4361ee" if m["type"]=="Credit" else "#7209b7"
            
            fig.add_trace(go.Scatter(
                x=[last, nxt], y=[i, i], mode="lines+markers",
                line=dict(color=col, width=6),
                marker=dict(size=12, color=[col, "#ef476f"]),
                name=m["name"][:20],
            ))
        
        fig.update_layout(height=280, showlegend=True, hovermode="closest")
        apply_black_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        st.subheader(t("mrr_regs"))
        regs = {"SR 11-7": 2, "EU AI Act": 2, "Basel III": 1, "IFRS 9": 1}
        
        fig2 = go.Figure(go.Bar(
            x=list(regs.keys()), y=list(regs.values()),
            marker_color=["#4361ee","#7209b7","#f72585","#06d6a0"],
            text=list(regs.values()), textposition="outside",
        ))
        fig2.update_layout(height=280, yaxis_title="Models", xaxis_tickangle=-20)
        apply_black_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Policy Status
# ══════════════════════════════════════════════════════════════════════════════

def page_policy():
    st.title("Policy Status")
    st.markdown('<p class="sub">Policy-as-Code Evaluation</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("PASS", "✅ 7/8")
    c2.metric("Blocked", "✅ 0")
    c3.metric("Warnings", "⚠️ 1")
    
    st.divider()
    fig = go.Figure(go.Pie(
        labels=["PASS", "WARN"], values=[7, 1],
        marker_colors=["#06d6a0", "#ffd166"],
        hole=0.55,
    ))
    fig.update_layout(height=300)
    apply_black_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Credit Risk
# ══════════════════════════════════════════════════════════════════════════════

def page_credit():
    st.title("Credit Risk — Deep Dive")
    st.markdown('<p class="sub">Model performance & regulatory compliance</p>', unsafe_allow_html=True)
    
    cd = get_credit_data()
    m = cd["metrics"]
    d = cd["demo"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gini", f"{m['gini']:.3f}", "✅")
    c2.metric("AUC-ROC", f"{m['auc']:.3f}", "✅")
    c3.metric("KS", f"{m['ks']:.3f}", "✅")
    c4.metric("PSI", f"{m['psi']:.3f}", "stable")
    
    st.divider()
    
    # ROC Curve
    st.subheader("ROC Curve")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=d['fpr'], y=d['tpr'],
                                  line=dict(color="#4361ee", width=3),
                                  name=f"Model (Gini={d['gini']:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                  line=dict(color="#adb5bd", dash="dash"),
                                  name="Random"))
    fig_roc.update_layout(height=350, xaxis_title="FPR", yaxis_title="TPR")
    apply_black_theme(fig_roc)
    st.plotly_chart(fig_roc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Market Risk
# ══════════════════════════════════════════════════════════════════════════════

def page_market():
    st.title("Market Risk — Deep Dive")
    st.markdown('<p class="sub">VaR backtesting & regulatory compliance</p>', unsafe_allow_html=True)
    
    md = get_market_data()
    bt = md["backtest"]
    ret = md["returns"]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Traffic Light", f"🟢 {bt['zone'].upper()}")
    c2.metric("Exceedances", f"{bt['n_exceedances']}/250")
    c3.metric("Kupiec p-val", f"{bt['kupiec_pval']:.3f}", "✅" if bt['kupiec_pval']>0.05 else "❌")
    
    st.divider()
    
    # Returns Plot
    st.subheader("Daily Returns (last 500 days)")
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(
        x=ret.index, y=ret.values,
        line=dict(color="#7209b7", width=1),
        fill="tozeroy",
        fillcolor="rgba(114,9,183,0.2)",
        name="Returns"
    ))
    fig_ret.update_layout(height=300, xaxis_title="Date", yaxis_title="Return")
    apply_black_theme(fig_ret)
    st.plotly_chart(fig_ret, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Comparative Metrics
# ══════════════════════════════════════════════════════════════════════════════

def page_comparative():
    st.title("Comparative Metrics — Credit vs Market")
    
    cd = get_credit_data()
    md = get_market_data()
    
    dims = ["SR 11-7 Score", "Drift Stability", "Stress Resilience", 
            "Fairness", "Regulatory Coverage"]
    credit_scores = [0.88, 0.94, 0.82, 0.96, 0.90]
    market_scores = [0.88, 0.92, 0.84, 0.70, 0.88]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=credit_scores+[credit_scores[0]],
        theta=dims+[dims[0]],
        fill="toself", name="Credit Risk",
        line=dict(color="#4361ee"), fillcolor="rgba(67,97,238,0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=market_scores+[market_scores[0]],
        theta=dims+[dims[0]],
        fill="toself", name="Market Risk",
        line=dict(color="#7209b7"), fillcolor="rgba(114,9,183,0.15)"
    ))
    fig.update_layout(height=450, polar=dict(radialaxis=dict(range=[0,1])))
    apply_black_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Fairness
# ══════════════════════════════════════════════════════════════════════════════

def page_fairness():
    st.title("Fairness Dashboard")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DPD", "0.030", "✅ < 0.05")
    c2.metric("DIR", "0.957", "✅ > 0.80")
    c3.metric("TPR Gap", "0.02", "✅ < 0.05")
    c4.metric("FPR Gap", "0.01", "✅ < 0.05")
    
    st.divider()
    
    fig = go.Figure(go.Bar(
        x=["Male", "Female"],
        y=[0.682, 0.712],
        marker_color=["#4361ee", "#7209b7"],
        text=["68.2%", "71.2%"],
        textposition="outside",
    ))
    fig.update_layout(height=300, yaxis_title="Approval Rate")
    apply_black_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: Stress Testing
# ══════════════════════════════════════════════════════════════════════════════

def page_stress():
    st.title("Stress Testing")
    
    scenarios = ["Income Shock", "Bureau Deterioration", "Market Crisis"]
    credit_gini = [0.672, 0.592, 0.55]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scenarios, y=credit_gini,
        marker_color=["#ffd166", "#f72585", "#ef476f"],
        text=[f"{g:.3f}" for g in credit_gini],
        textposition="outside",
    ))
    fig.add_hline(y=0.45, line_dash="dot", line_color="#4361ee", annotation_text="Min: 0.45")
    fig.update_layout(height=350, yaxis_title="Credit Gini")
    apply_black_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: EU AI Act
# ══════════════════════════════════════════════════════════════════════════════

def page_euai():
    st.title("EU AI Act Compliance")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Compliance", "86%")
    c2.metric("Credit Risk", "6/7 ✅")
    c3.metric("Market Risk", "6/7 ✅")
    
    st.divider()
    st.markdown("""
    **Art. 9** - Risk management: ✅ SR 11-7 + backtesting
    **Art. 10** - Data governance: ✅ Great Expectations
    **Art. 11** - Documentation: ✅ Model Cards + ADRs
    **Art. 12** - Record-keeping: ✅ SHA-256 audit trail
    **Art. 13** - Transparency: ✅ SHAP explanations
    **Art. 14** - Human oversight: ⚠️ Partially documented
    **Art. 15** - Accuracy & robustness: ✅ Stress testing
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9: Audit Trail
# ══════════════════════════════════════════════════════════════════════════════

def page_audit():
    st.title("Audit Trail")
    
    events = [
        {"ts": "2025-01-15 10:30", "event": "model_trained", "source": "credit", "hash": "abc123"},
        {"ts": "2025-01-15 10:45", "event": "sr117_completed", "source": "credit", "hash": "def456"},
        {"ts": "2025-01-15 11:00", "event": "backtesting_passed", "source": "market", "hash": "ghi789"},
    ]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chain Integrity", "✅ Valid")
    c2.metric("Total Events", "7")
    c3.metric("Credit Events", "3")
    c4.metric("Market Events", "3")
    
    st.divider()
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10: Reports
# ══════════════════════════════════════════════════════════════════════════════

def page_reports():
    st.title("Regulatory Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📋 Executive Summary**
        Portfolio-level overview for CRO and board
        """)
    
    with col2:
        st.markdown("""
        **📊 SR 11-7 Report**
        Full three-pillar validation
        """)
    
    st.divider()
    
    exec_data = {
        "Report Field": ["Report Date", "Portfolio", "Models", "Policy Blocks", "Fairness", "Compliance"],
        "Status": ["2025-01-15", "Credit+Market", "2", "✅ 0", "✅ Passed", "🟢 86%"],
    }
    st.dataframe(pd.DataFrame(exec_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🇺🇸 EN", use_container_width=True, type="primary" if st.session_state.lang=="en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    with c2:
        if st.button("🇦🇷 ES", use_container_width=True, type="primary" if st.session_state.lang=="es" else "secondary"):
            st.session_state.lang = "es"
            st.rerun()
    
    st.divider()
    page = st.radio(t("nav"), t("pages"), label_visibility="collapsed")
    st.divider()
    st.caption("AI Governance Framework v1.0 · 2025")

PAGES = {
    t("pages")[0]: page_mrr,
    t("pages")[1]: page_policy,
    t("pages")[2]: page_credit,
    t("pages")[3]: page_market,
    t("pages")[4]: page_comparative,
    t("pages")[5]: page_fairness,
    t("pages")[6]: page_stress,
    t("pages")[7]: page_euai,
    t("pages")[8]: page_audit,
    t("pages")[9]: page_reports,
}

PAGES.get(page, page_mrr)()
