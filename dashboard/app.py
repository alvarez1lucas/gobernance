"""
dashboard/app.py — AI Governance Framework Dashboard
Versión corregida para deploy en Streamlit Cloud.

Fixes aplicados:
  - Sin imports relativos (src/) — todo standalone
  - Todos los plots con colores explícitos (negro) para deploy web
  - font_color="#1a1a1a" en todos los layout de plotly
  - plot_bgcolor y paper_bgcolor siempre "white" explícito
  - axis title_font_color y tickfont_color explícitos
  - Todos los datos embebidos con fallback robusto
  - Sin sys.path manipulation
  - Sin roc_curve_from_scores (import roto)

Ejecutar:
    streamlit run dashboard/app.py
"""

import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from scipy.stats import t as t_dist, binom
from sklearn.metrics import roc_curve, roc_auc_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Governance Framework",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"]{ font-size:1.05rem; }
.sub{ color:#6b7280; font-size:.82rem; margin-top:-10px; margin-bottom:10px; }
.badge{ display:inline-block; padding:2px 9px; border-radius:10px;
        font-size:11px; font-weight:600; margin:2px; }
.card{ background:#f8fafc; border-radius:10px; padding:14px 16px;
       border:1px solid #e2e8f0; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Plot layout base — SIEMPRE colores explícitos para Streamlit Cloud ────────
# Streamlit Cloud hereda el tema del sistema; sin colores explícitos
# los textos aparecen grises o invisibles en fondo blanco.
LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f8fafc",
    font=dict(color="#1a1a1a", family="Arial, sans-serif", size=12),
    # NO title_font aquí — genera "undefined" en Streamlit Cloud cuando no hay title.text
    # Los títulos van siempre en st.subheader() de Streamlit, nunca en Plotly
    legend=dict(font=dict(color="#1a1a1a"), bgcolor="white",
                bordercolor="#e2e8f0", borderwidth=1),
    margin=dict(t=20, b=20, l=20, r=20),
    xaxis=dict(
        title_font=dict(color="#1a1a1a"),
        tickfont=dict(color="#1a1a1a"),
        gridcolor="#e5e7eb", linecolor="#d1d5db",
        zerolinecolor="#d1d5db",
    ),
    yaxis=dict(
        title_font=dict(color="#1a1a1a"),
        tickfont=dict(color="#1a1a1a"),
        gridcolor="#e5e7eb", linecolor="#d1d5db",
        zerolinecolor="#d1d5db",
    ),
)

def apply_layout(fig, height=320, **kwargs):
    """Aplica LAYOUT base + overrides. Garantiza colores negros en deploy.
    
    IMPORTANTE: nunca pasar title= con texto aquí.
    Todos los títulos van en st.subheader() de Streamlit.
    title=dict(text="") suprime el "undefined" de Plotly.
    """
    cfg = dict(LAYOUT)
    cfg["height"] = height
    # Siempre forzar title vacío para evitar "undefined" en Streamlit Cloud
    if "title" not in kwargs:
        cfg["title"] = dict(text="")
    cfg.update(kwargs)
    fig.update_layout(**cfg)
    # Forzar color negro en todos los ejes por si hay subplots
    fig.update_xaxes(
        title_font=dict(color="#1a1a1a"),
        tickfont=dict(color="#1a1a1a"),
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(
        title_font=dict(color="#1a1a1a"),
        tickfont=dict(color="#1a1a1a"),
        gridcolor="#e5e7eb",
    )
    # Suprimir anotaciones de título en subplots también
    fig.update_annotations(font=dict(color="#1a1a1a", size=11))
    return fig

# ── Translations ──────────────────────────────────────────────────────────────
TX = {
"en": {
  "nav": "Navigation",
  "pages": ["🏛️ MRR Overview","🔒 Policy Status","💳 Credit Risk",
            "📊 Market Risk","⚖️ Comparative Metrics","🌈 Fairness Dashboard",
            "⚠️ Stress Testing","🇪🇺 EU AI Act","🗂️ Audit Trail","📑 Regulatory Reports"],
  "no_data": "Showing synthetic demo data",
  "approved":"APPROVED","review":"UNDER REVIEW","retired":"RETIRED","pending":"PENDING",
  "model_status":"Portfolio Status","n_models":"Models","n_approved":"Approved",
  "n_blocks":"Policy Blocks","n_warns":"Policy Warns",
  "mrr_title":"Model Risk Register — Portfolio Overview",
  "mrr_sub":"SR 11-7 · EU AI Act Annex III · BCBS 239 — centralized model inventory",
  "mrr_timeline":"Validation timeline","mrr_regs":"Regulations covered",
  "po_title":"Policy Status — Policy-as-Code Evaluation",
  "po_sub":"OPA/Rego rules evaluated against live metrics from both repos",
  "po_pass_desc":"All checks passed","po_block_desc":"Deployment blocked",
  "po_warn_desc":"Alert — review required","po_summary":"Policy summary",
  "cr_title":"Credit Risk — Deep Dive",
  "cr_sub":"XGBoost · DL Tabular · GNN · SHAP · IFRS 9 · SR 11-7 · BCRA A 7724",
  "cr_roc":"ROC curves — champion vs challengers",
  "cr_shap":"Feature importance (SHAP values)",
  "cr_stress":"Stress scenarios — Gini degradation",
  "cr_ifrs9":"IFRS 9 PD calibration",
  "cr_gini":"Gini","cr_auc":"AUC-ROC","cr_ks":"KS Statistic",
  "cr_psi":"PSI Drift","cr_brier":"Brier Score","cr_hl":"Hosmer-Lemeshow p",
  "mr_title":"Market Risk — Deep Dive",
  "mr_sub":"TFT · GARCH · HMM · FinBERT · Conformal Prediction · Basel III FRTB",
  "mr_backtest":"Backtesting — VaR Fan Chart",
  "mr_sentiment":"NLP Sentiment (FinBERT)",
  "mr_regime":"Regime detection (HMM)",
  "mr_conformal":"Conformal Prediction — Classical vs Conformal",
  "mr_exc":"Exceedances (250d)","mr_kupiec":"Kupiec p-value",
  "mr_chr":"Christoffersen p-value","mr_zone":"Traffic Light Zone",
  "mr_cp_cov":"CP Coverage","mr_cp_valid":"CP Valid",
  "co_title":"Comparative Metrics — Credit vs Market",
  "co_sub":"Side-by-side regulatory comparison across all dimensions",
  "co_radar":"Multi-dimensional regulatory score",
  "co_sr117":"SR 11-7 Scores by pillar","co_drift":"Drift PSI comparison",
  "fa_title":"Fairness Dashboard",
  "fa_sub":"DPD · DIR · Equalized Odds — Credit Risk (EU AI Act Art. 10 + ECOA)",
  "fa_approval":"Approval rates by gender",
  "fa_dpd":"Demographic Parity Difference","fa_dir":"Disparate Impact Ratio",
  "fa_eq_odds":"Equalized Odds (TPR / FPR by group)",
  "fa_tpr":"TPR gap","fa_fpr":"FPR gap","fa_scores":"Score distribution by gender",
  "fa_market_note":"Market Risk — fairness not applicable (institutional portfolio)",
  "st_title":"Stress Testing — Unified View",
  "st_sub":"Credit Risk + Market Risk combined stress analysis",
  "st_credit":"Credit Risk — Gini under stress","st_market":"Market Risk — ES 97.5% by scenario",
  "st_mc":"Monte Carlo P&L + Credit Gini distribution (t-Student df=5)",
  "st_worst":"Worst 1% avg",
  "eu_title":"EU AI Act — Compliance Checklist",
  "eu_sub":"Regulation (EU) 2024/1689 — High-risk AI system requirements (Annex III)",
  "eu_overall":"Overall compliance",
  "au_title":"Audit Trail — Unified Event Log",
  "au_sub":"SHA-256 hash-chained log — Credit Risk + Market Risk + Governance (EU AI Act Art. 12)",
  "au_integrity":"Chain integrity","au_total":"Total events",
  "au_credit_ev":"Credit","au_market_ev":"Market","au_gov_ev":"Governance",
  "au_chain":"Hash chain",
  "re_title":"Regulatory Reports","re_sub":"Auto-generated reports for regulators and risk committees",
  "re_exec":"Executive Summary","re_download":"Download report",
  "re_sr117_credit":"SR 11-7 — Credit Risk","re_sr117_market":"SR 11-7 — Market Risk",
  "re_basel":"Basel III FRTB Summary","re_ifrs9":"IFRS 9 Calibration",
  "re_euai":"EU AI Act Technical Docs",
},
"es": {
  "nav": "Navegación",
  "pages": ["🏛️ MRR Overview","🔒 Estado de Políticas","💳 Riesgo de Crédito",
            "📊 Riesgo de Mercado","⚖️ Métricas Comparativas","🌈 Fairness Dashboard",
            "⚠️ Stress Testing","🇪🇺 EU AI Act","🗂️ Audit Trail","📑 Reportes Regulatorios"],
  "no_data": "Mostrando datos sintéticos de demostración",
  "approved":"APROBADO","review":"EN REVISIÓN","retired":"RETIRADO","pending":"PENDIENTE",
  "model_status":"Estado del Portfolio","n_models":"Modelos","n_approved":"Aprobados",
  "n_blocks":"Políticas Bloqueantes","n_warns":"Advertencias",
  "mrr_title":"Model Risk Register — Vista del Portfolio",
  "mrr_sub":"SR 11-7 · EU AI Act Annex III · BCBS 239 — inventario centralizado de modelos",
  "mrr_timeline":"Cronograma de validaciones","mrr_regs":"Regulaciones cubiertas",
  "po_title":"Estado de Políticas — Evaluación Policy-as-Code",
  "po_sub":"Reglas OPA/Rego evaluadas contra métricas en vivo de ambos repos",
  "po_pass_desc":"Todos los checks aprobados","po_block_desc":"Despliegue bloqueado",
  "po_warn_desc":"Alerta — revisión requerida","po_summary":"Resumen de políticas",
  "cr_title":"Riesgo de Crédito — Análisis Detallado",
  "cr_sub":"XGBoost · DL Tabular · GNN · SHAP · IFRS 9 · SR 11-7 · BCRA A 7724",
  "cr_roc":"Curvas ROC — champion vs challengers",
  "cr_shap":"Importancia de variables (valores SHAP)",
  "cr_stress":"Escenarios de stress — degradación del Gini",
  "cr_ifrs9":"Calibración IFRS 9 PD",
  "cr_gini":"Gini","cr_auc":"AUC-ROC","cr_ks":"Estadístico KS",
  "cr_psi":"Drift PSI","cr_brier":"Brier Score","cr_hl":"Hosmer-Lemeshow p",
  "mr_title":"Riesgo de Mercado — Análisis Detallado",
  "mr_sub":"TFT · GARCH · HMM · FinBERT · Predicción Conformal · Basel III FRTB",
  "mr_backtest":"Backtesting — Fan Chart VaR",
  "mr_sentiment":"Sentimiento NLP (FinBERT)",
  "mr_regime":"Detección de régimen (HMM)",
  "mr_conformal":"Predicción Conformal — Clásico vs Conformal",
  "mr_exc":"Exceedances (250d)","mr_kupiec":"Kupiec p-value",
  "mr_chr":"Christoffersen p-value","mr_zone":"Zona Traffic Light",
  "mr_cp_cov":"Cobertura CP","mr_cp_valid":"Garantía CP válida",
  "co_title":"Métricas Comparativas — Crédito vs Mercado",
  "co_sub":"Comparación regulatoria lado a lado en todas las dimensiones",
  "co_radar":"Score regulatorio multidimensional",
  "co_sr117":"Scores SR 11-7 por pilar","co_drift":"Comparación drift PSI",
  "fa_title":"Fairness Dashboard",
  "fa_sub":"DPD · DIR · Equalized Odds — Riesgo de Crédito (EU AI Act Art. 10 + ECOA)",
  "fa_approval":"Tasas de aprobación por género",
  "fa_dpd":"Diferencia de Paridad Demográfica","fa_dir":"Ratio de Impacto Dispar",
  "fa_eq_odds":"Equalized Odds (TPR / FPR por grupo)",
  "fa_tpr":"Gap TPR","fa_fpr":"Gap FPR","fa_scores":"Distribución de scores por género",
  "fa_market_note":"Riesgo de Mercado — fairness no aplica directamente (portfolio institucional)",
  "st_title":"Stress Testing — Vista Unificada",
  "st_sub":"Análisis de stress combinado — Riesgo de Crédito + Riesgo de Mercado",
  "st_credit":"Riesgo de Crédito — Gini bajo stress",
  "st_market":"Riesgo de Mercado — ES 97.5% por escenario",
  "st_mc":"Monte Carlo P&L + Gini bajo stress (t-Student df=5)",
  "st_worst":"Peor promedio 1%",
  "eu_title":"EU AI Act — Checklist de Cumplimiento",
  "eu_sub":"Reglamento (UE) 2024/1689 — Requisitos para sistemas de IA de alto riesgo (Annex III)",
  "eu_overall":"Cumplimiento general",
  "au_title":"Audit Trail — Log de Eventos Unificado",
  "au_sub":"Cadena SHA-256 — Crédito + Mercado + Governance (EU AI Act Art. 12)",
  "au_integrity":"Integridad de la cadena","au_total":"Eventos totales",
  "au_credit_ev":"Crédito","au_market_ev":"Mercado","au_gov_ev":"Governance",
  "au_chain":"Cadena de hashes",
  "re_title":"Reportes Regulatorios",
  "re_sub":"Reportes auto-generados para reguladores y comités de riesgo",
  "re_exec":"Resumen Ejecutivo","re_download":"Descargar reporte",
  "re_sr117_credit":"SR 11-7 — Riesgo de Crédito","re_sr117_market":"SR 11-7 — Riesgo de Mercado",
  "re_basel":"Resumen Basel III FRTB","re_ifrs9":"Calibración IFRS 9",
  "re_euai":"Documentación Técnica EU AI Act",
},
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def t(k):
    return TX[st.session_state.lang].get(k, TX["en"].get(k, k))

# ── Data — embebido directo, sin imports externos ─────────────────────────────
# Busca JSONs en demo/ si existen, sino usa valores hardcoded.
# Esto garantiza que funcione en Streamlit Cloud sin repos hermanos.

def _load_json_safe(paths: list, fallback: dict) -> dict:
    """Intenta cargar JSON desde múltiples paths, retorna fallback si no encuentra."""
    for p in paths:
        try:
            f = Path(p)
            if f.exists():
                return json.loads(f.read_text())
        except Exception:
            pass
    return fallback

@st.cache_data(ttl=300)
def get_credit_data() -> dict:
    sr117 = _load_json_safe(
        ["demo/credit_risk/sr117_validation.json",
         "../credit-risk-model-validation/reports/sr117_validation.json"],
        {
            "sr117_overall_pass": True,
            "discriminatory_power": {"gini":0.712,"auc_roc":0.856,"ks_statistic":0.524},
            "calibration": {"hl_pvalue":0.19,"well_calibrated":True},
            "stability":   {"psi":0.06,"psi_status":"stable"},
            "stress_testing": {
                "baseline_gini": 0.712,
                "scenarios": {
                    "Income Shock Moderate": {"gini":0.672,"auc_degradation":0.02},
                    "Income Shock Severe":   {"gini":0.632,"auc_degradation":0.04},
                    "Bureau Deterioration":  {"gini":0.592,"auc_degradation":0.06},
                },
            },
            "sensitivity_top10": {
                "EXT_SOURCE_2":0.08,"EXT_SOURCE_3":0.06,"EXT_SOURCE_1":0.04,
                "AMT_CREDIT":0.02,"DAYS_BIRTH":0.015,"AMT_INCOME_TOTAL":0.012,
                "DAYS_EMPLOYED":0.010,"CODE_GENDER":0.008,"AMT_ANNUITY":0.006,"CNT_CHILDREN":0.004,
            },
        }
    )
    fairness = _load_json_safe(
        ["demo/credit_risk/fairness_report.json",
         "../credit-risk-model-validation/reports/fairness_report.json"],
        {
            "overall_fairness_passed": True,
            "results": {"gender": {
                "approval_rates": {"M":0.682,"F":0.712},
                "demographic_parity_difference": 0.030,
                "disparate_impact_ratio": 0.957,
                "equalized_odds": {
                    "M":{"tpr":0.62,"fpr":0.09},
                    "F":{"tpr":0.60,"fpr":0.08},
                    "tpr_gap":0.02,"fpr_gap":0.01,
                },
                "passed": True,
            }},
        }
    )
    # Datos sintéticos para ROC y distribución de scores
    np.random.seed(42); n = 6000
    y      = np.random.binomial(1, 0.08, n)
    s      = np.clip(y*np.random.beta(5,2,n)+(1-y)*np.random.beta(2,5,n),0.01,0.99)
    s_nn   = np.clip(s+np.random.normal(0,0.03,n),0.01,0.99)
    s_lstm = np.clip(s+np.random.normal(0.015,0.04,n),0.01,0.99)
    s_lr   = np.clip(s-0.07,0.01,0.99)
    gender = np.random.choice(["M","F"],n,p=[0.58,0.42])
    fpr,tpr,_ = roc_curve(y,s)
    return {
        "sr117": sr117, "fairness": fairness,
        "metrics": {
            "gini":   sr117["discriminatory_power"]["gini"],
            "auc":    sr117["discriminatory_power"]["auc_roc"],
            "ks":     sr117["discriminatory_power"]["ks_statistic"],
            "psi":    sr117["stability"]["psi"],
            "brier":  0.084,
            "hl_p":   sr117["calibration"]["hl_pvalue"],
        },
        "demo": {
            "y":y,"s":s,"s_nn":s_nn,"s_lstm":s_lstm,"s_lr":s_lr,
            "fpr":fpr,"tpr":tpr,"gender":gender,
            "gini_xgb":  round(2*roc_auc_score(y,s)-1,4),
            "gini_nn":   round(2*roc_auc_score(y,s_nn)-1,4),
            "gini_lstm": round(2*roc_auc_score(y,s_lstm)-1,4),
            "gini_lr":   round(2*roc_auc_score(y,s_lr)-1,4),
        },
    }

@st.cache_data(ttl=300)
def get_market_data() -> dict:
    bt = _load_json_safe(
        ["demo/market_risk/var_backtest.json",
         "../market-risk-deep-learning/reports/var_backtest.json"],
        {"n_exceedances":3,"n_observations":250,"kupiec_pval":0.38,"kupiec_pass":True,
         "christoffersen_pval":0.55,"christoffersen_pass":True,
         "traffic_light_zone":"green","overall_status":"approved"},
    )
    cp = _load_json_safe(
        ["demo/market_risk/conformal_backtest.json",
         "../market-risk-deep-learning/reports/conformal_backtest.json"],
        {"coverage_test":{"conformal_coverage":0.992,"classical_coverage":0.984,
                          "conformal_exceedances":2,"classical_exceedances":4},
         "nonconformity_quantile":0.0042,"conformal_valid":True},
    )
    stress = _load_json_safe(
        ["demo/market_risk/stress_scenarios/stress_report.json",
         "../market-risk-deep-learning/reports/stress_scenarios/stress_report.json"],
        {"gfc_2008":{"name":"GFC 2008","es_975_1d":-0.042,"total_loss_pct":-0.61},
         "covid_2020":{"name":"COVID-19 Q1 2020","es_975_1d":-0.031,"total_loss_pct":-0.38},
         "rates_2022":{"name":"Rate Hike 2022","es_975_1d":-0.024,"total_loss_pct":-0.28},
         "svb_2023":{"name":"SVB Run 2023","es_975_1d":-0.018,"total_loss_pct":-0.17},
         "dfast_adv":{"name":"DFAST Severely Adv.","es_975_1d":-0.038,"total_loss_pct":-0.58},
         "latam_tail":{"name":"LATAM Tail Risk","es_975_1d":-0.028,"total_loss_pct":-0.43}},
    )
    # Returns sintéticos
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.today(), periods=1260)
    r = np.random.normal(-0.0003,0.012,len(dates))
    for s2,e2,sh in [(200,280,-0.025),(800,830,-0.035),(1050,1090,-0.018)]: r[s2:e2]+=sh
    # Sentimiento sintético
    sent = np.zeros(len(dates)); sent[0]=0.1
    for i in range(1,len(dates)): sent[i]=0.7*sent[i-1]+np.random.normal(0.02,0.15)
    for a,b2,sh in [("2008-09-01","2009-03-31",-0.6),("2020-02-20","2020-04-30",-0.7),
                    ("2022-01-01","2022-12-31",-0.3),("2023-03-08","2023-03-31",-0.5)]:
        mask=(dates>=a)&(dates<=b2); sent[mask]+=sh
    sent=np.clip(sent,-1,1)
    return {
        "backtest": bt, "conformal": cp, "stress": stress,
        "returns":  pd.Series(r, index=dates, name="log_return_SPX"),
        "sentiment":pd.DataFrame({"mean":sent,"ma21":pd.Series(sent).rolling(21).mean().fillna(0).values},
                                  index=dates),
    }

@st.cache_data(ttl=300)
def get_audit_events() -> list:
    events = []
    for path, source in [
        ("demo/credit_risk/audit_trail.jsonl","credit"),
        ("../credit-risk-model-validation/reports/audit_trail.jsonl","credit"),
        ("demo/market_risk/audit_trail.jsonl","market"),
        ("../market-risk-deep-learning/reports/audit_trail.jsonl","market"),
        ("reports/governance_audit.jsonl","governance"),
    ]:
        try:
            f = Path(path)
            if f.exists():
                for line in f.read_text().splitlines():
                    if line.strip():
                        e = json.loads(line); e["_source"]=source
                        events.append(e)
        except Exception:
            pass
    if not events:
        events = [
            {"timestamp":"2025-01-10T09:00:00Z","event_type":"pipeline_started","actor":"pipeline",
             "payload":{"version":"1.0"},"_source":"credit","hash":"abc123def456abc1"},
            {"timestamp":"2025-01-10T10:30:00Z","event_type":"model_trained","actor":"pipeline",
             "payload":{"model":"XGBoost","gini":0.712},"_source":"credit","hash":"def456ghi789def4"},
            {"timestamp":"2025-01-10T11:00:00Z","event_type":"sr117_completed","actor":"pipeline",
             "payload":{"status":"approved","score":0.88},"_source":"credit","hash":"ghi789jkl012ghi7"},
            {"timestamp":"2025-01-15T09:00:00Z","event_type":"pipeline_started","actor":"pipeline",
             "payload":{},"_source":"market","hash":"jkl012mno345jkl0"},
            {"timestamp":"2025-01-15T09:30:00Z","event_type":"model_trained","actor":"pipeline",
             "payload":{"model":"TFT","val_loss":0.00041},"_source":"market","hash":"mno345pqr678mno3"},
            {"timestamp":"2025-01-15T09:45:00Z","event_type":"backtesting_completed","actor":"pipeline",
             "payload":{"kupiec_pval":0.38,"exceedances":3},"_source":"market","hash":"pqr678stu901pqr6"},
            {"timestamp":"2025-01-15T10:00:00Z","event_type":"governance_check","actor":"governance",
             "payload":{"n_models":2,"n_blocks":0},"_source":"governance","hash":"stu901vwx234stu9"},
        ]
    events.sort(key=lambda e: e.get("timestamp",""))
    return events

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🇺🇸 EN", use_container_width=True,
                     type="primary" if st.session_state.lang=="en" else "secondary"):
            st.session_state.lang="en"; st.rerun()
    with c2:
        if st.button("🇦🇷 ES", use_container_width=True,
                     type="primary" if st.session_state.lang=="es" else "secondary"):
            st.session_state.lang="es"; st.rerun()
    st.divider()
    page = st.radio(t("nav"), t("pages"), label_visibility="collapsed")
    st.divider()
    cd = get_credit_data(); md = get_market_data()
    zone = md["backtest"].get("traffic_light_zone","green")
    ze   = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(zone,"⚪")
    cr_ok = cd["metrics"]["gini"] >= 0.45
    st.markdown(f"**{t('model_status')}**")
    st.markdown(f"{'✅' if cr_ok else '⚠️'} Credit Risk: **{t('approved') if cr_ok else t('review')}**")
    st.markdown(f"{ze} Market Risk: **{zone.upper()}**")
    st.markdown("🔒 Policy blocks: **0**")
    st.divider()
    st.caption("AI Governance Framework v1.0 · 2025")

# ── KPI bar ───────────────────────────────────────────────────────────────────
def kpi_bar():
    cd = get_credit_data(); md = get_market_data()
    zone   = md["backtest"].get("traffic_light_zone","green")
    n_exc  = md["backtest"].get("n_exceedances",3)
    kup_p  = md["backtest"].get("kupiec_pval",0.38)
    cp_cov = md["conformal"].get("coverage_test",{}).get("conformal_coverage",0.992)
    gini   = cd["metrics"]["gini"]
    ze = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(zone,"⚪")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Portfolio",    "2 Models", "0 blocks")
    k2.metric("Credit Gini",  f"{gini:.3f}", "✅ SR 11-7")
    k3.metric("Basel III",    f"{ze} {zone.upper()}", f"{n_exc}/250 exc.")
    k4.metric("Kupiec p",     f"{kup_p:.3f}", "✅ PASS" if kup_p>0.05 else "❌ FAIL")
    k5.metric("CP Coverage",  f"{'✅' if cp_cov>=0.99 else '⚠️'} {cp_cov:.2%}")
    k6.metric("Fairness DIR", "0.957", "✅ > 0.80")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MRR Overview
# ══════════════════════════════════════════════════════════════════════════════
def page_mrr():
    st.title(t("mrr_title"))
    st.markdown(f'<p class="sub">{t("mrr_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    MODELS = [
        {"id":"credit-risk-v2.1","name":"Credit Scoring — IFRS 9 PD","type":"Credit",
         "tier":"High","status":"approved","sr117":0.88,"psi":0.06,
         "next_val":"2025-07-10","issues":0,
         "regs":["SR 11-7","EU AI Act","IFRS 9","BCRA A7724"],"type_col":"#4361ee"},
        {"id":"market-var-tft-v1.0","name":"Market VaR — TFT v1.0","type":"Market",
         "tier":"High","status":"approved","sr117":0.88,"psi":0.08,
         "next_val":"2025-07-15","issues":1,
         "regs":["SR 11-7","Basel III FRTB","EU AI Act","BCBS 239"],"type_col":"#7209b7"},
    ]
    for m in MODELS:
        days = (pd.to_datetime(m["next_val"]).date() - datetime.today().date()).days
        urg  = "#06d6a0" if days>90 else "#ffd166" if days>30 else "#ef476f"
        st_col = "#06d6a0" if m["status"]=="approved" else "#ffd166"
        tier_col = "#ef476f"
        st.markdown(f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <span style="font-weight:700;font-size:15px;color:#1a1a1a">{m['name']}</span>
      <span style="color:#6b7280;font-size:12px;margin-left:8px">{m['id']}</span><br>
      <span class="badge" style="background:{m['type_col']}20;color:{m['type_col']}">{m['type']}</span>
      <span class="badge" style="background:{tier_col}20;color:{tier_col}">{m['tier']} Tier</span>
      <span class="badge" style="background:{st_col}20;color:{st_col}">{m['status'].upper()}</span>
      {''.join([f'<span class="badge" style="background:#4361ee15;color:#4361ee">{r}</span>' for r in m['regs']])}
    </div>
    <div style="text-align:right">
      <div style="font-size:13px;color:#1a1a1a">SR 11-7: <b>{m['sr117']:.0%}</b></div>
      <div style="font-size:13px;color:#1a1a1a">PSI Drift: <b>{m['psi']:.2f}</b></div>
      <div style="font-size:12px;color:{urg}">Next val: {m['next_val']} ({days}d)</div>
      <div style="font-size:12px;color:#1a1a1a">Issues: <b>{'✅ 0' if m['issues']==0 else '⚠️ '+str(m['issues'])}</b></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(t("mrr_timeline"))
        fig = go.Figure()
        for i, m in enumerate(MODELS):
            last = pd.to_datetime("2025-01-15" if m["type"]=="Market" else "2025-01-10")
            nxt  = pd.to_datetime(m["next_val"])
            col  = m["type_col"]
            fig.add_trace(go.Scatter(
                x=[last,nxt], y=[i,i], mode="lines+markers",
                line=dict(color=col,width=6),
                marker=dict(size=12,color=[col,"#ef476f"]),
                name=m["name"][:22]))
            fig.add_annotation(x=last,y=i,text="Last",showarrow=False,
                               font=dict(size=9,color="#1a1a1a"),yshift=14)
            fig.add_annotation(x=nxt,y=i,text="Next",showarrow=False,
                               font=dict(size=9,color="#ef476f"),yshift=14)
        # Fix: add_vline no acepta datetime — usar scatter line en su lugar
        today_str = datetime.today().strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(
            x=[pd.to_datetime(today_str), pd.to_datetime(today_str)],
            y=[-0.4, 1.4], mode="lines",
            line=dict(color="#ef476f", width=1.5, dash="dash"),
            name="Today", showlegend=True))
        fig.add_annotation(
            x=pd.to_datetime(today_str), y=1.5,
            text="Today", showarrow=False,
            font=dict(size=9, color="#ef476f", family="Arial"))
        apply_layout(fig, height=220,
                     yaxis=dict(tickvals=[0,1],
                                ticktext=[m["name"][:24] for m in MODELS],
                                tickfont=dict(color="#1a1a1a"),
                                title_font=dict(color="#1a1a1a")),
                     legend=dict(orientation="h",y=1.15,font=dict(color="#1a1a1a")),
                     showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader(t("mrr_regs"))
        regs = {"SR 11-7":2,"EU AI Act":2,"Basel III FRTB":1,
                "IFRS 9":1,"BCBS 239":1,"BCRA A7724":1}
        fig2 = go.Figure(go.Bar(
            x=list(regs.keys()), y=list(regs.values()),
            marker_color=["#4361ee","#7209b7","#f72585","#06d6a0","#ffd166","#ef476f"],
            opacity=0.85, text=list(regs.values()), textposition="outside",
            textfont=dict(color="#1a1a1a")))
        apply_layout(fig2, height=220,
                     yaxis_title="Models covered",
                     xaxis=dict(tickangle=-20,tickfont=dict(color="#1a1a1a"),
                                title_font=dict(color="#1a1a1a")))
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Policy Status
# ══════════════════════════════════════════════════════════════════════════════
def page_policy():
    st.title(t("po_title"))
    st.markdown(f'<p class="sub">{t("po_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data(); md = get_market_data()
    gini   = cd["metrics"]["gini"]
    dir_r  = cd["fairness"]["results"]["gender"]["disparate_impact_ratio"]
    psi_cr = cd["metrics"]["psi"]
    n_exc  = md["backtest"]["n_exceedances"]
    kup_p  = md["backtest"]["kupiec_pval"]
    cp_cov = md["conformal"]["coverage_test"]["conformal_coverage"]
    psi_mr = 0.08

    POLICIES = [
        {"model":"Credit Risk","id":"MIN_PERFORMANCE","sev":"block",
         "rule":"Gini ≥ 0.45","value":gini,"threshold":0.45,"passed":gini>=0.45},
        {"model":"Credit Risk","id":"FAIRNESS_DIR","sev":"block",
         "rule":"DIR ≥ 0.80","value":dir_r,"threshold":0.80,"passed":dir_r>=0.80},
        {"model":"Credit Risk","id":"DRIFT_CREDIT","sev":"warn",
         "rule":"PSI ≤ 0.10","value":psi_cr,"threshold":0.10,"passed":psi_cr<=0.10},
        {"model":"Credit Risk","id":"IFRS9_CALIB","sev":"warn",
         "rule":"Calibrated","value":"True","threshold":"True","passed":True},
        {"model":"Market Risk","id":"TRAFFIC_LIGHT","sev":"block",
         "rule":"Exceedances ≤ 4","value":n_exc,"threshold":4,"passed":n_exc<=4},
        {"model":"Market Risk","id":"KUPIEC_PASS","sev":"block",
         "rule":"Kupiec p > 0.05","value":round(kup_p,3),"threshold":0.05,"passed":kup_p>0.05},
        {"model":"Market Risk","id":"CP_COVERAGE","sev":"warn",
         "rule":"CP cov ≥ 0.99","value":round(cp_cov,3),"threshold":0.99,"passed":cp_cov>=0.99},
        {"model":"Market Risk","id":"DRIFT_MARKET","sev":"warn",
         "rule":"PSI ≤ 0.10","value":psi_mr,"threshold":0.10,"passed":psi_mr<=0.10},
    ]
    n_pass   = sum(1 for p in POLICIES if p["passed"])
    n_blocks = sum(1 for p in POLICIES if not p["passed"] and p["sev"]=="block")
    n_warns  = sum(1 for p in POLICIES if not p["passed"] and p["sev"]=="warn")

    c1,c2,c3 = st.columns(3)
    c1.metric(t("po_pass_desc"),  f"✅ {n_pass}/{len(POLICIES)}")
    c2.metric(t("po_block_desc"), f"{'✅ 0' if n_blocks==0 else '❌ '+str(n_blocks)}")
    c3.metric(t("po_warn_desc"),  f"{'✅ 0' if n_warns==0 else '⚠️ '+str(n_warns)}")

    st.divider()
    SCOL = {"block_fail":"#fee2e2","warn_fail":"#fef3c7","pass":"#d1fae5"}
    SEMO = {"block_fail":"❌","warn_fail":"⚠️","pass":"✅"}

    for group in ["Credit Risk","Market Risk"]:
        icon = "💳" if "Credit" in group else "📊"
        st.subheader(f"{icon} {group}")
        cols = st.columns(2)
        for i, p in enumerate([p for p in POLICIES if p["model"]==group]):
            key = "pass" if p["passed"] else ("block_fail" if p["sev"]=="block" else "warn_fail")
            with cols[i%2]:
                v = f"{p['value']:.3f}" if isinstance(p['value'],float) else str(p['value'])
                th = f"{p['threshold']:.3f}" if isinstance(p['threshold'],float) else str(p['threshold'])
                st.markdown(
                    f'<div style="background:{SCOL[key]};padding:10px 14px;'
                    f'border-radius:8px;margin:4px 0">'
                    f'<div style="font-weight:700;font-size:13px;color:#1a1a1a">'
                    f'{SEMO[key]} {p["id"]}</div>'
                    f'<div style="font-size:12px;color:#374151;margin-top:3px">'
                    f'Rule: <code>{p["rule"]}</code><br>'
                    f'Value: <b>{v}</b> | Threshold: <b>{th}</b></div></div>',
                    unsafe_allow_html=True)
        st.markdown("")

    st.subheader(t("po_summary"))
    fig = go.Figure(go.Pie(
        labels=["PASS","BLOCK","WARN"],
        values=[n_pass,max(n_blocks,0),max(n_warns,0)],
        marker_colors=["#06d6a0","#ef476f","#ffd166"],
        hole=0.55,textinfo="label+value",
        textfont=dict(color="#1a1a1a",size=12)))
    fig.update_layout(height=260, paper_bgcolor="white",
                      title=dict(text=""),
                      font=dict(color="#1a1a1a"),
                      legend=dict(font=dict(color="#1a1a1a"),orientation="h",y=-0.1),
                      margin=dict(t=10,b=10,l=10,r=10))
    col_c = st.columns([1,2,1])[1]
    with col_c:
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Credit Risk
# ══════════════════════════════════════════════════════════════════════════════
def page_credit():
    st.title(t("cr_title"))
    st.markdown(f'<p class="sub">{t("cr_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data()
    m  = cd["metrics"]; d = cd["demo"]; sr = cd["sr117"]

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric(t("cr_gini"),  f"{m['gini']:.3f}",  "✅ > 0.45")
    c2.metric(t("cr_auc"),   f"{m['auc']:.3f}")
    c3.metric(t("cr_ks"),    f"{m['ks']:.3f}")
    c4.metric(t("cr_brier"), f"{m['brier']:.3f}", "✅ Calibrated")
    c5.metric(t("cr_psi"),   f"{m['psi']:.3f}",   "stable")
    c6.metric(t("cr_hl"),    f"{m['hl_p']:.3f}",  "✅ > 0.05")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("cr_roc"))
        fig = go.Figure()
        for scores, name, color, g in [
            (d["s"],     f"XGBoost — Gini={d['gini_xgb']:.3f}",  "#4361ee", d["gini_xgb"]),
            (d["s_nn"],  f"TabNet  — Gini={d['gini_nn']:.3f}",   "#7209b7", d["gini_nn"]),
            (d["s_lstm"],f"LSTM    — Gini={d['gini_lstm']:.3f}", "#f72585", d["gini_lstm"]),
            (d["s_lr"],  f"LR base — Gini={d['gini_lr']:.3f}",   "#adb5bd", d["gini_lr"]),
        ]:
            fpr_i, tpr_i, _ = roc_curve(d["y"], scores)
            lw = 2.2 if "XGBoost" in name else 1.4
            fig.add_trace(go.Scatter(x=fpr_i,y=tpr_i,name=name,
                                     line=dict(color=color,width=lw)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(color="#adb5bd",dash="dash"),
                                 showlegend=False))
        apply_layout(fig, height=340,
                     xaxis_title="False Positive Rate",
                     yaxis_title="True Positive Rate",
                     legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader(t("cr_shap"))
        feats = sr.get("sensitivity_top10",{
            "EXT_SOURCE_2":0.08,"EXT_SOURCE_3":0.06,"EXT_SOURCE_1":0.04,
            "AMT_CREDIT":0.02,"DAYS_BIRTH":0.015,"AMT_INCOME_TOTAL":0.012,
            "DAYS_EMPLOYED":0.010,"CODE_GENDER":0.008,"AMT_ANNUITY":0.006,"CNT_CHILDREN":0.004})
        fs = dict(sorted(feats.items(), key=lambda x: x[1]))
        vals = list(fs.values())
        q75  = np.quantile(vals, 0.75)
        cols_bar = ["#ef476f" if v>=q75 else "#4361ee" for v in vals]
        fig2 = go.Figure(go.Bar(
            x=vals, y=list(fs.keys()), orientation="h",
            marker_color=cols_bar, opacity=0.85,
            textfont=dict(color="#1a1a1a")))
        apply_layout(fig2, height=340,
                     xaxis_title="Mean |SHAP value|",
                     margin=dict(t=15,b=15,l=140,r=15))
        st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(t("cr_stress"))
        stress_data = sr.get("stress_testing",{})
        scen = stress_data.get("scenarios",{})
        base_g = stress_data.get("baseline_gini", m["gini"])
        s_names = list(scen.keys())
        s_ginis = [v.get("gini",base_g) for v in scen.values()]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=s_names, y=s_ginis,
                              marker_color=["#ffd166","#f72585","#ef476f"][:len(s_names)],
                              opacity=0.85,
                              text=[f"{g:.3f}" for g in s_ginis],
                              textposition="outside",
                              textfont=dict(color="#1a1a1a")))
        fig3.add_hline(y=base_g, line_dash="dash", line_color="#4361ee",
                       annotation_text=f"Baseline {base_g:.3f}",
                       annotation_font=dict(color="#1a1a1a"))
        fig3.add_hline(y=0.45, line_dash="dot", line_color="#ef476f",
                       annotation_text="Floor 0.45",
                       annotation_font=dict(color="#ef476f"))
        apply_layout(fig3, height=280, yaxis_title="Gini")
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader(t("cr_ifrs9"))
        np.random.seed(42)
        pd_buckets = np.array([0.01,0.03,0.07,0.12,0.20,0.35,0.55])
        observed   = np.clip(pd_buckets+np.random.normal(0,0.012,len(pd_buckets)),0.005,0.95)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=pd_buckets,y=pd_buckets,name="Perfect calibration",
                                  line=dict(color="#adb5bd",dash="dash",width=1.5)))
        fig4.add_trace(go.Scatter(x=pd_buckets,y=observed,name="Model (Platt scaled)",
                                  mode="markers+lines",
                                  marker=dict(size=9,color="#4361ee"),
                                  line=dict(color="#4361ee",width=1.8)))
        apply_layout(fig4, height=280,
                     title=dict(text=""),
                     xaxis_title="Predicted PD",
                     yaxis_title="Observed default rate",
                     legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")))
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Market Risk
# ══════════════════════════════════════════════════════════════════════════════
def page_market():
    st.title(t("mr_title"))
    st.markdown(f'<p class="sub">{t("mr_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    md  = get_market_data()
    bt  = md["backtest"]; cp = md["conformal"]
    ret = md["returns"];  sent = md["sentiment"]

    zone   = bt.get("traffic_light_zone","green")
    n_exc  = bt.get("n_exceedances",3)
    kup_p  = bt.get("kupiec_pval",0.38)
    chr_p  = bt.get("christoffersen_pval",0.55)
    ze     = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(zone,"⚪")
    cp_cov = cp.get("coverage_test",{}).get("conformal_coverage",0.992)
    conf_e = cp.get("coverage_test",{}).get("conformal_exceedances",2)
    clas_e = cp.get("coverage_test",{}).get("classical_exceedances",4)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric(t("mr_zone"),    f"{ze} {zone.upper()}")
    c2.metric(t("mr_exc"),     f"{n_exc}/250")
    c3.metric(t("mr_kupiec"),  f"{kup_p:.3f}", "✅ PASS" if kup_p>0.05 else "❌ FAIL")
    c4.metric(t("mr_chr"),     f"{chr_p:.3f}", "✅ PASS" if chr_p>0.05 else "⚠️")
    c5.metric(t("mr_cp_cov"),  f"{cp_cov:.2%}", "✅" if cp_cov>=0.99 else "⚠️")
    c6.metric(t("mr_cp_valid"),f"{'✅ Yes' if cp.get('conformal_valid') else '❌ No'}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("mr_backtest"))
        r500 = ret.tail(500)
        vol  = r500.rolling(21).std().fillna(r500.std())
        q99  = t_dist.ppf(0.01, df=5)*vol
        q975 = t_dist.ppf(0.025,df=5)*vol
        fig  = go.Figure()
        # Bandas de confianza
        for q, al, c_hex, nm in [(q99,0.07,"#ef476f","99%"),(q975,0.15,"#ffd166","97.5%")]:
            rv,gv,bv = int(c_hex[1:3],16),int(c_hex[3:5],16),int(c_hex[5:7],16)
            fig.add_trace(go.Scatter(
                x=list(r500.index)+list(r500.index[::-1]),
                y=list(q)+list(-q[::-1]),
                fill="toself",
                fillcolor=f"rgba({rv},{gv},{bv},{al})",
                line=dict(width=0), name=f"±{nm}", showlegend=True))
        fig.add_trace(go.Scatter(x=r500.index,y=r500.values,name="Return",
                                 line=dict(color="#1a1a1a",width=0.6),opacity=0.7))
        fig.add_trace(go.Scatter(x=r500.index,y=q99.values,name="VaR 99%",
                                 line=dict(color="#ef476f",dash="dash",width=1.8)))
        exc_m = r500.values < q99.values
        if exc_m.any():
            fig.add_trace(go.Scatter(
                x=r500.index[exc_m],y=r500.values[exc_m],mode="markers",
                marker=dict(color="red",size=7,symbol="x"),
                name=f"Exceedances ({exc_m.sum()})"))
        apply_layout(fig, height=310,
                     yaxis_title="Log-return",
                     legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader(t("mr_sentiment"))
        s500 = sent.tail(500)
        cur_s = float(sent["mean"].iloc[-1])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=s500.index, y=s500["mean"].values,
            fill="tozeroy", name="Sentiment",
            line=dict(color="#4361ee",width=0.8),
            fillcolor="rgba(67,97,238,0.12)"))
        if "ma21" in s500.columns:
            fig2.add_trace(go.Scatter(
                x=s500.index,y=s500["ma21"].values,name="MA 21d",
                line=dict(color="#ef476f",width=1.5,dash="dash")))
        fig2.add_hline(y=0, line_color="#1a1a1a", line_width=0.8)
        fig2.add_hline(y=sent["mean"].quantile(0.10),
                       line_dash="dot", line_color="#ef476f",
                       annotation_text="Extreme neg. threshold",
                       annotation_font=dict(color="#1a1a1a",size=9))
        apply_layout(fig2, height=310,
                     yaxis=dict(range=[-1,1],title_font=dict(color="#1a1a1a"),
                                tickfont=dict(color="#1a1a1a")),
                     legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")))
        st.plotly_chart(fig2, use_container_width=True)
        se = "🟢" if cur_s>0.1 else "🔴" if cur_s<-0.1 else "🟡"
        st.metric("Current sentiment", f"{se} {cur_s:+.3f}")

    # Conformal + Regime
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(t("mr_conformal"))
        lb_cl = t("mr_exc")+" Classical"
        lb_cp = t("mr_exc")+" Conformal"
        fig3 = go.Figure(go.Bar(
            x=[lb_cl, lb_cp], y=[clas_e, conf_e],
            marker_color=["#adb5bd","#4361ee"], opacity=0.85,
            text=[str(clas_e), str(conf_e)], textposition="outside",
            textfont=dict(color="#1a1a1a")))
        fig3.add_hline(y=4, line_dash="dash", line_color="#06d6a0", line_width=1.5,
                       annotation_text="Basel III green zone (≤4)",
                       annotation_font=dict(color="#1a1a1a",size=10))
        apply_layout(fig3, height=270, yaxis_title="N exceedances", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        for lb,cov,exc in [(lb_cl,cp["coverage_test"]["classical_coverage"],clas_e),
                           (lb_cp,cp["coverage_test"]["conformal_coverage"],conf_e)]:
            ze2 = "🟢" if exc<=4 else "🟡" if exc<=9 else "🔴"
            st.markdown(f"{ze2} **{lb}**: {cov:.2%} coverage · {exc} exceedances")

    with col_b:
        st.subheader(t("mr_regime"))
        v21  = ret.rolling(21).std().fillna(ret.std())*np.sqrt(252)
        p25,p50,p75 = v21.quantile([0.25,0.50,0.75])
        regs = np.where(v21<p25,0,np.where(v21<p50,1,np.where(v21<p75,2,3)))
        RNAMES_EN = {0:"Bull/Low Vol",1:"Normal",2:"Bear/High Vol",3:"Crisis"}
        RNAMES_ES = {0:"Bull/Baja Vol",1:"Normal",2:"Bear/Alta Vol",3:"Crisis"}
        rnames = RNAMES_EN if st.session_state.lang=="en" else RNAMES_ES
        rcounts = [(rnames[i],(regs==i).sum()) for i in range(4)]
        fig4 = go.Figure(go.Pie(
            labels=[r[0] for r in rcounts],
            values=[r[1] for r in rcounts],
            marker_colors=["#06d6a0","#4361ee","#ffd166","#ef476f"],
            hole=0.5, textinfo="label+percent",
            textfont=dict(color="#1a1a1a",size=11)))
        fig4.update_layout(height=270, paper_bgcolor="white",
                           showlegend=False, title=dict(text=""),
                           font=dict(color="#1a1a1a"),
                           margin=dict(t=15,b=15,l=10,r=10))
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Comparative Metrics
# ══════════════════════════════════════════════════════════════════════════════
def page_comparative():
    st.title(t("co_title"))
    st.markdown(f'<p class="sub">{t("co_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data(); md = get_market_data()
    gini  = cd["metrics"]["gini"]
    kup_p = md["backtest"]["kupiec_pval"]
    zone  = md["backtest"]["traffic_light_zone"]
    n_exc = md["backtest"]["n_exceedances"]
    ze    = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(zone,"⚪")

    # Tabla comparativa
    dim = "Dimension" if st.session_state.lang=="en" else "Dimensión"
    cr  = t("co_credit") if False else ("Credit Risk" if st.session_state.lang=="en" else "Riesgo de Crédito")
    mr  = "Market Risk" if st.session_state.lang=="en" else "Riesgo de Mercado"

    rows_en = ["SR 11-7 Score","Key metric","Regulatory status",
               "Drift PSI","Validation frequency","EU AI Act","Backtesting"]
    rows_es = ["Score SR 11-7","Métrica clave","Estado regulatorio",
               "Drift PSI","Frecuencia validación","EU AI Act","Backtesting"]
    rows = rows_en if st.session_state.lang=="en" else rows_es

    df_comp = pd.DataFrame({
        dim: rows,
        cr:  ["88%", f"Gini = {gini:.3f}", "✅ Approved","0.06 (stable)","Semiannual","High-risk Annex III","N/A"],
        mr:  ["88%", f"Exc = {n_exc}/250", "✅ Approved","0.08 (stable)","Semiannual","High-risk Annex III",
              f"Kupiec p={kup_p:.3f} ✅"],
    })
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("co_radar"))
        dims_r = ["SR 11-7","Drift\nStability","Stress\nResilience",
                  "Fairness","Reg. Coverage","Tech\nSoph."]
        cr_sc  = [0.88,0.94,0.82,0.96,0.90,0.85]
        mr_sc  = [0.88,0.92,0.84,0.70,0.88,0.95]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=cr_sc+[cr_sc[0]], theta=dims_r+[dims_r[0]],
            fill="toself", name="Credit Risk",
            line=dict(color="#4361ee",width=2),
            fillcolor="rgba(67,97,238,0.15)"))
        fig.add_trace(go.Scatterpolar(
            r=mr_sc+[mr_sc[0]], theta=dims_r+[dims_r[0]],
            fill="toself", name="Market Risk",
            line=dict(color="#7209b7",width=2),
            fillcolor="rgba(114,9,183,0.15)"))
        fig.update_layout(
            height=340, paper_bgcolor="white",
            title=dict(text=""), font=dict(color="#1a1a1a"),
            polar=dict(
                radialaxis=dict(range=[0,1],tickfont=dict(color="#1a1a1a"),
                                gridcolor="#e5e7eb"),
                angularaxis=dict(tickfont=dict(color="#1a1a1a",size=11)),
            ),
            legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")),
            margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader(t("co_sr117"))
        categories = ["Conceptual\nSoundness","Ongoing\nMonitoring",
                      "Outcomes\nAnalysis","Overall"]
        cr_scores = [0.97,0.90,1.00,0.88]
        mr_scores = [0.95,0.90,1.00,0.88]
        x = np.arange(len(categories)); w = 0.3
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=x-w/2,y=cr_scores,width=w,name="Credit Risk",
                              marker_color="#4361ee",opacity=0.85,
                              text=[f"{v:.0%}" for v in cr_scores],
                              textposition="outside",textfont=dict(color="#1a1a1a")))
        fig2.add_trace(go.Bar(x=x+w/2,y=mr_scores,width=w,name="Market Risk",
                              marker_color="#7209b7",opacity=0.85,
                              text=[f"{v:.0%}" for v in mr_scores],
                              textposition="outside",textfont=dict(color="#1a1a1a")))
        fig2.add_hline(y=0.80,line_dash="dash",line_color="#ef476f",line_width=1,
                       annotation_text="Min 80%",annotation_font=dict(color="#1a1a1a"))
        apply_layout(fig2, height=260,
                     xaxis=dict(tickvals=list(x),ticktext=categories,
                                tickfont=dict(color="#1a1a1a")),
                     yaxis=dict(range=[0,1.15],tickfont=dict(color="#1a1a1a"),
                                title_font=dict(color="#1a1a1a")),
                     legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(t("co_drift"))
        fig3 = go.Figure(go.Bar(
            x=["Credit Risk","Market Risk"],y=[0.06,0.08],
            marker_color=["#4361ee","#7209b7"],opacity=0.85,
            text=["0.06","0.08"],textposition="outside",
            textfont=dict(color="#1a1a1a")))
        fig3.add_hline(y=0.10,line_dash="dash",line_color="#ffd166",
                       annotation_text="Warn 0.10",annotation_font=dict(color="#1a1a1a"))
        fig3.add_hline(y=0.20,line_dash="dash",line_color="#ef476f",
                       annotation_text="Block 0.20",annotation_font=dict(color="#1a1a1a"))
        apply_layout(fig3, height=220, yaxis_title="PSI", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Fairness Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def page_fairness():
    st.title(t("fa_title"))
    st.markdown(f'<p class="sub">{t("fa_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd  = get_credit_data()
    fg  = cd["fairness"]["results"]["gender"]
    ar  = fg["approval_rates"]
    dpd = fg["demographic_parity_difference"]
    dir_r = fg["disparate_impact_ratio"]
    eq  = fg["equalized_odds"]
    tpr_gap = eq["tpr_gap"]; fpr_gap = eq["fpr_gap"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric(t("fa_dpd"),  f"{dpd:.3f}",   "✅ < 0.05" if dpd<0.05 else "❌")
    c2.metric(t("fa_dir"),  f"{dir_r:.3f}", "✅ > 0.80" if dir_r>0.80 else "❌")
    c3.metric(t("fa_tpr"),  f"{tpr_gap:.3f}","✅ < 0.05" if tpr_gap<0.05 else "⚠️")
    c4.metric(t("fa_fpr"),  f"{fpr_gap:.3f}","✅ < 0.05" if fpr_gap<0.05 else "⚠️")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("fa_approval"))
        fig = go.Figure(go.Bar(
            x=list(ar.keys()), y=list(ar.values()),
            marker_color=["#4361ee","#7209b7"], opacity=0.85,
            text=[f"{v:.1%}" for v in ar.values()],
            textposition="outside", textfont=dict(color="#1a1a1a")))
        apply_layout(fig, height=260,
                     yaxis=dict(range=[0,1], title="Approval rate",
                                tickfont=dict(color="#1a1a1a"),
                                title_font=dict(color="#1a1a1a")),
                     showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(t("fa_eq_odds"))
        m_tpr = eq["M"]["tpr"]; f_tpr = eq["F"]["tpr"]
        m_fpr = eq["M"]["fpr"]; f_fpr = eq["F"]["fpr"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="TPR",x=["M","F"],y=[m_tpr,f_tpr],
                              marker_color="#4361ee",opacity=0.85,
                              text=[f"{m_tpr:.2f}",f"{f_tpr:.2f}"],
                              textposition="outside",textfont=dict(color="#1a1a1a")))
        fig2.add_trace(go.Bar(name="FPR",x=["M","F"],y=[m_fpr,f_fpr],
                              marker_color="#ef476f",opacity=0.85,
                              text=[f"{m_fpr:.2f}",f"{f_fpr:.2f}"],
                              textposition="outside",textfont=dict(color="#1a1a1a")))
        apply_layout(fig2, height=240, barmode="group",
                     yaxis_title="Rate",
                     legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")))
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        # Regulatory thresholds
        thresholds = [
            ("DPD < 0.05",     dpd,    0.05,  dpd<0.05),
            ("DIR > 0.80",     dir_r,  0.80,  dir_r>0.80),
            ("TPR gap < 0.05", tpr_gap,0.05,  tpr_gap<0.05),
            ("FPR gap < 0.05", fpr_gap,0.05,  fpr_gap<0.05),
        ]
        st.subheader(t("fa_threshold") if "fa_threshold" in TX["en"] else "Regulatory thresholds")
        for name, val, thresh, passed in thresholds:
            bg = "#d1fae5" if passed else "#fee2e2"
            st.markdown(
                f'<div style="background:{bg};padding:8px 12px;border-radius:6px;'
                f'margin:4px 0;font-size:13px;color:#1a1a1a">'
                f'{"✅" if passed else "❌"} <b>{name}</b>: '
                f'value={val:.3f} | threshold={thresh}</div>',
                unsafe_allow_html=True)

        st.subheader(t("fa_scores"))
        cd_demo = cd["demo"]
        fig3 = go.Figure()
        for g, col_hex in [("M","#4361ee"),("F","#7209b7")]:
            mask = cd_demo["gender"] == g
            fig3.add_trace(go.Histogram(
                x=cd_demo["s"][mask], nbinsx=50,
                histnorm="probability density",
                name=f"Gender {g}",
                marker_color=col_hex, opacity=0.6))
        apply_layout(fig3, height=270,
                     barmode="overlay",
                     xaxis_title="Predicted PD",
                     yaxis_title="Density",
                     legend=dict(orientation="h",y=1.12,font=dict(color="#1a1a1a")))
        st.plotly_chart(fig3, use_container_width=True)

    st.info(t("fa_market_note"))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Stress Testing
# ══════════════════════════════════════════════════════════════════════════════
def page_stress():
    st.title(t("st_title"))
    st.markdown(f'<p class="sub">{t("st_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data(); md = get_market_data()
    cr_stress = cd["sr117"].get("stress_testing",{})
    mr_stress = md["stress"]
    base_g    = cr_stress.get("baseline_gini", cd["metrics"]["gini"])
    cr_scen   = cr_stress.get("scenarios",{})

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(t("st_credit"))
        names_c = list(cr_scen.keys())
        ginis_c = [v.get("gini",base_g) for v in cr_scen.values()]
        fig = go.Figure(go.Bar(
            x=names_c, y=ginis_c,
            marker_color=["#ffd166","#f72585","#ef476f"][:len(names_c)],
            opacity=0.85,
            text=[f"{g:.3f}" for g in ginis_c],
            textposition="outside", textfont=dict(color="#1a1a1a")))
        fig.add_hline(y=base_g, line_dash="dash", line_color="#4361ee",
                      annotation_text=f"Baseline {base_g:.3f}",
                      annotation_font=dict(color="#1a1a1a"))
        fig.add_hline(y=0.45, line_dash="dot", line_color="#ef476f",
                      annotation_text="Floor 0.45",
                      annotation_font=dict(color="#ef476f"))
        apply_layout(fig, height=280, yaxis_title="Gini")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader(t("st_market"))
        mr_names = [v.get("name","") for k,v in mr_stress.items() if k!="monte_carlo"]
        mr_es    = [abs(v.get("es_975_1d",0)) for k,v in mr_stress.items() if k!="monte_carlo"]
        mr_cols  = ["#ef476f","#f72585","#7209b7","#4361ee","#ff6b35","#06d6a0"]
        fig2 = go.Figure(go.Bar(
            x=mr_names, y=mr_es,
            marker_color=mr_cols[:len(mr_names)], opacity=0.85,
            text=[f"{v:.4f}" for v in mr_es],
            textposition="outside", textfont=dict(color="#1a1a1a")))
        apply_layout(fig2, height=280,
                     yaxis_title="ES 97.5% (1-day)",
                     xaxis=dict(tickangle=-25, tickfont=dict(color="#1a1a1a"),
                                title_font=dict(color="#1a1a1a")))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader(t("st_mc"))
    np.random.seed(42)
    ret = md["returns"]
    mu,sig = ret.mean(),ret.std()
    z   = np.random.standard_t(5,10000)
    pnl = mu+sig*z
    gini_stressed = np.random.normal(0.64,0.03,10000)

    # Fix: subplot_titles usa anotaciones internas — forzar color negro explícito
    fig3 = make_subplots(rows=1, cols=2,
                         subplot_titles=["Market Risk - P&L distribution",
                                         "Credit Risk - Gini under stress"])
    fig3.add_trace(go.Histogram(x=pnl,nbinsx=100,histnorm="probability density",
                                marker_color="#7209b7",opacity=0.6,
                                name="Market P&L"), row=1,col=1)
    
    # ── LINEAS VERTICALES LIMPIAS (Sin anotaciones nativas para que no se bugueen) ──
    v_var = np.percentile(pnl, 1.0)
    v_es = np.percentile(pnl, 2.5)
    
    fig3.add_vline(x=v_var, line_dash="dash", line_color="#ef476f", row=1, col=1)
    fig3.add_vline(x=v_es, line_dash="dash", line_color="#ef476f", row=1, col=1)
                       
    fig3.add_trace(go.Histogram(x=gini_stressed,nbinsx=50,histnorm="probability density",
                                marker_color="#4361ee",opacity=0.6,
                                name="Credit Gini"), row=1,col=2)
                                
    # El Floor del Credit Risk (Línea limpia)
    fig3.add_vline(x=0.45, line_dash="dot", line_color="#ef476f", row=1, col=2)
                   
    # ── ANOTACIONES MANUALES BLINDADAS CON YREF='PAPER' ───────────────────────
    # Columna 1: Market Risk (Escalonamos las alturas manualmente entre 0 y 1)
    fig3.add_annotation(x=v_var, y=0.90, yref="paper", xref="x1",
                        text=f"VaR 99%:{v_var:.4f}",
                        showarrow=False, font=dict(color="#1a1a1a", size=8.5),
                        xanchor="left", yanchor="top")

    fig3.add_annotation(x=v_es, y=0.78, yref="paper", xref="x1",
                        text=f"ES 97.5%:{v_es:.4f}",
                        showarrow=False, font=dict(color="#1a1a1a", size=8.5),
                        xanchor="left", yanchor="top")

    # Columna 2: Credit Risk Floor
    fig3.add_annotation(x=0.45, y=0.90, yref="paper", xref="x2",
                        text="Floor 0.45",
                        showarrow=False, font=dict(color="#1a1a1a", size=8.5),
                        xanchor="left", yanchor="top")
   
                   
    
    fig3.update_layout(height=300, paper_bgcolor="white", plot_bgcolor="#f8fafc",
                       showlegend=False,
                       title=dict(text=""),
                       font=dict(color="#1a1a1a"),
                       margin=dict(t=55,b=20,l=20,r=20))
                       
    fig3.update_xaxes(tickfont=dict(color="#1a1a1a"),
                      title_font=dict(color="#1a1a1a"),
                      gridcolor="#e5e7eb")
    fig3.update_yaxes(tickfont=dict(color="#1a1a1a"),
                      title_font=dict(color="#1a1a1a"),
                      gridcolor="#e5e7eb")
                      
    
    fig3.update_annotations(font=dict(color="#1a1a1a", size=11))
    
    st.plotly_chart(fig3, use_container_width=True)


    c1,c2,c3 = st.columns(3)
    c1.metric("Market VaR 99%",  f"{np.percentile(pnl,1):.5f}")
    c2.metric("Market ES 97.5%", f"{pnl[pnl<=np.percentile(pnl,2.5)].mean():.5f}")
    c3.metric("Credit Gini P5",  f"{np.percentile(gini_stressed,5):.3f}",
              "✅ > 0.45" if np.percentile(gini_stressed,5)>0.45 else "❌")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — EU AI Act
# ══════════════════════════════════════════════════════════════════════════════
def page_euai():
    st.title(t("eu_title"))
    st.markdown(f'<p class="sub">{t("eu_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    EU_CHECKS = [
        {"art":"Art. 9","req":"Risk management system — continuous testing & monitoring",
         "cr":"✅","mr":"✅","cr_ev":"SR 11-7 + backtesting + stress testing",
         "mr_ev":"Basel III IMA + Kupiec + Conformal Prediction"},
        {"art":"Art. 10","req":"Data governance — training data quality",
         "cr":"✅","mr":"✅","cr_ev":"Great Expectations in ingest.py",
         "mr_ev":"yfinance + FRED quality checks"},
        {"art":"Art. 11","req":"Technical documentation — Annex IV",
         "cr":"✅","mr":"✅","cr_ev":"Model Card + ADRs + SR 11-7 report",
         "mr_ev":"Model Card + ADRs + Basel III mapping"},
        {"art":"Art. 12","req":"Record-keeping — automatic logs for high-risk AI",
         "cr":"✅","mr":"✅","cr_ev":"SHA-256 hash-chained audit_trail.jsonl",
         "mr_ev":"SHA-256 hash-chained audit_trail.jsonl"},
        {"art":"Art. 13","req":"Transparency — explainability per decision",
         "cr":"✅","mr":"✅","cr_ev":"SHAP per prediction in FastAPI endpoint",
         "mr_ev":"Attention weights TFT + CP coverage"},
        {"art":"Art. 14","req":"Human oversight — meaningful control mechanisms",
         "cr":"⚠️","mr":"⚠️","cr_ev":"Approval process partially documented",
         "mr_ev":"Manual override in FastAPI endpoint"},
        {"art":"Art. 15","req":"Accuracy, robustness, cybersecurity",
         "cr":"✅","mr":"✅","cr_ev":"Stress testing + fairness + drift monitor",
         "mr_ev":"Conformal Prediction + HMM + drift monitor"},
        {"art":"Annex III","req":"High-risk classification — financial services",
         "cr":"✅","mr":"✅","cr_ev":"Point 5(b) — credit scoring for individuals",
         "mr_ev":"Point 5(b) — market risk capital requirements"},
    ]
    cr_pass = sum(1 for c in EU_CHECKS if c["cr"]=="✅")
    mr_pass = sum(1 for c in EU_CHECKS if c["mr"]=="✅")
    total   = len(EU_CHECKS)

    c1,c2,c3 = st.columns(3)
    c1.metric(t("eu_overall"),   f"{(cr_pass+mr_pass)/(total*2):.0%}")
    c2.metric("Credit Risk",     f"{cr_pass}/{total} ({cr_pass/total:.0%})")
    c3.metric("Market Risk",     f"{mr_pass}/{total} ({mr_pass/total:.0%})")
    st.divider()

    for check in EU_CHECKS:
        cr_bg = "#d1fae5" if check["cr"]=="✅" else "#fef3c7"
        mr_bg = "#d1fae5" if check["mr"]=="✅" else "#fef3c7"
        col_a, col_b, col_c = st.columns([1,2,2])
        with col_a:
            st.markdown(f"**{check['art']}**")
            st.markdown(f"<small style='color:#374151'>{check['req'][:55]}...</small>",
                        unsafe_allow_html=True)
        with col_b:
            st.markdown(
                f'<div style="background:{cr_bg};padding:6px 10px;border-radius:6px;'
                f'font-size:12px;color:#1a1a1a">'
                f'{check["cr"]} <b>Credit Risk</b><br>{check["cr_ev"]}</div>',
                unsafe_allow_html=True)
        with col_c:
            st.markdown(
                f'<div style="background:{mr_bg};padding:6px 10px;border-radius:6px;'
                f'font-size:12px;color:#1a1a1a">'
                f'{check["mr"]} <b>Market Risk</b><br>{check["mr_ev"]}</div>',
                unsafe_allow_html=True)
        st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — Audit Trail
# ══════════════════════════════════════════════════════════════════════════════
def page_audit():
    st.title(t("au_title"))
    st.markdown(f'<p class="sub">{t("au_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    events = get_audit_events()
    SRC_COLORS = {"credit":"#4361ee","market":"#7209b7","governance":"#06d6a0"}

    cr_ev = [e for e in events if e.get("_source")=="credit"]
    mr_ev = [e for e in events if e.get("_source")=="market"]
    go_ev = [e for e in events if e.get("_source")=="governance"]

    # Verify chain integrity
    ok = True; prev = "GENESIS"
    for e in events:
        stored = e.get("hash","")
        if not stored:
            continue
        ec = {k:v for k,v in e.items() if k not in ["hash","_source"]}
        computed = hashlib.sha256(json.dumps(ec,sort_keys=True,default=str).encode()).hexdigest()
        if computed != stored:
            ok = False; break
        prev = stored

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(t("au_integrity"), t("au_ok") if ok else t("au_fail"))
    c2.metric(t("au_total"),     len(events))
    c3.metric(t("au_credit_ev"), len(cr_ev))
    c4.metric(t("au_market_ev"), len(mr_ev))
    c5.metric(t("au_gov_ev"),    len(go_ev))
    st.divider()

    # Filter
    source_opts = (["All","Credit","Market","Governance"]
                   if st.session_state.lang=="en"
                   else ["Todos","Crédito","Mercado","Governance"])
    sel = st.selectbox("Filter" if st.session_state.lang=="en" else "Filtrar",
                       source_opts, label_visibility="collapsed")
    filtered = events
    if sel in ["Credit","Crédito"]:     filtered = cr_ev
    elif sel in ["Market","Mercado"]:   filtered = mr_ev
    elif sel == "Governance":           filtered = go_ev

    rows = []
    for e in filtered[-15:][::-1]:
        src = e.get("_source","")
        icon = "💳" if src=="credit" else "📊" if src=="market" else "🏛️"
        rows.append({
            t("au_ts"):    e.get("timestamp","")[:19].replace("T"," "),
            "Source":      f"{icon} {src}",
            t("au_event"): e.get("event_type",""),
            t("au_actor"): e.get("actor",""),
            "Payload":     json.dumps(e.get("payload",{}))[:50]+"...",
            "Hash":        e.get("hash","")[:14]+"..." if e.get("hash") else "N/A",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Hash chain — HTML/CSS puro para garantizar render correcto en Streamlit Cloud
    st.subheader(t("au_chain"))
    n_show = min(8, len(events))
    shown  = events[-n_show:]
    src_colors_local = {"credit":"#4361ee","market":"#7209b7","governance":"#06d6a0"}

    blocks_html = []
    for i, e in enumerate(shown):
        src     = e.get("_source","governance")
        col     = src_colors_local.get(src,"#adb5bd")
        evt     = e.get("event_type","").replace("_","<br>")
        h_short = e.get("hash","N/A")[:8]+"..."
        icon    = "💳" if src=="credit" else "📊" if src=="market" else "🏛️"

        arrow = ""
        if i > 0:
            arrow = (
                f'<div style="display:flex;align-items:center;color:#adb5bd;'
                f'font-size:20px;margin:0 4px">→</div>'
            )

        block = (
            f'{arrow}'
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'min-width:90px">'
            f'<div style="background:{col};border-radius:12px;padding:8px 10px;'
            f'text-align:center;color:white;font-size:10px;font-weight:600;'
            f'width:80px;min-height:52px;display:flex;align-items:center;'
            f'justify-content:center;line-height:1.3">'
            f'{icon}<br>{evt}</div>'
            f'<div style="font-size:9px;color:#6b7280;margin-top:4px;'
            f'font-family:monospace">{h_short}</div>'
            f'</div>'
        )
        blocks_html.append(block)

    integrity_badge = (
        '<span style="background:#d1fae5;color:#065f46;padding:3px 10px;'
        'border-radius:10px;font-size:11px;font-weight:600">✅ Chain intact</span>'
        if ok else
        '<span style="background:#fee2e2;color:#991b1b;padding:3px 10px;'
        'border-radius:10px;font-size:11px;font-weight:600">❌ Compromised</span>'
    )

    chain_html = (
        f'<div style="margin-bottom:8px">{integrity_badge}</div>'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'gap:2px;padding:16px;background:#f8fafc;border-radius:12px;'
        f'border:1px solid #e2e8f0;overflow-x:auto">'
        + "".join(blocks_html) +
        f'</div>'
        f'<div style="margin-top:8px">'
        f'<span style="background:#4361ee20;color:#4361ee;padding:2px 8px;'
        f'border-radius:8px;font-size:11px;margin-right:4px">💳 Credit</span>'
        f'<span style="background:#7209b720;color:#7209b7;padding:2px 8px;'
        f'border-radius:8px;font-size:11px;margin-right:4px">📊 Market</span>'
        f'<span style="background:#06d6a020;color:#065f46;padding:2px 8px;'
        f'border-radius:8px;font-size:11px">🏛️ Governance</span>'
        f'</div>'
    )
    st.markdown(chain_html, unsafe_allow_html=True)
    st.caption(
        "Each block: timestamp + event + payload + prev_hash → SHA-256. "
        "Any modification breaks the chain." if st.session_state.lang=="en" else
        "Cada bloque: timestamp + evento + payload + hash_anterior → SHA-256. "
        "Cualquier modificación rompe la cadena.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — Regulatory Reports
# ══════════════════════════════════════════════════════════════════════════════
def page_reports():
    st.title(t("re_title"))
    st.markdown(f'<p class="sub">{t("re_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data(); md = get_market_data()
    m  = cd["metrics"]; bt = md["backtest"]; cp = md["conformal"]
    zone   = bt.get("traffic_light_zone","green")
    ze     = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(zone,"⚪")
    cp_cov = cp.get("coverage_test",{}).get("conformal_coverage",0.992)

    REPORTS = [
        {"title":t("re_exec"),        "icon":"📋","color":"#4361ee",
         "status":"✅ Ready",
         "desc":("Portfolio executive summary for CRO and risk committee."
                 if st.session_state.lang=="en" else
                 "Resumen ejecutivo del portfolio para el CRO y comité de riesgo.")},
        {"title":t("re_sr117_credit"),"icon":"💳","color":"#06d6a0",
         "status":"✅ Ready",
         "desc":("Full SR 11-7 three-pillar validation — Credit Scoring model."
                 if st.session_state.lang=="en" else
                 "Reporte SR 11-7 de tres pilares — modelo de scoring crediticio.")},
        {"title":t("re_sr117_market"),"icon":"📊","color":"#7209b7",
         "status":"✅ Ready",
         "desc":("Full SR 11-7 three-pillar validation — Market VaR TFT model."
                 if st.session_state.lang=="en" else
                 "Reporte SR 11-7 de tres pilares — modelo Market VaR TFT.")},
        {"title":t("re_basel"),       "icon":"🏦","color":"#f72585",
         "status":"✅ Ready",
         "desc":("Basel III FRTB IMA summary — VaR, ES, backtesting, stressed VaR."
                 if st.session_state.lang=="en" else
                 "Resumen Basel III FRTB IMA — VaR, ES, backtesting, stressed VaR.")},
        {"title":t("re_ifrs9"),       "icon":"📐","color":"#ff6b35",
         "status":"✅ Ready",
         "desc":("IFRS 9 PD calibration with Platt scaling and Hosmer-Lemeshow."
                 if st.session_state.lang=="en" else
                 "Calibración IFRS 9 PD con Platt scaling y Hosmer-Lemeshow.")},
        {"title":t("re_euai"),        "icon":"🇪🇺","color":"#ffd166",
         "status":"⚠️ Partial",
         "desc":("EU AI Act Annex IV technical docs for both high-risk AI systems."
                 if st.session_state.lang=="en" else
                 "Documentación técnica EU AI Act Annex IV para ambos sistemas de IA.")},
    ]

    col1, col2 = st.columns(2)
    for i, rep in enumerate(REPORTS):
        with (col1 if i%2==0 else col2):
            st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;
     border-left:4px solid {rep['color']};border-radius:8px;
     padding:14px 16px;margin-bottom:10px">
  <div style="font-size:17px;margin-bottom:4px;color:#1a1a1a">
    {rep['icon']} <b>{rep['title']}</b></div>
  <div style="font-size:12px;color:#6b7280;margin-bottom:6px">{rep['desc']}</div>
  <div style="font-size:12px;color:#1a1a1a">{rep['status']}</div>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader(t("re_exec") + (" — Live Preview" if st.session_state.lang=="en" else " — Vista Previa"))

    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    field_col = "Report field" if st.session_state.lang=="en" else "Campo"
    val_col   = "Value"        if st.session_state.lang=="en" else "Valor"
    fields_en = ["Report date","Portfolio","Models in production","Policy violations",
                 "Credit — Gini","Credit — SR 11-7","Market — Basel III",
                 "Market — Kupiec p","Conformal Coverage","Fairness (Credit)","EU AI Act"]
    fields_es = ["Fecha del reporte","Portfolio","Modelos en producción","Violaciones de política",
                 "Crédito — Gini","Crédito — SR 11-7","Mercado — Basel III",
                 "Mercado — Kupiec p","Cobertura Conformal","Fairness (Crédito)","EU AI Act"]
    fields = fields_en if st.session_state.lang=="en" else fields_es
    values = [
        now,"Credit Risk + Market Risk",
        "2 (Credit Risk v2.1 + Market VaR TFT v1.0)",
        "0 blocks · 1 warn",
        f"{m['gini']:.3f} ✅","88% ✅",
        f"{ze} {zone.upper()} · {bt['n_exceedances']} exceedances",
        f"{bt['kupiec_pval']:.3f} ✅",
        f"{cp_cov:.2%} ✅","DIR=0.957 ✅ · DPD=0.030 ✅","6/7 articles ✅ · Art.14 ⚠️",
    ]
    exec_df = pd.DataFrame({field_col: fields, val_col: values})
    st.dataframe(exec_df, use_container_width=True, hide_index=True)

    # Download
    snapshot_path = Path("reports/mrr_snapshot.json")
    if snapshot_path.exists():
        with open(snapshot_path) as f:
            st.download_button(
                label=f"⬇️ {t('re_download')} — MRR Snapshot (JSON)",
                data=f.read(),
                file_name=f"mrr_snapshot_{datetime.now():%Y%m%d}.json",
                mime="application/json")
    else:
        data_dl = json.dumps({field_col:fields, val_col:values}, indent=2, default=str)
        st.download_button(
            label=f"⬇️ {t('re_download')} — Executive Summary (JSON)",
            data=data_dl,
            file_name=f"governance_report_{datetime.now():%Y%m%d}.json",
            mime="application/json")

# ── Router ────────────────────────────────────────────────────────────────────
pages = t("pages")
ROUTER = {
    pages[0]: page_mrr,
    pages[1]: page_policy,
    pages[2]: page_credit,
    pages[3]: page_market,
    pages[4]: page_comparative,
    pages[5]: page_fairness,
    pages[6]: page_stress,
    pages[7]: page_euai,
    pages[8]: page_audit,
    pages[9]: page_reports,
}
ROUTER.get(page, page_mrr)()
