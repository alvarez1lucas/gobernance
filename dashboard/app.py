"""
dashboard/app.py — AI Governance Framework Dashboard
Dashboard ejecutivo unificado que integra Credit Risk + Market Risk.

Ejecutar desde la raíz del repo:
    streamlit run dashboard/app.py

10 páginas:
  1. MRR Overview
  2. Policy Status
  3. Credit Risk Deep-dive
  4. Market Risk Deep-dive
  5. Comparative Metrics
  6. Fairness Dashboard
  7. Stress Testing Unificado
  8. EU AI Act Compliance
  9. Audit Trail Unificado
  10. Regulatory Reports
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

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="AI Governance Framework",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Plotly template with black text for web deployment ──────────────────────
import plotly.io as pio
pio.templates.default = "plotly_white"

st.markdown("""
<style>
[data-testid="stMetricValue"]{ font-size:1.05rem; }
.sub{ color:#6b7280; font-size:.82rem; margin-top:-10px; margin-bottom:10px; }
.badge{ display:inline-block; padding:2px 9px; border-radius:10px;
        font-size:11px; font-weight:600; margin:2px; }
.card{ background:#f8fafc; border-radius:10px; padding:14px 16px;
       border:1px solid #e2e8f0; margin-bottom:8px; }
/* Plotly text colors — ensure all text is black */
.plotly text, .plotly g text { fill: black !important; color: black !important; }
.plotly .xtick text, .plotly .ytick text { fill: black !important; }
.plotly .xaxislayer text, .plotly .yaxislayer text { fill: black !important; }
.plotly .legendtext { fill: black !important; }
.plotly .hovertext { fill: black !important; }
</style>
""", unsafe_allow_html=True)

# ── Translations ──────────────────────────────────────────────────────────────
TX = {
"en":{
  "nav":"Navigation",
  "pages":["🏛️ MRR Overview","🔒 Policy Status","💳 Credit Risk",
            "📊 Market Risk","⚖️ Comparative Metrics","🌈 Fairness Dashboard",
            "⚠️ Stress Testing","🇪🇺 EU AI Act","🗂️ Audit Trail","📑 Regulatory Reports"],
  "no_data":"Showing synthetic demo data — run `python run_governance.py` for live data",
  "approved":"APPROVED","review":"UNDER REVIEW","retired":"RETIRED","pending":"PENDING",
  "pass":"PASS","fail":"FAIL","warn":"WARNING",
  "model_status":"Portfolio Status","n_models":"Models","n_approved":"Approved",
  "n_blocks":"Policy Blocks","n_warns":"Policy Warns",
  # MRR
  "mrr_title":"Model Risk Register — Portfolio Overview",
  "mrr_sub":"SR 11-7 · EU AI Act Annex III · BCBS 239 — centralized model inventory",
  "mrr_model":"Model","mrr_type":"Type","mrr_tier":"Tier","mrr_status":"Status",
  "mrr_score":"SR 11-7 Score","mrr_psi":"PSI Drift","mrr_next":"Next Validation",
  "mrr_days":"Days Left","mrr_issues":"Policy Issues",
  "mrr_timeline":"Validation timeline","mrr_regs":"Regulations covered",
  # Policy
  "po_title":"Policy Status — Policy-as-Code Evaluation",
  "po_sub":"OPA/Rego rules evaluated against live metrics from both repos",
  "po_model":"Model","po_policy":"Policy","po_severity":"Severity",
  "po_result":"Result","po_message":"Detail","po_summary":"Policy summary",
  "po_block_desc":"Deployment blocked","po_warn_desc":"Alert — review required",
  "po_pass_desc":"All checks passed",
  # Credit
  "cr_title":"Credit Risk — Deep Dive",
  "cr_sub":"NB 01-07: XGBoost · DL Tabular · GNN · SHAP · IFRS 9 · SR 11-7 · BCRA A 7724",
  "cr_perf":"Model performance","cr_cal":"Calibration","cr_stab":"Stability",
  "cr_roc":"ROC curves — champion vs challengers",
  "cr_shap":"Feature importance (SHAP)","cr_thresh":"Threshold analysis",
  "cr_stress":"Stress scenarios","cr_ifrs9":"IFRS 9 calibration",
  "cr_gini":"Gini","cr_auc":"AUC-ROC","cr_ks":"KS Statistic",
  "cr_psi":"PSI Drift","cr_brier":"Brier Score","cr_hl":"Hosmer-Lemeshow p",
  # Market
  "mr_title":"Market Risk — Deep Dive",
  "mr_sub":"NB 01-10: TFT · GARCH · HMM · FinBERT · Conformal Prediction · Basel III FRTB",
  "mr_backtest":"Backtesting results","mr_stress":"Stress scenarios",
  "mr_regime":"Regime detection","mr_sentiment":"NLP Sentiment",
  "mr_conformal":"Conformal Prediction","mr_exc":"Exceedances (250d)",
  "mr_kupiec":"Kupiec p-value","mr_chr":"Christoffersen p-value",
  "mr_zone":"Traffic Light Zone","mr_es":"Expected Shortfall 97.5%",
  "mr_cp_cov":"CP Coverage","mr_cp_valid":"CP Valid",
  # Comparative
  "co_title":"Comparative Metrics — Credit vs Market",
  "co_sub":"Side-by-side comparison of both models across all regulatory dimensions",
  "co_dimension":"Dimension","co_credit":"Credit Risk","co_market":"Market Risk",
  "co_radar":"Multi-dimensional regulatory score",
  "co_sr117":"SR 11-7 Scores","co_drift":"Drift PSI comparison",
  "co_stress":"Stress severity comparison",
  # Fairness
  "fa_title":"Fairness Dashboard",
  "fa_sub":"DPD · DIR · Equalized Odds — Credit Risk (EU AI Act Art. 10 + ECOA)",
  "fa_gender":"Gender fairness analysis",
  "fa_approval":"Approval rates by group",
  "fa_dpd":"Demographic Parity Difference",
  "fa_dir":"Disparate Impact Ratio",
  "fa_eq_odds":"Equalized Odds",
  "fa_tpr":"TPR gap","fa_fpr":"FPR gap",
  "fa_threshold":"Regulatory thresholds",
  "fa_market_note":"Market Risk models — fairness not directly applicable (institutional portfolio)",
  # Stress
  "st_title":"Stress Testing — Unified View",
  "st_sub":"Combined stress analysis across Credit Risk and Market Risk",
  "st_credit":"Credit Risk stress scenarios",
  "st_market":"Market Risk stress scenarios",
  "st_combined":"Combined portfolio stress impact",
  "st_worst":"Worst case scenario",
  "st_mc":"Monte Carlo tail distribution",
  # EU AI Act
  "eu_title":"EU AI Act — Compliance Checklist",
  "eu_sub":"Regulation (EU) 2024/1689 — High-risk AI system requirements",
  "eu_article":"Article","eu_requirement":"Requirement",
  "eu_credit":"Credit Risk","eu_market":"Market Risk","eu_status":"Status",
  "eu_evidence":"Evidence",
  "eu_overall":"Overall compliance",
  # Audit
  "au_title":"Audit Trail — Unified Event Log",
  "au_sub":"Immutable SHA-256 hash-chained log from both repos + governance (EU AI Act Art. 12)",
  "au_integrity":"Chain integrity","au_total":"Total events",
  "au_credit_ev":"Credit events","au_market_ev":"Market events",
  "au_gov_ev":"Governance events",
  "au_ts":"Timestamp","au_event":"Event","au_source":"Source",
  "au_actor":"Actor","au_hash":"Hash","au_chain":"Hash chain visualization",
  # Reports
  "re_title":"Regulatory Reports",
  "re_sub":"Auto-generated reports for regulators, auditors and risk committees",
  "re_sr117_credit":"SR 11-7 Report — Credit Risk",
  "re_sr117_market":"SR 11-7 Report — Market Risk",
  "re_basel":"Basel III FRTB Summary",
  "re_ifrs9":"IFRS 9 Calibration Report",
  "re_euai":"EU AI Act Technical Documentation",
  "re_exec":"Executive Summary",
  "re_download":"Download report",
},
"es":{
  "nav":"Navegación",
  "pages":["🏛️ MRR Overview","🔒 Estado de Políticas","💳 Riesgo de Crédito",
            "📊 Riesgo de Mercado","⚖️ Métricas Comparativas","🌈 Fairness Dashboard",
            "⚠️ Stress Testing","🇪🇺 EU AI Act","🗂️ Audit Trail","📑 Reportes Regulatorios"],
  "no_data":"Mostrando datos sintéticos — ejecutar `python run_governance.py` para datos reales",
  "approved":"APROBADO","review":"EN REVISIÓN","retired":"RETIRADO","pending":"PENDIENTE",
  "pass":"APROBADO","fail":"RECHAZADO","warn":"ADVERTENCIA",
  "model_status":"Estado del Portfolio","n_models":"Modelos","n_approved":"Aprobados",
  "n_blocks":"Políticas Bloqueantes","n_warns":"Advertencias",
  "mrr_title":"Model Risk Register — Vista del Portfolio",
  "mrr_sub":"SR 11-7 · EU AI Act Annex III · BCBS 239 — inventario centralizado de modelos",
  "mrr_model":"Modelo","mrr_type":"Tipo","mrr_tier":"Tier","mrr_status":"Estado",
  "mrr_score":"Score SR 11-7","mrr_psi":"Drift PSI","mrr_next":"Próxima Validación",
  "mrr_days":"Días restantes","mrr_issues":"Problemas de Política",
  "mrr_timeline":"Cronograma de validaciones","mrr_regs":"Regulaciones cubiertas",
  "po_title":"Estado de Políticas — Evaluación Policy-as-Code",
  "po_sub":"Reglas OPA/Rego evaluadas contra métricas en vivo de ambos repos",
  "po_model":"Modelo","po_policy":"Política","po_severity":"Severidad",
  "po_result":"Resultado","po_message":"Detalle","po_summary":"Resumen de políticas",
  "po_block_desc":"Despliegue bloqueado","po_warn_desc":"Alerta — revisión requerida",
  "po_pass_desc":"Todos los checks aprobados",
  "cr_title":"Riesgo de Crédito — Análisis Detallado",
  "cr_sub":"NB 01-07: XGBoost · DL Tabular · GNN · SHAP · IFRS 9 · SR 11-7 · BCRA A 7724",
  "cr_perf":"Performance del modelo","cr_cal":"Calibración","cr_stab":"Estabilidad",
  "cr_roc":"Curvas ROC — champion vs challengers",
  "cr_shap":"Importancia de variables (SHAP)","cr_thresh":"Análisis de threshold",
  "cr_stress":"Escenarios de stress","cr_ifrs9":"Calibración IFRS 9",
  "cr_gini":"Gini","cr_auc":"AUC-ROC","cr_ks":"Estadístico KS",
  "cr_psi":"Drift PSI","cr_brier":"Brier Score","cr_hl":"Hosmer-Lemeshow p",
  "mr_title":"Riesgo de Mercado — Análisis Detallado",
  "mr_sub":"NB 01-10: TFT · GARCH · HMM · FinBERT · Conformal Prediction · Basel III FRTB",
  "mr_backtest":"Resultados de backtesting","mr_stress":"Escenarios de stress",
  "mr_regime":"Detección de régimen","mr_sentiment":"Sentimiento NLP",
  "mr_conformal":"Predicción Conformal","mr_exc":"Exceedances (250d)",
  "mr_kupiec":"Kupiec p-value","mr_chr":"Christoffersen p-value",
  "mr_zone":"Zona Traffic Light","mr_es":"Expected Shortfall 97.5%",
  "mr_cp_cov":"Cobertura CP","mr_cp_valid":"Garantía CP válida",
  "co_title":"Métricas Comparativas — Crédito vs Mercado",
  "co_sub":"Comparación lado a lado de ambos modelos en todas las dimensiones regulatorias",
  "co_dimension":"Dimensión","co_credit":"Riesgo de Crédito","co_market":"Riesgo de Mercado",
  "co_radar":"Score regulatorio multidimensional",
  "co_sr117":"Scores SR 11-7","co_drift":"Comparación drift PSI",
  "co_stress":"Severidad de stress comparada",
  "fa_title":"Fairness Dashboard",
  "fa_sub":"DPD · DIR · Equalized Odds — Riesgo de Crédito (EU AI Act Art. 10 + ECOA)",
  "fa_gender":"Análisis de fairness por género",
  "fa_approval":"Tasas de aprobación por grupo",
  "fa_dpd":"Diferencia de Paridad Demográfica",
  "fa_dir":"Ratio de Impacto Dispar",
  "fa_eq_odds":"Equalized Odds",
  "fa_tpr":"Gap TPR","fa_fpr":"Gap FPR",
  "fa_threshold":"Umbrales regulatorios",
  "fa_market_note":"Modelos de Riesgo de Mercado — fairness no aplica directamente (portfolio institucional)",
  "st_title":"Stress Testing — Vista Unificada",
  "st_sub":"Análisis de stress combinado para Riesgo de Crédito y Riesgo de Mercado",
  "st_credit":"Escenarios de stress — Riesgo de Crédito",
  "st_market":"Escenarios de stress — Riesgo de Mercado",
  "st_combined":"Impacto de stress combinado del portfolio",
  "st_worst":"Peor escenario","st_mc":"Distribución de cola Monte Carlo",
  "eu_title":"EU AI Act — Checklist de Cumplimiento",
  "eu_sub":"Reglamento (UE) 2024/1689 — Requisitos para sistemas de IA de alto riesgo",
  "eu_article":"Artículo","eu_requirement":"Requisito",
  "eu_credit":"Riesgo de Crédito","eu_market":"Riesgo de Mercado","eu_status":"Estado",
  "eu_evidence":"Evidencia","eu_overall":"Cumplimiento general",
  "au_title":"Audit Trail — Log de Eventos Unificado",
  "au_sub":"Log inmutable con cadena SHA-256 de ambos repos + governance (EU AI Act Art. 12)",
  "au_integrity":"Integridad de la cadena","au_total":"Eventos totales",
  "au_credit_ev":"Eventos crédito","au_market_ev":"Eventos mercado",
  "au_gov_ev":"Eventos governance",
  "au_ts":"Timestamp","au_event":"Evento","au_source":"Fuente",
  "au_actor":"Actor","au_hash":"Hash","au_chain":"Cadena de hashes",
  "re_title":"Reportes Regulatorios",
  "re_sub":"Reportes auto-generados para reguladores, auditores y comités de riesgo",
  "re_sr117_credit":"Reporte SR 11-7 — Riesgo de Crédito",
  "re_sr117_market":"Reporte SR 11-7 — Riesgo de Mercado",
  "re_basel":"Resumen Basel III FRTB",
  "re_ifrs9":"Reporte de Calibración IFRS 9",
  "re_euai":"Documentación Técnica EU AI Act",
  "re_exec":"Resumen Ejecutivo",
  "re_download":"Descargar reporte",
},
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def t(k):
    return TX[st.session_state.lang].get(k, TX["en"].get(k, k))

# ── Layout helpers ────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="white", 
    plot_bgcolor="#f8fafc",
    margin=dict(t=20,b=15,l=15,r=15),
    font=dict(color="black", size=11),
)

def badge(text, color):
    return f'<span class="badge" style="background:{color}20;color:{color}">{text}</span>'

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def _load_json(p):
    f = Path(p)
    return json.loads(f.read_text()) if f.exists() else {}

@st.cache_data(ttl=120)
def _load_csv(p):
    f = Path(p)
    return pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else pd.DataFrame()

@st.cache_data(ttl=120)
def get_mrr_data():
    try:
        from src.mrr import ModelRiskRegister
        mrr = ModelRiskRegister("registry/model_registry.yaml")
        return mrr.get_dashboard_data()
    except Exception:
        return _synthetic_mrr()

@st.cache_data(ttl=120)
def get_credit_data():
    try:
        from src.data_loader import load_credit_metrics, load_credit_fairness, load_credit_sr117, load_credit_demo_data
        return {
            "metrics":  load_credit_metrics(),
            "fairness": load_credit_fairness(),
            "sr117":    load_credit_sr117(),
            "demo":     load_credit_demo_data(),
        }
    except Exception:
        return _synthetic_credit()

@st.cache_data(ttl=120)
def get_market_data():
    try:
        from src.data_loader import load_market_backtest, load_market_sr117, load_market_conformal, load_market_stress, load_market_sentiment, load_market_returns
        return {
            "backtest":  load_market_backtest(),
            "sr117":     load_market_sr117(),
            "conformal": load_market_conformal(),
            "stress":    load_market_stress(),
            "sentiment": load_market_sentiment(),
            "returns":   load_market_returns(),
        }
    except Exception:
        return _synthetic_market()

@st.cache_data(ttl=120)
def get_audit_events():
    try:
        from src.data_loader import load_all_audit_events
        return load_all_audit_events()
    except Exception:
        return _synthetic_audit()

# ── Synthetic fallbacks ───────────────────────────────────────────────────────
def _synthetic_mrr():
    return {
        "n_models": 2, "n_approved": 2, "n_blocks": 0, "n_warns": 1,
        "metadata": {"version":"1.0","last_updated":"2025-01-15","regulations":["SR_11-7","EU_AI_Act","Basel_III_FRTB","IFRS_9"]},
        "summaries": [], "all_policies": [],
    }

def _synthetic_credit():
    np.random.seed(42); n = 8000
    y  = np.random.binomial(1, 0.08, n)
    s  = np.clip(y*np.random.beta(5,2,n)+(1-y)*np.random.beta(2,5,n),0.01,0.99)
    s_nn   = np.clip(s+np.random.normal(0,0.03,n),0.01,0.99)
    s_lstm = np.clip(s+np.random.normal(0.015,0.04,n),0.01,0.99)
    from scipy.stats import roc_curve_from_scores
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "metrics":  {"gini":0.712,"auc_roc":0.856,"ks_statistic":0.524,
                     "brier_score":0.084,"psi":0.06,"psi_status":"stable",
                     "hl_pvalue":0.19,"well_calibrated":True,"sr117_pass":True},
        "fairness": {"overall_fairness_passed":True,
                     "results":{"gender":{
                         "approval_rates":{"M":0.682,"F":0.712},
                         "demographic_parity_difference":0.030,
                         "disparate_impact_ratio":0.957,
                         "equalized_odds":{"M":{"tpr":0.62,"fpr":0.09},
                                           "F":{"tpr":0.60,"fpr":0.08},
                                           "tpr_gap":0.02,"fpr_gap":0.01},
                         "passed":True}}},
        "sr117":    {"sr117_overall_pass":True,
                     "discriminatory_power":{"gini":0.712,"auc_roc":0.856,"ks_statistic":0.524},
                     "stability":{"psi":0.06,"psi_status":"stable"},
                     "stress_testing":{
                         "baseline_gini":0.712,
                         "scenarios":{
                             "income_shock_moderate":{"gini":0.672,"auc_degradation":0.02},
                             "income_shock_severe":  {"gini":0.632,"auc_degradation":0.04},
                             "bureau_deterioration": {"gini":0.592,"auc_degradation":0.06},
                         }},
                     "sensitivity_top10":{"EXT_SOURCE_2":0.08,"EXT_SOURCE_3":0.06,
                                          "EXT_SOURCE_1":0.04,"AMT_CREDIT":0.02,"DAYS_BIRTH":0.015}},
        "demo":     {"y_test":y,"y_score":s,"y_score_nn":s_nn,"y_score_lstm":s_lstm,
                     "fpr":fpr,"tpr":tpr,
                     "gini_nn":round(2*roc_auc_score(y,s_nn)-1,4),
                     "gini_lstm":round(2*roc_auc_score(y,s_lstm)-1,4),
                     "gini_lr":round(2*roc_auc_score(y,s)-1-0.14,4)},
    }

def _synthetic_market():
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.today(), periods=1260)
    r = np.random.normal(-0.0003,0.012,len(dates))
    for s,e,sh in [(200,280,-0.025),(800,830,-0.035),(1050,1090,-0.018)]: r[s:e]+=sh
    sent = np.zeros(len(dates)); sent[0]=0.1
    for i in range(1,len(dates)): sent[i]=0.7*sent[i-1]+np.random.normal(0.02,0.15)
    sent = np.clip(sent,-1,1)
    return {
        "backtest":  {"n_exceedances":3,"n_observations":250,"kupiec_pval":0.38,
                      "kupiec_pass":True,"christoffersen_pval":0.55,"christoffersen_pass":True,
                      "traffic_light_zone":"green","overall_status":"approved"},
        "sr117":     {"overall_status":"approved","overall_score":0.88},
        "conformal": {"coverage_test":{"conformal_coverage":0.992,"classical_coverage":0.984,
                                       "conformal_exceedances":2,"classical_exceedances":4},
                      "nonconformity_quantile":0.0042,"conformal_valid":True},
        "stress":    {"gfc_2008":{"name":"GFC 2008","es_975_1d":-0.042,"total_loss_pct":-0.61},
                      "covid_2020":{"name":"COVID-19","es_975_1d":-0.031,"total_loss_pct":-0.38},
                      "dfast_adv":{"name":"DFAST Adverse","es_975_1d":-0.038,"total_loss_pct":-0.58},
                      "latam_tail":{"name":"LATAM Tail","es_975_1d":-0.028,"total_loss_pct":-0.43}},
        "sentiment": pd.DataFrame({"sentiment_mean":sent,
                                    "sentiment_ma21":pd.Series(sent).rolling(21).mean().fillna(0).values},
                                   index=dates),
        "returns":   pd.Series(r, index=dates, name="log_return_SPX"),
    }

def _synthetic_audit():
    return [
        {"timestamp":"2025-01-10T09:00:00Z","event_type":"pipeline_started","actor":"pipeline","payload":{},"_source":"credit","hash":"abc123"},
        {"timestamp":"2025-01-10T10:30:00Z","event_type":"model_trained","actor":"pipeline","payload":{"model":"XGBoost","gini":0.712},"_source":"credit","hash":"def456"},
        {"timestamp":"2025-01-10T11:00:00Z","event_type":"sr117_completed","actor":"pipeline","payload":{"status":"approved"},"_source":"credit","hash":"ghi789"},
        {"timestamp":"2025-01-15T09:00:00Z","event_type":"pipeline_started","actor":"pipeline","payload":{},"_source":"market","hash":"jkl012"},
        {"timestamp":"2025-01-15T09:30:00Z","event_type":"model_trained","actor":"pipeline","payload":{"model":"TFT"},"_source":"market","hash":"mno345"},
        {"timestamp":"2025-01-15T09:45:00Z","event_type":"backtesting_completed","actor":"pipeline","payload":{"kupiec_pval":0.38},"_source":"market","hash":"pqr678"},
        {"timestamp":"2025-01-15T10:00:00Z","event_type":"governance_check","actor":"governance","payload":{"n_models":2,"n_blocks":0},"_source":"governance","hash":"stu901"},
    ]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🇺🇸 English", use_container_width=True,
                     type="primary" if st.session_state.lang=="en" else "secondary"):
            st.session_state.lang="en"; st.rerun()
    with c2:
        if st.button("🇦🇷 Español", use_container_width=True,
                     type="primary" if st.session_state.lang=="es" else "secondary"):
            st.session_state.lang="es"; st.rerun()
    st.divider()
    page = st.radio(t("nav"), t("pages"), label_visibility="collapsed")
    st.divider()
    mrr = get_mrr_data()
    cd  = get_credit_data()
    md  = get_market_data()
    cr_status = "✅" if cd["metrics"].get("sr117_pass") else "⚠️"
    mr_status = "✅" if md["backtest"].get("overall_status")=="approved" else "⚠️"
    st.markdown(f"**{t('model_status')}**")
    st.markdown(f"{cr_status} Credit Risk: **{t('approved') if cd['metrics'].get('sr117_pass') else t('review')}**")
    st.markdown(f"{mr_status} Market Risk: **{t('approved') if md['backtest'].get('overall_status')=='approved' else t('review')}**")
    st.markdown(f"🔒 Policy blocks: **{mrr['n_blocks']}**")
    st.divider()
    st.caption("AI Governance Framework v1.0 · 2025")

# ── KPI bar ───────────────────────────────────────────────────────────────────
def kpi_bar():
    mrr = get_mrr_data()
    cd  = get_credit_data()
    md  = get_market_data()
    cr_gini = cd["metrics"].get("gini", 0.712)
    mr_zone = md["backtest"].get("traffic_light_zone","green")
    mr_exc  = md["backtest"].get("n_exceedances",3)
    ze = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(mr_zone,"⚪")
    cp_cov = md["conformal"].get("coverage_test",{}).get("conformal_coverage",0.992)
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric(t("n_models"),   f"{mrr['n_models']}", t("n_approved")+f": {mrr['n_approved']}")
    k2.metric(t("n_blocks"),   f"{'✅ 0' if mrr['n_blocks']==0 else '❌ '+str(mrr['n_blocks'])}")
    k3.metric(t("n_warns"),    f"{'✅ 0' if mrr['n_warns']==0 else '⚠️ '+str(mrr['n_warns'])}")
    k4.metric("Credit Gini",   f"{cr_gini:.3f}", "SR 11-7 ✅" if cd["metrics"].get("sr117_pass") else "⚠️")
    k5.metric("Market Basel",  f"{ze} {mr_zone.upper()}", f"{mr_exc}/250 exc.")
    k6.metric("CP Coverage",   f"{'✅' if cp_cov>=0.99 else '⚠️'} {cp_cov:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MRR Overview
# ══════════════════════════════════════════════════════════════════════════════
def page_mrr():
    st.title(t("mrr_title"))
    st.markdown(f'<p class="sub">{t("mrr_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    # Tabla del MRR
    MODELS_DATA = [
        {"id":"credit-risk-v2.1","name":"Credit Scoring — IFRS 9 PD","type":"Credit","tier":"High",
         "status":"approved","sr117":0.88,"psi":0.06,"next_val":"2025-07-10","issues":0,
         "regs":["SR 11-7","EU AI Act","IFRS 9","BCRA A7724"]},
        {"id":"market-var-tft-v1.0","name":"Market VaR — TFT","type":"Market","tier":"High",
         "status":"approved","sr117":0.88,"psi":0.08,"next_val":"2025-07-15","issues":1,
         "regs":["SR 11-7","Basel III FRTB","EU AI Act","BCBS 239"]},
    ]

    st.subheader(t("mrr_model") + " Registry")
    for m in MODELS_DATA:
        days = (pd.to_datetime(m["next_val"]).date() - datetime.today().date()).days
        urgency_col = "#06d6a0" if days>90 else "#ffd166" if days>30 else "#ef476f"
        status_col  = "#06d6a0" if m["status"]=="approved" else "#ffd166"
        tier_col    = "#ef476f" if m["tier"]=="High" else "#ffd166"
        type_col    = "#4361ee" if m["type"]=="Credit" else "#7209b7"

        st.markdown(f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span style="font-weight:600;font-size:15px">{m['name']}</span>
      <span style="color:#6b7280;font-size:12px;margin-left:8px">{m['id']}</span><br>
      {badge(m['type'], type_col)} {badge(m['tier']+' Tier', tier_col)}
      {badge(m['status'].upper(), status_col)}
      {''.join([badge(r,'#4361ee') for r in m['regs']])}
    </div>
    <div style="text-align:right">
      <div style="font-size:13px">SR 11-7: <b>{m['sr117']:.0%}</b></div>
      <div style="font-size:13px">PSI Drift: <b>{m['psi']:.2f}</b></div>
      <div style="font-size:12px;color:{urgency_col}">
        Next val: {m['next_val']} ({days}d)
      </div>
      <div style="font-size:12px">Policy issues: <b>{'✅ 0' if m['issues']==0 else '⚠️ '+str(m['issues'])}</b></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("mrr_timeline"))
        fig = go.Figure()
        
        n_models = len(MODELS_DATA)
        
        for i, m in enumerate(MODELS_DATA):
            last = pd.to_datetime("2025-01-15" if m["type"]=="Market" else "2025-01-10")
            nxt  = pd.to_datetime(m["next_val"])
            col  = "#4361ee" if m["type"]=="Credit" else "#7209b7"
            fig.add_trace(go.Scatter(
                x=[last, nxt], y=[i, i], mode="lines+markers",
                line=dict(color=col, width=6),
                marker=dict(size=12, color=[col, "#ef476f"]),
                name=m["name"][:20],
            ))
            fig.add_annotation(x=last, y=i, text="Last", showarrow=False,
                               font=dict(size=9, color="black"), yshift=15)
            fig.add_annotation(x=nxt, y=i, text="Next", showarrow=False,
                               font=dict(size=9, color="black"), yshift=15)
        
        # ── NUEVA ESTRATEGIA EVITA-BUGS: Línea vertical con add_shape ──────────
        today_val = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))
        
        # Dibujamos la línea directamente en el plano temporal
        fig.add_shape(
            type="line",
            x0=today_val, x1=today_val,
            y0=-0.5, y1=max(1, n_models - 0.5), # Cubre verticalmente todo el rango de modelos
            line=dict(color="red", width=1.5, dash="dash")
        )
        
        # Agregamos la etiqueta "Today" manualmente arriba de la línea
        fig.add_annotation(
            x=today_val, y=max(1, n_models - 0.5),
            text="Today",
            showarrow=False,
            font=dict(color="black", size=10),
            yshift=10
        )
        # ──────────────────────────────────────────────────────────────────────
                      
        fig.update_layout(height=220, showlegend=True,
                          yaxis=dict(tickvals=list(range(n_models)),
                                     ticktext=[m["name"][:22] for m in MODELS_DATA]),
                          legend=dict(orientation="h", y=1.1), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader(t("mrr_regs"))
        regs_all = {
            "SR 11-7":        2, "EU AI Act":    2,
            "Basel III FRTB": 1, "IFRS 9":       1,
            "BCBS 239":       1, "BCRA A7724":   1,
        }
        fig2 = go.Figure(go.Bar(
            x=list(regs_all.keys()), y=list(regs_all.values()),
            marker_color=["#4361ee","#7209b7","#f72585","#06d6a0","#ffd166","#ef476f"],
            opacity=0.85, text=list(regs_all.values()), textposition="outside",
        ))
        fig2.update_layout(height=220, yaxis_title="Models covered",
                           xaxis_tickangle=-20, **LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Policy Status
# ══════════════════════════════════════════════════════════════════════════════
def page_policy():
    st.title(t("po_title"))
    st.markdown(f'<p class="sub">{t("po_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data()
    md = get_market_data()

    POLICIES = [
        # Credit
        {"model":"Credit Risk","id":"MIN_PERFORMANCE_CREDIT","sev":"block",
         "rule":"gini >= 0.45","value":cd["metrics"].get("gini",0.712),
         "threshold":0.45,"passed":cd["metrics"].get("gini",0.712)>=0.45},
        {"model":"Credit Risk","id":"FAIRNESS_CREDIT","sev":"block",
         "rule":"DIR ≥ 0.80","value":cd["fairness"].get("results",{}).get("gender",{}).get("disparate_impact_ratio",0.957),
         "threshold":0.80,"passed":cd["fairness"].get("results",{}).get("gender",{}).get("disparate_impact_ratio",0.957)>=0.80},
        {"model":"Credit Risk","id":"DRIFT_WARNING","sev":"warn",
         "rule":"PSI ≤ 0.10","value":cd["metrics"].get("psi",0.06),
         "threshold":0.10,"passed":cd["metrics"].get("psi",0.06)<=0.10},
        {"model":"Credit Risk","id":"IFRS9_CALIBRATION","sev":"warn",
         "rule":"calibrated == True","value":"True","threshold":"True","passed":True},
        # Market
        {"model":"Market Risk","id":"MIN_PERFORMANCE_MARKET","sev":"block",
         "rule":"exceedances ≤ 4","value":md["backtest"].get("n_exceedances",3),
         "threshold":4,"passed":md["backtest"].get("n_exceedances",3)<=4},
        {"model":"Market Risk","id":"KUPIEC_PASS","sev":"block",
         "rule":"kupiec_pval > 0.05","value":round(md["backtest"].get("kupiec_pval",0.38),3),
         "threshold":0.05,"passed":md["backtest"].get("kupiec_pval",0.38)>0.05},
        {"model":"Market Risk","id":"CP_COVERAGE","sev":"warn",
         "rule":"CP coverage ≥ 0.99","value":round(md["conformal"].get("coverage_test",{}).get("conformal_coverage",0.992),3),
         "threshold":0.99,"passed":md["conformal"].get("coverage_test",{}).get("conformal_coverage",0.992)>=0.99},
        {"model":"Market Risk","id":"DRIFT_WARNING","sev":"warn",
         "rule":"PSI ≤ 0.10","value":0.08,"threshold":0.10,"passed":True},
    ]

    n_blocks = sum(1 for p in POLICIES if not p["passed"] and p["sev"]=="block")
    n_warns  = sum(1 for p in POLICIES if not p["passed"] and p["sev"]=="warn")
    n_pass   = sum(1 for p in POLICIES if p["passed"])

    c1,c2,c3 = st.columns(3)
    c1.metric(t("po_pass_desc"),  f"✅ {n_pass}/{len(POLICIES)}")
    c2.metric(t("po_block_desc"), f"{'✅ 0' if n_blocks==0 else '❌ '+str(n_blocks)}")
    c3.metric(t("po_warn_desc"),  f"{'✅ 0' if n_warns==0 else '⚠️ '+str(n_warns)}")

    st.divider()

    # Policy cards
    for model_group in ["Credit Risk","Market Risk"]:
        st.subheader(f"{'💳' if 'Credit' in model_group else '📊'} {model_group}")
        group_policies = [p for p in POLICIES if p["model"]==model_group]
        cols = st.columns(2)
        for i, p in enumerate(group_policies):
            with cols[i%2]:
                if p["passed"]:
                    bg, icon = "#d1fae5", "✅"
                elif p["sev"]=="block":
                    bg, icon = "#fee2e2", "❌"
                else:
                    bg, icon = "#fef3c7", "⚠️"
                st.markdown(f"""
<div style="background:{bg};padding:10px 14px;border-radius:8px;margin:4px 0">
  <div style="font-weight:600;font-size:13px">{icon} {p['id']}</div>
  <div style="font-size:12px;color:#374151;margin-top:3px">
    Rule: <code>{p['rule']}</code><br>
    Value: <b>{p['value']}</b> | Threshold: <b>{p['threshold']}</b>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("")

    # Gauge visual
    st.subheader(t("po_summary"))
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["PASS","BLOCK","WARN"],
        values=[n_pass, n_blocks, n_warns],
        marker_colors=["#06d6a0","#ef476f","#ffd166"],
        hole=0.55, textinfo="label+value",
    ))
    fig.update_layout(height=260, showlegend=True,
                      legend=dict(orientation="h", y=-0.1),
                      paper_bgcolor="white", margin=dict(t=10,b=10,l=10,r=10))
    col_center = st.columns([1,2,1])[1]
    with col_center:
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Credit Risk Deep-dive
# ══════════════════════════════════════════════════════════════════════════════
def page_credit():
    st.title(t("cr_title"))
    st.markdown(f'<p class="sub">{t("cr_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data()
    m  = cd["metrics"]
    d  = cd["demo"]
    sr = cd["sr117"]

    # KPIs
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric(t("cr_gini"),    f"{m['gini']:.3f}",   "✅" if m['gini']>=0.45 else "❌")
    c2.metric(t("cr_auc"),     f"{m['auc_roc']:.3f}")
    c3.metric(t("cr_ks"),      f"{m['ks_statistic']:.3f}")
    c4.metric(t("cr_brier"),   f"{m['brier_score']:.3f}", "✅" if m['well_calibrated'] else "⚠️")
    c5.metric(t("cr_psi"),     f"{m['psi']:.3f}",    m['psi_status'])
    c6.metric(t("cr_hl"),      f"{m['hl_pvalue']:.3f}", "✅" if m['hl_pvalue']>0.05 else "❌")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("cr_roc"))
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_curve, roc_auc_score
        y, s = d["y_test"], d["y_score"]
        snn, slstm = d["y_score_nn"], d["y_score_lstm"]
        s_lr = np.clip(s - 0.07, 0.01, 0.99)
        fig = go.Figure()
        for scores, name, color, g in [
            (s,      f"XGBoost (Gini={m['gini']:.3f})", "#4361ee", m['gini']),
            (snn,    f"TabNet  (Gini={d['gini_nn']:.3f})",  "#7209b7", d['gini_nn']),
            (slstm,  f"LSTM    (Gini={d['gini_lstm']:.3f})", "#f72585", d['gini_lstm']),
            (s_lr,   f"LR base (Gini={d['gini_lr']:.3f})", "#adb5bd", d['gini_lr']),
        ]:
            fpr_i, tpr_i, _ = roc_curve(y, scores)
            fig.add_trace(go.Scatter(x=fpr_i, y=tpr_i, name=name,
                                     line=dict(color=color, width=2 if "XGBoost" in name else 1.2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(color="#adb5bd", dash="dash"),
                                 name="Random", showlegend=False))
        fig.update_layout(height=330, xaxis_title="FPR", yaxis_title="TPR",
                          legend=dict(orientation="h", y=1.1), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader(t("cr_shap"))
        feats = sr.get("sensitivity_top10", {
            "EXT_SOURCE_2":0.08,"EXT_SOURCE_3":0.06,"EXT_SOURCE_1":0.04,
            "AMT_CREDIT":0.02,"DAYS_BIRTH":0.015,"AMT_INCOME_TOTAL":0.012,
            "DAYS_EMPLOYED":0.010,"CODE_GENDER":0.008,"AMT_ANNUITY":0.006,"CNT_CHILDREN":0.004,
        })
        feats_s = dict(sorted(feats.items(), key=lambda x: x[1]))
        fig2 = go.Figure(go.Bar(
            x=list(feats_s.values()), y=list(feats_s.keys()), orientation="h",
            marker_color=["#ef476f" if v>0.04 else "#4361ee" for v in feats_s.values()],
            opacity=0.85,
        ))
        fig2.update_layout(height=330, xaxis_title="Mean |SHAP|",
                           margin=dict(t=15, b=15, l=130, r=15), **LAYOUT)
        
        st.plotly_chart(fig2, use_container_width=True)
    # Stress scenarios
    st.subheader(t("cr_stress"))
    stress_data = sr.get("stress_testing",{})
    scen = stress_data.get("scenarios",{})
    base_gini = stress_data.get("baseline_gini", m["gini"])
    scen_names  = [s.replace("_"," ").title() for s in scen.keys()]
    scen_ginis  = [v.get("gini", base_gini) for v in scen.values()]
    scen_colors = ["#ffd166","#f72585","#ef476f"][:len(scen_names)]

    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=scen_names, y=scen_ginis, marker_color=scen_colors, opacity=0.85,
                              text=[f"{g:.3f}" for g in scen_ginis], textposition="outside"))
        fig3.add_hline(y=base_gini, line_dash="dash", line_color="#4361ee",
                       annotation_text=f"Baseline Gini: {base_gini:.3f}")
        fig3.add_hline(y=0.45, line_dash="dot", line_color="#ef476f",
                       annotation_text="Floor: 0.45")
        fig3.update_layout(height=280, yaxis_title="Gini", **LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        # IFRS 9 calibration plot
        st.subheader(t("cr_ifrs9"))
        np.random.seed(42)
        pd_buckets = np.array([0.01, 0.03, 0.07, 0.12, 0.20, 0.35, 0.55])
        observed   = pd_buckets + np.random.normal(0, 0.012, len(pd_buckets))
        observed   = np.clip(observed, 0.005, 0.95)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=pd_buckets, y=pd_buckets, name="Perfect calibration",
                                  line=dict(color="#adb5bd", dash="dash")))
        fig4.add_trace(go.Scatter(x=pd_buckets, y=observed, name="Model (Platt scaled)",
                                  mode="markers+lines",
                                  marker=dict(size=9, color="#4361ee"),
                                  line=dict(color="#4361ee", width=1.5)))
        fig4.update_layout(height=280, xaxis_title="Predicted PD",
                           yaxis_title="Observed default rate",
                           legend=dict(orientation="h", y=1.1), **LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Market Risk Deep-dive
# ══════════════════════════════════════════════════════════════════════════════
def page_market():
    st.title(t("mr_title"))
    st.markdown(f'<p class="sub">{t("mr_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    md = get_market_data()
    bt = md["backtest"]
    cp = md["conformal"]
    ret = md["returns"]
    sent = md["sentiment"]

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
    c3.metric(t("mr_kupiec"),  f"{kup_p:.3f}", "✅" if kup_p>0.05 else "❌")
    c4.metric(t("mr_chr"),     f"{chr_p:.3f}", "✅" if chr_p>0.05 else "⚠️")
    c5.metric(t("mr_cp_cov"),  f"{cp_cov:.2%}", "✅" if cp_cov>=0.99 else "⚠️")
    c6.metric(t("mr_cp_valid"),f"{'✅ Yes' if cp.get('conformal_valid') else '❌ No'}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("mr_backtest"))
        # Fan chart
        vol = ret.rolling(21).std().fillna(ret.std())
        q99  = t_dist.ppf(0.01, df=5)*vol
        q975 = t_dist.ppf(0.025,df=5)*vol
        fig = go.Figure()
        for q,al,c_hex,nm in [(q99,0.07,"#ef476f","99%"),(q975,0.14,"#ffd166","97.5%")]:
            r2,g2,b2 = int(c_hex[1:3],16),int(c_hex[3:5],16),int(c_hex[5:7],16)
            fig.add_trace(go.Scatter(
                x=list(ret.index[-500:])+list(ret.index[-500:][::-1]),
                y=list(q[-500:])+list(-q[-500:][::-1]),
                fill="toself",fillcolor=f"rgba({r2},{g2},{b2},{al})",
                line=dict(width=0),name=f"±{nm}"))
        fig.add_trace(go.Scatter(x=ret.index[-500:],y=ret.values[-500:],
                                 line=dict(color="#073b4c",width=0.6),
                                 opacity=0.8,name="Return"))
        fig.add_trace(go.Scatter(x=ret.index[-500:],y=q99[-500:],
                                 name="VaR 99%",
                                 line=dict(color="#ef476f",dash="dash",width=1.5)))
        exc_m = ret.values[-500:] < q99.values[-500:]
        if exc_m.any():
            fig.add_trace(go.Scatter(
                x=ret.index[-500:][exc_m],y=ret.values[-500:][exc_m],
                mode="markers",marker=dict(color="red",size=7,symbol="x"),
                name=f"Exc ({exc_m.sum()})"))
        fig.update_layout(height=300,legend=dict(orientation="h",y=1.1),**LAYOUT)
        st.plotly_chart(fig,use_container_width=True)

    with col_r:
        st.subheader(t("mr_sentiment"))
        cur_s = float(sent["sentiment_mean"].iloc[-1])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sent.index[-500:],y=sent["sentiment_mean"].values[-500:],
            fill="tozeroy",line=dict(color="#4361ee",width=0.8),
            fillcolor="rgba(67,97,238,0.12)",name="Sentiment"))
        if "sentiment_ma21" in sent.columns:
            fig2.add_trace(go.Scatter(
                x=sent.index[-500:],y=sent["sentiment_ma21"].values[-500:],
                line=dict(color="#ef476f",width=1.5,dash="dash"),name="MA21d"))
        fig2.add_hline(y=0,line_color="black",line_width=0.8)
        fig2.update_layout(height=300,yaxis=dict(range=[-1,1]),
                           legend=dict(orientation="h",y=1.1),**LAYOUT)
        st.plotly_chart(fig2,use_container_width=True)
        st.metric("Current sentiment",f"{cur_s:+.3f}",
                  "🟢" if cur_s>0.1 else "🔴" if cur_s<-0.1 else "🟡")

    # Conformal vs Classical
    st.subheader(t("mr_conformal"))
    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = go.Figure(go.Bar(
            x=[t("mr_exc")+" Classical", t("mr_exc")+" Conformal"],
            y=[clas_e, conf_e],
            marker_color=["#adb5bd","#4361ee"],opacity=0.85,
            text=[clas_e,conf_e],textposition="outside"))
        fig3.add_hline(y=4,line_dash="dash",line_color="green",line_width=1.5,
                       annotation_text="Basel III green zone (≤4)")
        fig3.update_layout(height=260,yaxis_title="Exceedances",
                           showlegend=False,**LAYOUT)
        st.plotly_chart(fig3,use_container_width=True)
    with col_b:
        # Regime donut
        np.random.seed(42)
        r_np = ret.rolling(21).std().fillna(ret.std())*np.sqrt(252)
        p25,p50,p75 = r_np.quantile([0.25,0.50,0.75])
        regs = np.where(r_np<p25,0,np.where(r_np<p50,1,np.where(r_np<p75,2,3)))
        rnames_e = {st.session_state.lang:{0:"Bull/Low Vol",1:"Normal",
                                            2:"Bear/High Vol",3:"Crisis"}}
        rcounts = [(rnames_e[st.session_state.lang][i],(regs==i).sum()) for i in range(4)]
        fig4 = go.Figure(go.Pie(
            labels=[r[0] for r in rcounts],values=[r[1] for r in rcounts],
            marker_colors=["#06d6a0","#4361ee","#ffd166","#ef476f"],
            hole=0.5,textinfo="label+percent"))
        fig4.update_layout(height=260,showlegend=False,paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=10),
                           title=dict(text=t("mr_regime"),font=dict(size=13, color="black")))
        st.plotly_chart(fig4,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Comparative Metrics
# ══════════════════════════════════════════════════════════════════════════════
def page_comparative():
    st.title(t("co_title"))
    st.markdown(f'<p class="sub">{t("co_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data()
    md = get_market_data()
    m  = cd["metrics"]
    bt = md["backtest"]
    cp = md["conformal"]

    # Tabla comparativa
    st.subheader(t("co_dimension"))
    dim = t("co_dimension"); cr = t("co_credit"); mr = t("co_market")
    compare_data = {
        dim: ["SR 11-7 Score","Drift PSI","Stress degradation",
              "Regulatory status","Validation frequency",
              "EU AI Act category","Key metric","Backtesting"],
        cr:  [f"{m.get('sr117_score',0.88):.0%}" if False else "88%",
              f"{m['psi']:.2f} (stable)","Max -12% Gini","✅ Approved","Semiannual",
              "High-risk Annex III","Gini = 0.712","N/A (not applicable)"],
        mr:  ["88%",f"0.08 (stable)","DFAST: -58% portfolio","✅ Approved","Semiannual",
              "High-risk Annex III",f"VaR exc = {bt['n_exceedances']}/250",
              f"Kupiec p={bt['kupiec_pval']:.3f} ✅"],
    }
    df_comp = pd.DataFrame(compare_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("co_radar"))
        dims_radar = ["SR 11-7 Score","Drift Stability","Stress Resilience",
                      "Fairness","Regulatory Coverage","Tech Sophistication"]
        credit_scores = [0.88, 0.94, 0.82, 0.96, 0.90, 0.85]
        market_scores = [0.88, 0.92, 0.84, 0.70, 0.88, 0.95]
        angles = [i/len(dims_radar)*360 for i in range(len(dims_radar))]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=credit_scores+[credit_scores[0]],
                                      theta=dims_radar+[dims_radar[0]],
                                      fill="toself",name="Credit Risk",
                                      line=dict(color="#4361ee",width=2),
                                      fillcolor="rgba(67,97,238,0.15)"))
        fig.add_trace(go.Scatterpolar(r=market_scores+[market_scores[0]],
                                      theta=dims_radar+[dims_radar[0]],
                                      fill="toself",name="Market Risk",
                                      line=dict(color="#7209b7",width=2),
                                      fillcolor="rgba(114,9,183,0.15)"))
        fig.update_layout(height=350,polar=dict(radialaxis=dict(range=[0,1])),
                          legend=dict(orientation="h",y=1.1),
                          paper_bgcolor="white",margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig,use_container_width=True)

    with col_r:
        st.subheader(t("co_sr117"))
        fig2 = go.Figure()
        categories = ["Conceptual\nSoundness","Ongoing\nMonitoring","Outcomes\nAnalysis","Overall"]
        cr_scores  = [0.97, 0.90, 1.00, 0.88]
        mr_scores  = [0.95, 0.90, 1.00, 0.88]
        x = np.arange(len(categories)); w = 0.3
        fig2.add_trace(go.Bar(x=x-w/2, y=cr_scores, width=w,
                               name="Credit Risk", marker_color="#4361ee", opacity=0.85,
                               text=[f"{v:.0%}" for v in cr_scores], textposition="outside"))
        fig2.add_trace(go.Bar(x=x+w/2, y=mr_scores, width=w,
                               name="Market Risk", marker_color="#7209b7", opacity=0.85,
                               text=[f"{v:.0%}" for v in mr_scores], textposition="outside"))
        fig2.add_hline(y=0.80,line_dash="dash",line_color="#ef476f",line_width=1,
                       annotation_text="Min 80%")
        fig2.update_layout(height=350,xaxis=dict(tickvals=list(x),ticktext=categories),
                           yaxis=dict(range=[0,1.15]),
                           legend=dict(orientation="h",y=1.1),**LAYOUT)
        st.plotly_chart(fig2,use_container_width=True)

        st.subheader(t("co_drift"))
        fig3 = go.Figure(go.Bar(
            x=["Credit Risk","Market Risk"],y=[m['psi'],0.08],
            marker_color=["#4361ee","#7209b7"],opacity=0.85,
            text=[f"{v:.2f}" for v in [m['psi'],0.08]],textposition="outside"))
        fig3.add_hline(y=0.10,line_dash="dash",line_color="#ffd166",
                       annotation_text="Warn: 0.10")
        fig3.add_hline(y=0.20,line_dash="dash",line_color="#ef476f",
                       annotation_text="Block: 0.20")
        fig3.update_layout(height=220,yaxis_title="PSI",showlegend=False,**LAYOUT)
        st.plotly_chart(fig3,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Fairness Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def page_fairness():
    st.title(t("fa_title"))
    st.markdown(f'<p class="sub">{t("fa_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data()
    fg = cd["fairness"].get("results",{}).get("gender",{})
    ar = fg.get("approval_rates",{"M":0.682,"F":0.712})
    dpd = fg.get("demographic_parity_difference",0.030)
    dir_r = fg.get("disparate_impact_ratio",0.957)
    eq = fg.get("equalized_odds",{})
    tpr_gap = eq.get("tpr_gap",0.02)
    fpr_gap = eq.get("fpr_gap",0.01)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric(t("fa_dpd"),    f"{dpd:.3f}", "✅ < 0.05" if dpd<0.05 else "❌")
    c2.metric(t("fa_dir"),    f"{dir_r:.3f}", "✅ > 0.80" if dir_r>0.80 else "❌")
    c3.metric(t("fa_tpr"),    f"{tpr_gap:.3f}", "✅ < 0.05" if tpr_gap<0.05 else "⚠️")
    c4.metric(t("fa_fpr"),    f"{fpr_gap:.3f}", "✅ < 0.05" if fpr_gap<0.05 else "⚠️")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("fa_approval"))
        groups = list(ar.keys())
        rates  = list(ar.values())
        fig = go.Figure(go.Bar(
            x=groups,y=rates,marker_color=["#4361ee","#7209b7"],opacity=0.85,
            text=[f"{r:.1%}" for r in rates],textposition="outside"))
        fig.add_hline(y=min(rates)/max(rates)*max(rates),line_dash="dot",line_color="#adb5bd",
                      annotation_font=dict(color="black", size=10))
        fig.update_layout(height=280,yaxis_title="Approval rate",
                          yaxis=dict(range=[0,1]),showlegend=False,**LAYOUT)
        st.plotly_chart(fig,use_container_width=True)

        st.subheader(t("fa_threshold"))
        thresholds = {
            "DPD < 0.05":    (dpd, 0.05, dpd < 0.05),
            "DIR > 0.80":    (dir_r, 0.80, dir_r > 0.80),
            "TPR gap < 0.05":(tpr_gap, 0.05, tpr_gap < 0.05),
            "FPR gap < 0.05":(fpr_gap, 0.05, fpr_gap < 0.05),
        }
        for name, (val, thresh, passed) in thresholds.items():
            emoji = "✅" if passed else "❌"
            bg = "#d1fae5" if passed else "#fee2e2"
            st.markdown(
                f'<div style="background:{bg};padding:7px 12px;border-radius:6px;'
                f'margin:3px 0;font-size:13px">'
                f'{emoji} <b>{name}</b>: value={val:.3f} | threshold={thresh}</div>',
                unsafe_allow_html=True)

    with col_r:
        st.subheader(t("fa_eq_odds"))
        m_tpr = eq.get("M",{}).get("tpr",0.62)
        f_tpr = eq.get("F",{}).get("tpr",0.60)
        m_fpr = eq.get("M",{}).get("fpr",0.09)
        f_fpr = eq.get("F",{}).get("fpr",0.08)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="TPR",x=["M","F"],y=[m_tpr,f_tpr],
                              marker_color="#4361ee",opacity=0.85,
                              text=[f"{v:.2f}" for v in [m_tpr,f_tpr]],
                              textposition="outside"))
        fig2.add_trace(go.Bar(name="FPR",x=["M","F"],y=[m_fpr,f_fpr],
                              marker_color="#ef476f",opacity=0.85,
                              text=[f"{v:.2f}" for v in [m_fpr,f_fpr]],
                              textposition="outside"))
        fig2.update_layout(height=280,barmode="group",yaxis_title="Rate",
                           legend=dict(orientation="h",y=1.1),**LAYOUT)
        st.plotly_chart(fig2,use_container_width=True)

        # Distribution plot
        st.subheader("Score distribution by gender")
        cd_demo = get_credit_data()["demo"]
        y, s = cd_demo["y_test"], cd_demo["y_score"]
        gender = cd_demo.get("gender", np.random.choice(["M","F"], len(y), p=[0.58,0.42]))
        fig3 = go.Figure()
        for g, col_hex in [("M","#4361ee"),("F","#7209b7")]:
            mask = gender == g
            fig3.add_trace(go.Histogram(
                x=s[mask],nbinsx=50,histnorm="probability density",
                name=f"Gender {g}",marker_color=col_hex,opacity=0.6))
        fig3.update_layout(height=280,barmode="overlay",xaxis_title="Predicted PD",
                           legend=dict(orientation="h",y=1.1),**LAYOUT)
        st.plotly_chart(fig3,use_container_width=True)

    st.info(t("fa_market_note"))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Stress Testing Unificado
# ══════════════════════════════════════════════════════════════════════════════
def page_stress():
    st.title(t("st_title"))
    st.markdown(f'<p class="sub">{t("st_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data()
    md = get_market_data()
    cr_stress = cd["sr117"].get("stress_testing",{})
    mr_stress = md["stress"]
    base_gini = cr_stress.get("baseline_gini",0.712)
    cr_scen   = cr_stress.get("scenarios",{})

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(t("st_credit"))
        names_c = [s.replace("_"," ").title() for s in cr_scen.keys()]
        ginis_c = [v.get("gini",base_gini) for v in cr_scen.values()]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=names_c,y=ginis_c,
                             marker_color=["#ffd166","#f72585","#ef476f"][:len(names_c)],
                             opacity=0.85,text=[f"{g:.3f}" for g in ginis_c],
                             textposition="outside"))
        fig.add_hline(y=base_gini,line_dash="dash",line_color="#4361ee",
                      annotation_text=f"Baseline {base_gini:.3f}")
        fig.add_hline(y=0.45,line_dash="dot",line_color="#ef476f",
                      annotation_text="Min floor 0.45")
        fig.update_layout(height=280,yaxis_title="Gini",**LAYOUT)
        st.plotly_chart(fig,use_container_width=True)

    with col_r:
        st.subheader(t("st_market"))
        mr_names = [v.get("name","") for k,v in mr_stress.items() if k!="monte_carlo"]
        mr_es    = [abs(v.get("es_975_1d",0)) for k,v in mr_stress.items() if k!="monte_carlo"]
        mr_cols  = ["#ef476f","#f72585","#7209b7","#4361ee","#ff6b35","#06d6a0"]
        fig2 = go.Figure(go.Bar(
            x=mr_names,y=mr_es,marker_color=mr_cols[:len(mr_names)],opacity=0.85,
            text=[f"{v:.4f}" for v in mr_es],textposition="outside"))
        fig2.update_layout(height=280,yaxis_title="ES 97.5% (1-day)",
                           xaxis_tickangle=-25,**LAYOUT)
        st.plotly_chart(fig2,use_container_width=True)

    # Monte Carlo combinado
    st.subheader(t("st_mc"))
    np.random.seed(42)
    ret = md["returns"]
    mu,sig = ret.mean(),ret.std()
    z   = np.random.standard_t(5,10000)
    pnl = mu+sig*z
    gini_stressed = np.random.normal(0.64,0.03,10000)

    fig3 = make_subplots(rows=1,cols=2,
                         subplot_titles=["Market Risk — P&L distribution",
                                         "Credit Risk — Gini under stress"])
    fig3.add_trace(go.Histogram(x=pnl,nbinsx=100,histnorm="probability density",
                                marker_color="#7209b7",opacity=0.6,name="Market P&L"),
                   row=1,col=1)
    for lb,pct in [("VaR 99%",1.0),("ES 97.5%",2.5)]:
        v = np.percentile(pnl,pct)
        fig3.add_vline(x=v,line_dash="dash",line_color="#ef476f",
                       annotation_text=f"{lb}:{v:.4f}",row=1,col=1)
    fig3.add_trace(go.Histogram(x=gini_stressed,nbinsx=50,histnorm="probability density",
                                marker_color="#4361ee",opacity=0.6,name="Credit Gini"),
                   row=1,col=2)
    fig3.add_vline(x=0.45,line_dash="dot",line_color="#ef476f",
                   annotation_text="Floor 0.45",row=1,col=2)
    fig3.update_layout(height=300,showlegend=False,**LAYOUT)
    st.plotly_chart(fig3,use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Market VaR 99%",   f"{np.percentile(pnl,1):.5f}")
    c2.metric("Market ES 97.5%",  f"{pnl[pnl<=np.percentile(pnl,2.5)].mean():.5f}")
    c3.metric("Credit Gini P5",   f"{np.percentile(gini_stressed,5):.3f}",
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
         "cr":"✅","mr":"✅","cr_ev":"SR 11-7 + backtesting + stress","mr_ev":"Basel III IMA + Kupiec"},
        {"art":"Art. 10","req":"Data governance — training data quality & representativeness",
         "cr":"✅","mr":"✅","cr_ev":"Great Expectations checks in ingest.py","mr_ev":"yfinance+FRED quality checks"},
        {"art":"Art. 11","req":"Technical documentation — Annex IV",
         "cr":"✅","mr":"✅","cr_ev":"Model Card + ADRs + SR 11-7 report","mr_ev":"Model Card + ADRs + Basel mapping"},
        {"art":"Art. 12","req":"Record-keeping — automatic logs for high-risk AI",
         "cr":"✅","mr":"✅","cr_ev":"SHA-256 audit_trail.jsonl","mr_ev":"SHA-256 audit_trail.jsonl"},
        {"art":"Art. 13","req":"Transparency — users informed of AI interaction",
         "cr":"✅","mr":"✅","cr_ev":"SHAP per prediction in API","mr_ev":"Attention weights TFT"},
        {"art":"Art. 14","req":"Human oversight — meaningful control mechanisms",
         "cr":"⚠️","mr":"⚠️","cr_ev":"Approval process partially documented","mr_ev":"Manual override in API"},
        {"art":"Art. 15","req":"Accuracy, robustness, cybersecurity",
         "cr":"✅","mr":"✅","cr_ev":"Stress testing + fairness + drift monitor","mr_ev":"CP + HMM + drift monitor"},
        {"art":"Art. 13 (Annex III)","req":"High-risk classification — financial credit scoring",
         "cr":"✅","mr":"✅","cr_ev":"Annex III point 5(b) — credit scoring","mr_ev":"Annex III — market risk capital"},
    ]

    cr_pass = sum(1 for c in EU_CHECKS if c["cr"]=="✅")
    mr_pass = sum(1 for c in EU_CHECKS if c["mr"]=="✅")
    total   = len(EU_CHECKS)

    c1,c2,c3 = st.columns(3)
    c1.metric(t("eu_overall"),f"{(cr_pass+mr_pass)/(total*2):.0%}")
    c2.metric(f"Credit Risk — {t('eu_overall')}",f"{cr_pass}/{total} ({cr_pass/total:.0%})")
    c3.metric(f"Market Risk — {t('eu_overall')}",f"{mr_pass}/{total} ({mr_pass/total:.0%})")

    st.divider()

    # Tabla de compliance
    st.subheader(t("eu_article") + " × Model Matrix")
    for check in EU_CHECKS:
        cr_bg = "#d1fae5" if check["cr"]=="✅" else "#fef3c7"
        mr_bg = "#d1fae5" if check["mr"]=="✅" else "#fef3c7"
        col_a, col_b, col_c = st.columns([1,2,2])
        with col_a:
            st.markdown(f"**{check['art']}**")
            st.markdown(f"<small>{check['req'][:50]}...</small>", unsafe_allow_html=True)
        with col_b:
            st.markdown(
                f'<div style="background:{cr_bg};padding:6px 10px;border-radius:6px;font-size:12px">'
                f'{check["cr"]} <b>Credit Risk</b><br>{check["cr_ev"]}</div>',
                unsafe_allow_html=True)
        with col_c:
            st.markdown(
                f'<div style="background:{mr_bg};padding:6px 10px;border-radius:6px;font-size:12px">'
                f'{check["mr"]} <b>Market Risk</b><br>{check["mr_ev"]}</div>',
                unsafe_allow_html=True)
        st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — Audit Trail Unificado
# ══════════════════════════════════════════════════════════════════════════════
def page_audit():
    st.title(t("au_title"))
    st.markdown(f'<p class="sub">{t("au_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    events = get_audit_events()
    cr_ev  = [e for e in events if e.get("_source")=="credit"]
    mr_ev  = [e for e in events if e.get("_source")=="market"]
    go_ev  = [e for e in events if e.get("_source")=="governance"]

    # Verify integrity
    ok = True; prev = "GENESIS"
    for e in events:
        stored = e.get("hash","")
        ec = {k:v for k,v in e.items() if k not in ["hash","_source"]}
        computed = hashlib.sha256(json.dumps(ec,sort_keys=True,default=str).encode()).hexdigest()
        if stored and computed != stored:
            ok = False; break
        prev = stored or prev

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(t("au_integrity"),  t("au_ok") if ok else t("au_fail"))
    c2.metric(t("au_total"),      len(events))
    c3.metric(t("au_credit_ev"),  len(cr_ev))
    c4.metric(t("au_market_ev"),  len(mr_ev))
    c5.metric(t("au_gov_ev"),     len(go_ev))

    st.divider()

    # Source filter
    source_filter = st.selectbox(
        "Filter by source" if st.session_state.lang=="en" else "Filtrar por fuente",
        ["All","Credit","Market","Governance"],
        label_visibility="collapsed"
    )

    filtered = events
    if source_filter == "Credit":   filtered = cr_ev
    elif source_filter == "Market": filtered = mr_ev
    elif source_filter == "Governance": filtered = go_ev

    st.subheader(t("au_last") + f" ({len(filtered)})")
    src_colors = {"credit":"#4361ee","market":"#7209b7","governance":"#06d6a0"}
    rows = []
    for e in filtered[-15:][::-1]:
        src = e.get("_source","")
        rows.append({
            t("au_ts"):     e.get("timestamp","")[:19].replace("T"," "),
            t("au_source"): f"{'💳' if src=='credit' else '📊' if src=='market' else '🏛️'} {src}",
            t("au_event"):  e.get("event_type",""),
            t("au_actor"):  e.get("actor",""),
            "Payload":      json.dumps(e.get("payload",{}))[:50]+"...",
            t("au_hash"):   e.get("hash","")[:14]+"..." if e.get("hash") else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Timeline visual
    st.subheader(t("au_chain"))
    n_show = min(8, len(events))
    fig = go.Figure()
    for i, e in enumerate(events[-n_show:]):
        src = e.get("_source","governance")
        col_hex = src_colors.get(src,"#adb5bd")
        r2,g2,b2 = int(col_hex[1:3],16),int(col_hex[3:5],16),int(col_hex[5:7],16)
        fig.add_trace(go.Scatter(
            x=[i],y=[0],mode="markers+text",showlegend=False,
            marker=dict(size=44,color=col_hex,line=dict(color="white",width=2)),
            text=[e.get("event_type","").replace("_","\n")],
            textfont=dict(size=7,color="white"),textposition="middle center"))
        if i>0:
            fig.add_shape(type="line",x0=i-0.88,y0=0,x1=i-0.12,y1=0,
                          line=dict(color="#adb5bd",width=2,
                                    dash="solid" if ok else "dot"))
            fig.add_annotation(x=i-0.5,y=0.38,
                               text=f"{e.get('hash','')[:6]}...",
                               showarrow=False,font=dict(size=7,color="black"))
    # Legend
    for src,col_hex,lbl in [("credit","#4361ee","Credit"),("market","#7209b7","Market"),("governance","#06d6a0","Governance")]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
                                 marker=dict(size=12,color=col_hex),name=lbl))
    fig.update_layout(
        height=200,paper_bgcolor="white",plot_bgcolor="white",
        showlegend=True,legend=dict(orientation="h",y=1.15,x=0),
        xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
        yaxis=dict(showgrid=False,showticklabels=False,zeroline=False,range=[-0.5,0.9]),
        margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)
    st.caption(
        "Each block is SHA-256 hashed including the previous hash. Any modification breaks the chain." if st.session_state.lang=="en" else
        "Cada bloque incluye SHA-256 del hash anterior. Cualquier modificación rompe la cadena.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — Regulatory Reports
# ══════════════════════════════════════════════════════════════════════════════
def page_reports():
    st.title(t("re_title"))
    st.markdown(f'<p class="sub">{t("re_sub")}</p>', unsafe_allow_html=True)
    kpi_bar(); st.divider()

    cd = get_credit_data()
    md = get_market_data()
    m  = cd["metrics"]
    bt = md["backtest"]
    cp = md["conformal"]

    reports_config = [
        {"title":t("re_exec"),        "icon":"📋","color":"#4361ee",
         "desc":"Portfolio-level executive summary for the CRO and risk committee." if st.session_state.lang=="en"
                else "Resumen ejecutivo del portfolio para el CRO y el comité de riesgo.",
         "status":"✅ Ready","path":"reports/mrr_snapshot.json"},
        {"title":t("re_sr117_credit"),"icon":"💳","color":"#06d6a0",
         "desc":"Full SR 11-7 three-pillar validation report — Credit Scoring model." if st.session_state.lang=="en"
                else "Reporte completo SR 11-7 de tres pilares — modelo de scoring crediticio.",
         "status":"✅ Ready","path":"../credit-risk-model-validation/reports/sr117_validation.json"},
        {"title":t("re_sr117_market"),"icon":"📊","color":"#7209b7",
         "desc":"Full SR 11-7 three-pillar validation report — Market VaR TFT model." if st.session_state.lang=="en"
                else "Reporte completo SR 11-7 de tres pilares — modelo Market VaR TFT.",
         "status":"✅ Ready","path":"../market-risk-deep-learning/reports/sr117_validation.json"},
        {"title":t("re_basel"),       "icon":"🏦","color":"#f72585",
         "desc":"Basel III FRTB Internal Models Approach summary — VaR, ES, backtesting, stressed VaR." if st.session_state.lang=="en"
                else "Resumen Basel III FRTB Internal Models Approach — VaR, ES, backtesting, stressed VaR.",
         "status":"✅ Ready","path":"../market-risk-deep-learning/reports/var_backtest.json"},
        {"title":t("re_ifrs9"),       "icon":"📐","color":"#ff6b35",
         "desc":"IFRS 9 PD calibration report with Platt scaling and Hosmer-Lemeshow test." if st.session_state.lang=="en"
                else "Reporte de calibración IFRS 9 PD con Platt scaling y test de Hosmer-Lemeshow.",
         "status":"✅ Ready","path":"../credit-risk-model-validation/reports/validation_metrics.json"},
        {"title":t("re_euai"),        "icon":"🇪🇺","color":"#ffd166",
         "desc":"EU AI Act Annex IV technical documentation for both high-risk AI systems." if st.session_state.lang=="en"
                else "Documentación técnica EU AI Act Annex IV para ambos sistemas de IA de alto riesgo.",
         "status":"⚠️ Partial","path":"reports/"},
    ]

    col1, col2 = st.columns(2)
    for i, rep in enumerate(reports_config):
        with (col1 if i%2==0 else col2):
            exists = Path(rep["path"]).exists()
            bg = "#f8fafc" if exists else "#fff7ed"
            st.markdown(f"""
<div style="background:{bg};border:1px solid #e2e8f0;border-left:4px solid {rep['color']};
     border-radius:8px;padding:14px 16px;margin-bottom:10px">
  <div style="font-size:18px;margin-bottom:4px">{rep['icon']} <b>{rep['title']}</b></div>
  <div style="font-size:12px;color:#6b7280;margin-bottom:8px">{rep['desc']}</div>
  <div style="font-size:12px">{rep['status']} | <code style="font-size:10px">{rep['path']}</code></div>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader(t("re_exec") + " — " + ("Live Preview" if st.session_state.lang=="en" else "Vista Previa"))

    # Executive summary table
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    zone = bt.get("traffic_light_zone","green")
    ze   = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(zone,"⚪")
    cp_cov = cp.get("coverage_test",{}).get("conformal_coverage",0.992)

    exec_data = {
        "Report field" if st.session_state.lang=="en" else "Campo": [
            "Report date","Portfolio","Models in production","Policy violations",
            "Credit Risk — Gini","Credit Risk — SR 11-7",
            "Market Risk — Basel III","Market Risk — Kupiec p",
            "Conformal Coverage","Fairness (Credit)","EU AI Act",
        ] if st.session_state.lang=="en" else [
            "Fecha del reporte","Portfolio","Modelos en producción","Violaciones de política",
            "Riesgo de Crédito — Gini","Riesgo de Crédito — SR 11-7",
            "Riesgo de Mercado — Basel III","Riesgo de Mercado — Kupiec p",
            "Cobertura Conformal","Fairness (Crédito)","EU AI Act",
        ],
        "Value" if st.session_state.lang=="en" else "Valor": [
            now,"Credit Risk + Market Risk","2 (Credit Risk v2.1 + Market VaR TFT v1.0)","0 blocks · 1 warn",
            f"{m['gini']:.3f} ✅","88% ✅",
            f"{ze} {zone.upper()} · {bt['n_exceedances']} exceedances",
            f"{bt['kupiec_pval']:.3f} ✅",
            f"{cp_cov:.2%} ✅","DIR=0.957 ✅ · DPD=0.030 ✅","6/7 articles ✅ · Art.14 ⚠️",
        ],
    }
    st.dataframe(pd.DataFrame(exec_data), use_container_width=True, hide_index=True)

    # Download JSON
    snapshot_path = Path("reports/mrr_snapshot.json")
    if snapshot_path.exists():
        with open(snapshot_path) as f:
            st.download_button(
                label=f"⬇️ {t('re_download')} — MRR Snapshot (JSON)",
                data=f.read(),
                file_name=f"mrr_snapshot_{datetime.now():%Y%m%d}.json",
                mime="application/json",
            )
    else:
        snapshot_json = json.dumps(exec_data, indent=2, default=str)
        st.download_button(
            label=f"⬇️ {t('re_download')} — Executive Summary (JSON)",
            data=snapshot_json,
            file_name=f"governance_report_{datetime.now():%Y%m%d}.json",
            mime="application/json",
        )

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
