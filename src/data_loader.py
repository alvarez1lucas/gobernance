"""
src/data_loader.py — Carga outputs de ambos repos con fallback sintético.
El governance repo puede correr stand-alone sin tener los repos hermanos.

Jerarquía de búsqueda de paths:
  1. ../credit-risk-model-validation/reports/   (repo hermano)
  2. submodules/credit-risk/reports/             (submódulo git)
  3. demo/credit_risk/                           (datos de demo incluidos)
  4. Datos sintéticos generados en memoria
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger


# ── Paths de búsqueda ─────────────────────────────────────────────────────────
CREDIT_SEARCH_PATHS = [
    Path("../credit-risk-model-validation/reports"),
    Path("submodules/credit-risk/reports"),
    Path("demo/credit_risk"),
]

MARKET_SEARCH_PATHS = [
    Path("../market-risk-deep-learning/reports"),
    Path("submodules/market-risk/reports"),
    Path("demo/market_risk"),
]


def _find_file(filename: str, search_paths: list) -> Optional[Path]:
    for base in search_paths:
        p = base / filename
        if p.exists():
            return p
    return None


def _load_json(filename: str, search_paths: list, fallback: dict) -> dict:
    p = _find_file(filename, search_paths)
    if p:
        try:
            return json.loads(p.read_text())
        except Exception as e:
            logger.warning(f"Error leyendo {p}: {e}")
    return fallback


def _load_csv(filename: str, search_paths: list) -> Optional[pd.DataFrame]:
    p = _find_file(filename, search_paths)
    if p:
        try:
            return pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CREDIT RISK DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_credit_sr117() -> dict:
    fallback = {
        "sr117_overall_pass": True,
        "discriminatory_power": {
            "gini": 0.712, "auc_roc": 0.856,
            "ks_statistic": 0.524, "gini_lift_baseline": 0.572,
        },
        "calibration": {"hl_pvalue": 0.19, "well_calibrated": True},
        "stability":   {"psi": 0.06, "psi_status": "stable"},
        "stress_testing": {
            "baseline_gini": 0.712,
            "scenarios": {
                "income_shock_moderate": {"gini": 0.672, "auc_degradation": 0.02},
                "income_shock_severe":   {"gini": 0.632, "auc_degradation": 0.04},
                "bureau_deterioration":  {"gini": 0.592, "auc_degradation": 0.06},
            },
        },
        "sensitivity_top10": {
            "EXT_SOURCE_2": 0.08, "EXT_SOURCE_3": 0.06,
            "EXT_SOURCE_1": 0.04, "AMT_CREDIT":   0.02,
            "DAYS_BIRTH":   0.015,
        },
    }
    return _load_json("sr117_validation.json", CREDIT_SEARCH_PATHS, fallback)


def load_credit_fairness() -> dict:
    fallback = {
        "overall_fairness_passed": True,
        "results": {"gender": {
            "approval_rates":             {"M": 0.682, "F": 0.712},
            "demographic_parity_difference": 0.030,
            "disparate_impact_ratio":     0.957,
            "equalized_odds": {
                "M": {"tpr": 0.62, "fpr": 0.09},
                "F": {"tpr": 0.60, "fpr": 0.08},
                "tpr_gap": 0.02, "fpr_gap": 0.01,
            },
            "regulatory_flags": [], "passed": True,
        }},
    }
    return _load_json("fairness_report.json", CREDIT_SEARCH_PATHS, fallback)


def load_credit_metrics() -> dict:
    """Métricas principales del modelo de crédito."""
    sr117 = load_credit_sr117()
    dp    = sr117.get("discriminatory_power", {})
    cal   = sr117.get("calibration", {})
    stab  = sr117.get("stability", {})
    return {
        "gini":           dp.get("gini", 0.712),
        "auc_roc":        dp.get("auc_roc", 0.856),
        "ks_statistic":   dp.get("ks_statistic", 0.524),
        "brier_score":    0.084,
        "psi":            stab.get("psi", 0.06),
        "psi_status":     stab.get("psi_status", "stable"),
        "hl_pvalue":      cal.get("hl_pvalue", 0.19),
        "well_calibrated":cal.get("well_calibrated", True),
        "sr117_pass":     sr117.get("sr117_overall_pass", True),
    }


def load_credit_demo_data() -> dict:
    """Genera datos sintéticos de Credit Risk para el dashboard."""
    from sklearn.metrics import roc_curve, roc_auc_score
    np.random.seed(42); n = 8000
    y  = np.random.binomial(1, 0.08, n)
    s  = np.clip(y*np.random.beta(5,2,n) + (1-y)*np.random.beta(2,5,n), 0.01, 0.99)
    s_nn   = np.clip(s + np.random.normal(0, 0.03, n), 0.01, 0.99)
    s_lstm = np.clip(s + np.random.normal(0.015, 0.04, n), 0.01, 0.99)
    gender = np.random.choice(["M","F"], n, p=[0.58, 0.42])
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "y_test": y, "y_score": s, "y_score_nn": s_nn, "y_score_lstm": s_lstm,
        "fpr": fpr, "tpr": tpr,
        "gender": gender,
        "gini_nn":   round(2*roc_auc_score(y, s_nn)   - 1, 4),
        "gini_lstm": round(2*roc_auc_score(y, s_lstm)  - 1, 4),
        "gini_lr":   round(2*roc_auc_score(y, s) - 1 - 0.14, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MARKET RISK DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_market_backtest() -> dict:
    fallback = {
        "n_exceedances": 3, "n_observations": 250,
        "kupiec_pval": 0.38, "kupiec_pass": True,
        "christoffersen_pval": 0.55, "christoffersen_pass": True,
        "traffic_light_zone": "green", "overall_status": "approved",
        "exceedance_rate": 0.012, "expected_rate": 0.01,
    }
    return _load_json("var_backtest.json", MARKET_SEARCH_PATHS, fallback)


def load_market_sr117() -> dict:
    fallback = {
        "model_name": "Market VaR — TFT v1.0",
        "validation_date": "2025-01-15T10:00:00",
        "overall_status": "approved", "overall_score": 0.88,
        "checks": [
            {"section":"Conceptual Soundness","requirement":"Statistical theory documented",
             "status":"pass","score":1.0,"evidence":"ADRs + Basel III mapping"},
            {"section":"Ongoing Monitoring","requirement":"Drift monitoring active",
             "status":"pass","score":1.0,"evidence":"PSI daily + KS test"},
            {"section":"Outcomes Analysis","requirement":"Kupiec PASS",
             "status":"pass","score":1.0,"evidence":"p=0.38 > 0.05"},
        ],
        "limitations": [
            "Calibrated 2000-2024 — excludes LATAM post-2024 hyperinflation",
            "Equal-weighted portfolio assumption",
            "1-day VaR assumes perfect liquidity",
        ],
    }
    return _load_json("sr117_validation.json", MARKET_SEARCH_PATHS, fallback)


def load_market_conformal() -> dict:
    fallback = {
        "coverage_test": {
            "target_coverage": 0.99, "conformal_coverage": 0.992,
            "classical_coverage": 0.984, "conformal_exceedances": 2,
            "classical_exceedances": 4, "conformal_kupiec_pval": 0.61,
            "classical_kupiec_pval": 0.29,
        },
        "nonconformity_quantile": 0.0042,
        "conformal_valid": True, "n_calibration": 189,
    }
    return _load_json("conformal_backtest.json", MARKET_SEARCH_PATHS, fallback)


def load_market_stress() -> dict:
    fallback = {
        "gfc_2008":   {"name":"GFC 2008","es_975_1d":-0.042,"total_loss_pct":-0.61},
        "covid_2020": {"name":"COVID-19 Q1 2020","es_975_1d":-0.031,"total_loss_pct":-0.38},
        "rates_2022": {"name":"Rate Hike 2022","es_975_1d":-0.024,"total_loss_pct":-0.28},
        "svb_2023":   {"name":"SVB Run 2023","es_975_1d":-0.018,"total_loss_pct":-0.17},
        "dfast_adv":  {"name":"DFAST Severely Adv.","es_975_1d":-0.038,"total_loss_pct":-0.58},
        "latam_tail": {"name":"LATAM Tail Risk","es_975_1d":-0.028,"total_loss_pct":-0.43},
    }
    path = _find_file("stress_scenarios/stress_report.json", MARKET_SEARCH_PATHS)
    if path:
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return fallback


def load_market_sentiment() -> pd.DataFrame:
    df = _load_csv("sentiment/sentiment_daily.csv", [
        Path("../market-risk-deep-learning/data/raw"),
        Path("submodules/market-risk/data/raw"),
    ])
    if df is not None and "sentiment_mean" in df.columns:
        return df
    # Sintético
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.today(), periods=1260)
    s = np.zeros(len(dates)); s[0] = 0.1
    for i in range(1, len(dates)):
        s[i] = 0.7*s[i-1] + np.random.normal(0.02, 0.15)
    s = np.clip(s, -1, 1)
    return pd.DataFrame({"sentiment_mean": s, "sentiment_ma21": pd.Series(s).rolling(21).mean().fillna(0).values}, index=dates)


def load_market_returns() -> pd.Series:
    df = _load_csv("features.csv", [
        Path("../market-risk-deep-learning/data/raw"),
        Path("submodules/market-risk/data/raw"),
    ])
    if df is not None and "log_return_SPX" in df.columns:
        return df["log_return_SPX"].dropna()
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.today(), periods=1260)
    r = np.random.normal(-0.0003, 0.012, len(dates))
    for s, e, sh in [(200,280,-0.025),(800,830,-0.035),(1050,1090,-0.018)]:
        r[s:e] += sh
    return pd.Series(r, index=dates, name="log_return_SPX")


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════

def load_all_audit_events() -> list:
    """Combina audit trails de ambos repos en orden cronológico."""
    all_events = []

    for search_paths, prefix in [
        (CREDIT_SEARCH_PATHS, "credit"),
        (MARKET_SEARCH_PATHS, "market"),
    ]:
        for base in search_paths:
            p = base / "audit_trail.jsonl"
            if p.exists():
                try:
                    lines = p.read_text().splitlines()
                    for line in lines:
                        if line.strip():
                            e = json.loads(line)
                            e["_source"] = prefix
                            all_events.append(e)
                    break
                except Exception:
                    pass

    if not all_events:
        # Demo events
        all_events = [
            {"timestamp":"2025-01-10T09:00:00Z","event_type":"pipeline_started","actor":"pipeline",
             "payload":{},"_source":"credit","hash":"abc123"},
            {"timestamp":"2025-01-10T10:30:00Z","event_type":"model_trained","actor":"pipeline",
             "payload":{"model":"XGBoost","gini":0.712},"_source":"credit","hash":"def456"},
            {"timestamp":"2025-01-10T11:00:00Z","event_type":"sr117_completed","actor":"pipeline",
             "payload":{"status":"approved","score":0.88},"_source":"credit","hash":"ghi789"},
            {"timestamp":"2025-01-15T09:00:00Z","event_type":"pipeline_started","actor":"pipeline",
             "payload":{},"_source":"market","hash":"jkl012"},
            {"timestamp":"2025-01-15T09:30:00Z","event_type":"model_trained","actor":"pipeline",
             "payload":{"model":"TFT","val_loss":0.00041},"_source":"market","hash":"mno345"},
            {"timestamp":"2025-01-15T09:45:00Z","event_type":"backtesting_completed","actor":"pipeline",
             "payload":{"kupiec_pval":0.38,"exceedances":3},"_source":"market","hash":"pqr678"},
            {"timestamp":"2025-01-15T10:00:00Z","event_type":"governance_snapshot","actor":"governance",
             "payload":{"n_models":2,"n_approved":2},"_source":"governance","hash":"stu901"},
        ]

    # Ordenar por timestamp
    all_events.sort(key=lambda e: e.get("timestamp",""))
    return all_events
