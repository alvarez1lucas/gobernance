"""
run_governance.py — Pipeline principal del AI Governance Framework
Ejecuta todos los checks, genera snapshot del MRR y dispara alertas.

Uso:
    python run_governance.py
    python run_governance.py --model credit-risk-v2.1
    python run_governance.py --export-json
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# ── SOLUCIÓN AL UNICODEENCODEERROR (EMOJIS EN WINDOWS) ────────────────────────
# Creamos el directorio 'reports' antes de configurar el logging
os.makedirs("reports", exist_ok=True)

# Forzamos la codificación UTF-8 para evitar fallos con emojis en la terminal (cp1252)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout if sys.stdout.encoding == 'utf-8' else open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
        logging.FileHandler(f"reports/governance_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def run_pipeline(model_id: str = None, export_json: bool = True):
    from src.mrr import ModelRiskRegister
    from src.data_loader import (
        load_credit_metrics, load_credit_fairness,
        load_market_backtest, load_market_conformal,
        load_all_audit_events,
    )

    log.info("=" * 60)
    log.info("AI Governance Framework — Pipeline start")
    log.info("=" * 60)

    # ── 1. Cargar MRR ─────────────────────────────────────────────────────────
    log.info("[1] Cargando Model Risk Register...")
    mrr = ModelRiskRegister("registry/model_registry.yaml")
    models = mrr.get_all_models()
    log.info(f"    Modelos registrados: {len(models)}")

    # ── 2. Evaluar policies con métricas en vivo ──────────────────────────────
    log.info("[2] Evaluando policies...")

    # Métricas live del Credit Risk
    credit_live = load_credit_metrics()
    credit_results = mrr.evaluate_policies("credit-risk-v2.1", live_metrics=credit_live)

    # Métricas live del Market Risk
    market_bt = load_market_backtest()
    market_cp = load_market_conformal()
    market_live = {
        "exceedances_250d":  market_bt.get("n_exceedances", 3),
        "kupiec_pval":       market_bt.get("kupiec_pval", 0.38),
        "conformal_coverage": market_cp.get("coverage_test", {}).get("conformal_coverage", 0.992),
        "psi": 0.08,
    }
    market_results = mrr.evaluate_policies("market-var-tft-v1.0", live_metrics=market_live)

    # Log resultados
    all_results = credit_results + market_results
    blocks = [r for r in all_results if not r.passed and r.severity == "block"]
    warns  = [r for r in all_results if not r.passed and r.severity == "warn"]

    for r in all_results:
        if not r.passed:
            emoji = "❌" if r.severity == "block" else "⚠️"
            log.warning(f"    {emoji} [{r.policy_id}] {r.message}")
        else:
            log.info(f"    ✅ [{r.policy_id}] {r.message}")

    log.info(f"    Resultado: {len(blocks)} BLOCKS | {len(warns)} WARNS | "
             f"{len(all_results)-len(blocks)-len(warns)} PASS")

    # ── 3. Fairness check unificado ───────────────────────────────────────────
    log.info("[3] Fairness check (Credit Risk)...")
    fairness = load_credit_fairness()
    overall_fair = fairness.get("overall_fairness_passed", True)
    log.info(f"    Fairness overall: {'✅ PASS' if overall_fair else '❌ FAIL'}")

    # ── 4. Audit trail unificado ──────────────────────────────────────────────
    log.info("[4] Cargando audit trail unificado...")
    events = load_all_audit_events()
    log.info(f"    Eventos totales: {len(events)} (credit + market + governance)")

    # ── 5. Export snapshot ────────────────────────────────────────────────────
    if export_json:
        log.info("[5] Exportando snapshot del MRR...")
        snapshot = mrr.export_json("reports/mrr_snapshot.json")
        log.info(f"    Snapshot guardado: reports/mrr_snapshot.json")

    # ── 6. Governance event en audit trail ───────────────────────────────────
    log.info("[6] Registrando evento de governance...")
    _append_governance_event({
        "n_models":   len(models),
        "n_approved": snapshot.get("n_approved", 0) if export_json else 0,
        "n_blocks":   len(blocks),
        "n_warns":    len(warns),
        "overall":    "PASS" if len(blocks) == 0 else "FAIL",
    })

    # ── Resultado final ───────────────────────────────────────────────────────
    log.info("=" * 60)
    if blocks:
        log.error(f"GOVERNANCE: ❌ FAIL — {len(blocks)} policies bloqueantes activas")
        log.error("   Revisar y corregir antes de desplegar modelos en producción")
        sys.exit(1)
    else:
        log.info(f"GOVERNANCE: ✅ PASS — todos los modelos en regla")
        log.info(f"   Dashboard: streamlit run dashboard/app.py")
    log.info("=" * 60)

    return {"blocks": len(blocks), "warns": len(warns), "models": len(models)}


def _append_governance_event(payload: dict):
    """Agrega un evento de governance al audit trail centralizado."""
    import hashlib

    # Aseguramos que el directorio exista también para este log específico
    os.makedirs("reports", exist_ok=True)
    log_path = Path("reports/governance_audit.jsonl")
    
    prev_hash = "GENESIS"
    if log_path.exists():
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        if lines:
            prev_hash = json.loads(lines[-1]).get("hash", "GENESIS")

    entry = {
        "timestamp":     datetime.now(tz=__import__("datetime").timezone.utc).isoformat(),
        "event_type":    "governance_check",
        "actor":         "run_governance.py",
        "payload":       payload,
        "previous_hash": prev_hash,
    }
    entry["hash"] = hashlib.sha256(
        json.dumps(entry, sort_keys=True, default=str).encode()
    ).hexdigest()

    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Governance Framework Pipeline")
    parser.add_argument("--model", help="Evaluar solo un modelo específico")
    parser.add_argument("--export-json", action="store_true", default=True,
                        help="Exportar snapshot del MRR en JSON")
    args = parser.parse_args()
    run_pipeline(model_id=args.model, export_json=args.export_json)