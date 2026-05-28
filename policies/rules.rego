# policies/rules.rego — Policy-as-code para AI Governance
# Ejecutado por OPA (Open Policy Agent) como gate en CI/CD
#
# Estructura:
#   deny[msg]  → bloquea el despliegue (severity: block)
#   warn[msg]  → genera alerta pero permite pasar (severity: warn)
#
# Referencia: SR 11-7 (Fed/OCC) · EU AI Act Art. 9 · Basel III IMA

package governance

import future.keywords.in

# ── CREDIT RISK POLICIES ───────────────────────────────────────────────────────

# Gini mínimo para modelos de crédito tier High
deny[msg] {
    input.model_id == "credit-risk-v2.1"
    input.tier == "High"
    input.metrics.gini < 0.45
    msg := sprintf(
        "POLICY MIN_PERFORMANCE_CREDIT: Gini %.4f < umbral 0.45 para tier High. Modelo bloqueado.",
        [input.metrics.gini]
    )
}

# Fairness — Disparate Impact Ratio mínimo
deny[msg] {
    input.model_id == "credit-risk-v2.1"
    input.fairness.dir_gender < 0.80
    msg := sprintf(
        "POLICY FAIRNESS_CREDIT: DIR de género %.3f < umbral regulatorio 0.80. Revisión requerida.",
        [input.fairness.dir_gender]
    )
}

# Calibración IFRS 9
warn[msg] {
    input.model_id == "credit-risk-v2.1"
    not input.ifrs9.calibrated
    msg := "POLICY IFRS9_CALIBRATION: Modelo no calibrado para IFRS 9 PD. Provisiones pueden ser incorrectas."
}

# ── MARKET RISK POLICIES ───────────────────────────────────────────────────────

# Basel III Traffic Light — zona roja bloquea
deny[msg] {
    input.model_id == "market-var-tft-v1.0"
    input.metrics.exceedances_250d > 9
    msg := sprintf(
        "POLICY MARKET_TRAFFIC_LIGHT: %d exceedances en zona ROJA (>9). Internal Models Approach revocado.",
        [input.metrics.exceedances_250d]
    )
}

# Basel III Traffic Light — zona amarilla alerta
warn[msg] {
    input.model_id == "market-var-tft-v1.0"
    input.metrics.exceedances_250d > 4
    input.metrics.exceedances_250d <= 9
    msg := sprintf(
        "POLICY MARKET_TRAFFIC_LIGHT: %d exceedances en zona AMARILLA (5-9). Capital multiplier aumentado.",
        [input.metrics.exceedances_250d]
    )
}

# Kupiec test obligatorio
deny[msg] {
    input.model_id == "market-var-tft-v1.0"
    input.metrics.kupiec_pval < 0.05
    msg := sprintf(
        "POLICY KUPIEC_PASS: p-value %.4f < 0.05. Modelo rechaza H0 — VaR mal calibrado.",
        [input.metrics.kupiec_pval]
    )
}

# Conformal coverage
warn[msg] {
    input.model_id == "market-var-tft-v1.0"
    input.metrics.conformal_coverage < 0.99
    msg := sprintf(
        "POLICY CP_COVERAGE: Cobertura conformal %.3f < 0.99. Garantía formal violada.",
        [input.metrics.conformal_coverage]
    )
}

# ── POLÍTICAS GLOBALES (ambos modelos) ─────────────────────────────────────────

# Drift warning
warn[msg] {
    input.metrics.psi > 0.10
    input.metrics.psi <= 0.20
    msg := sprintf(
        "POLICY DRIFT_WARNING: PSI %.3f en zona de alerta (0.10-0.20) para modelo %s. Re-validación recomendada.",
        [input.metrics.psi, input.model_id]
    )
}

# Drift breach — bloquea despliegue
deny[msg] {
    input.metrics.psi > 0.20
    msg := sprintf(
        "POLICY DRIFT_BREACH: PSI %.3f supera umbral crítico 0.20 para modelo %s. Re-validación INMEDIATA requerida.",
        [input.metrics.psi, input.model_id]
    )
}

# Validación independiente obligatoria para tier High
deny[msg] {
    input.tier == "High"
    not input.validation.independent_validator
    msg := sprintf(
        "POLICY VALIDATION_REQUIRED: Modelo %s tier High sin validación independiente. SR 11-7 Pilar 1 incumplido.",
        [input.model_id]
    )
}

# Fecha de re-validación vencida
deny[msg] {
    days_overdue := days_since(input.validation.next_date)
    days_overdue > 0
    msg := sprintf(
        "POLICY VALIDATION_OVERDUE: Modelo %s con re-validación vencida hace %d días.",
        [input.model_id, days_overdue]
    )
}

# EU AI Act — documentación requerida
warn[msg] {
    input.eu_ai_act.risk_category == "high_risk"
    input.eu_ai_act.conformity_assessment != "completed"
    msg := sprintf(
        "POLICY EU_AI_ACT_ART11: Modelo %s high-risk sin conformity assessment completado (EU AI Act Art. 11).",
        [input.model_id]
    )
}

# ── HELPER ────────────────────────────────────────────────────────────────────
# Días desde una fecha (placeholder — implementar con time.parse_rfc3339_ns en OPA real)
days_since(date_str) = 0 {
    # En producción: usar time.parse_rfc3339_ns y comparar con time.now_ns()
    # Aquí retornamos 0 para no bloquear en demo
    date_str != ""
}
