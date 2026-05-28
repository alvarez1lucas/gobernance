# AI Governance Framework

Repositorio central de AI Governance que integra **Credit Risk** y **Market Risk**
como submódulos, implementando el ciclo completo de Model Risk Management bajo
SR 11-7, EU AI Act, Basel III FRTB e IFRS 9.

## Estructura

```
ai-governance-framework/
├── registry/
│   └── model_registry.yaml     # Model Risk Register centralizado
├── policies/
│   └── rules.rego              # Policy-as-code (OPA/Rego)
├── src/
│   ├── mrr.py                  # Model Risk Register engine
│   └── data_loader.py          # Lee outputs de ambos repos
├── dashboard/
│   └── app.py                  # Streamlit ejecutivo (EN/ES, 10 páginas)
├── reports/                    # Outputs consolidados
├── run_governance.py           # Pipeline principal
└── README.md
```

## Quickstart

```bash
# 1. Clonar
git clone https://github.com/[user]/ai-governance-framework
cd ai-governance-framework

# 2. (Opcional) Agregar repos hermanos como submódulos
git submodule add https://github.com/[user]/credit-risk-model-validation submodules/credit-risk
git submodule add https://github.com/[user]/market-risk-deep-learning    submodules/market-risk

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Correr pipeline de governance
python run_governance.py

# 5. Dashboard ejecutivo
streamlit run dashboard/app.py
```

## Dashboard — 10 páginas

| # | Página | Contenido |
|---|--------|-----------|
| 1 | 🏛️ MRR Overview | Inventario de modelos, timeline de validaciones, regulaciones |
| 2 | 🔒 Policy Status | Evaluación OPA/Rego en vivo, blocks y warns |
| 3 | 💳 Credit Risk | ROC, SHAP, stress scenarios, IFRS 9 calibration |
| 4 | 📊 Market Risk | VaR fan chart, backtesting, sentimiento NLP, conformal |
| 5 | ⚖️ Comparative | Radar regulatorio, scores SR 11-7, drift PSI lado a lado |
| 6 | 🌈 Fairness | DPD, DIR, Equalized Odds, distribución por género |
| 7 | ⚠️ Stress Testing | Escenarios unificados crédito + mercado + Monte Carlo |
| 8 | 🇪🇺 EU AI Act | Checklist Art. 9-15, compliance matrix por modelo |
| 9 | 🗂️ Audit Trail | Log unificado SHA-256, timeline visual, filtro por fuente |
| 10 | 📑 Reports | Executive summary, descarga de reportes, preview live |

## Regulatorio cubierto

- **SR 11-7** (Fed/OCC) — Model Risk Management, tres pilares
- **EU AI Act** (2024/1689) — Annex III/IV, Art. 9-15
- **Basel III FRTB** — Internal Models Approach, VaR/ES
- **IFRS 9** — PD calibration, provisiones
- **BCBS 239** — Risk data aggregation
- **BCRA A 7724** — Regulación local Argentina

## Conexión con repos hermanos

El dashboard busca los outputs en este orden:

```
1. ../credit-risk-model-validation/reports/    ← repo hermano
2. submodules/credit-risk/reports/             ← submódulo git
3. demo/credit_risk/                           ← datos de demo
4. Datos sintéticos en memoria                 ← fallback automático
```

Si no se encuentran outputs reales, el dashboard muestra datos
sintéticos realistas para demostración.

## Stack

Python · PyYAML · Streamlit · Plotly · Scikit-learn · Scipy · NumPy · Pandas

## Roles target

AI Governance Officer · Model Risk Manager · Quant Risk Engineer · FRM candidate
