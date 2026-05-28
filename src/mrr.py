"""
src/mrr.py — Model Risk Register Engine
Lee model_registry.yaml y expone métodos para consultar y validar modelos.
"""

import yaml
import json
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PolicyResult:
    policy_id: str
    severity: str          # "block" | "warn" | "pass"
    model_id: str
    message: str
    passed: bool


@dataclass
class ModelSummary:
    id: str
    name: str
    type: str
    tier: str
    status: str
    owner: str
    sr117_score: float
    last_validation: str
    next_validation: str
    psi: float
    psi_status: str
    regulations: List[str]
    policy_results: List[PolicyResult] = field(default_factory=list)

    @property
    def overall_status_emoji(self):
        return {"approved": "✅", "under_review": "⚠️",
                "pending": "🔵", "retired": "⛔"}.get(self.status, "❓")

    @property
    def tier_color(self):
        return {"High": "#ef476f", "Medium": "#ffd166", "Low": "#06d6a0"}.get(self.tier, "#adb5bd")

    @property
    def days_to_next_validation(self) -> int:
        try:
            nxt = datetime.strptime(self.next_validation, "%Y-%m-%d").date()
            return (nxt - date.today()).days
        except Exception:
            return 999

    @property
    def validation_urgency(self) -> str:
        d = self.days_to_next_validation
        if d < 0:   return "OVERDUE"
        if d < 30:  return "URGENT"
        if d < 90:  return "UPCOMING"
        return "OK"


class ModelRiskRegister:
    """
    Motor central del Model Risk Register.
    Lee model_registry.yaml y expone métodos para:
      - Listar todos los modelos con su estado
      - Evaluar policies contra las métricas actuales
      - Generar resumen ejecutivo para el dashboard
    """

    def __init__(self, registry_path: str = "registry/model_registry.yaml"):
        self.registry_path = Path(registry_path)
        self._data = self._load()

    def _load(self) -> dict:
        if not self.registry_path.exists():
            return {"models": [], "policies": []}
        with open(self.registry_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_all_models(self) -> List[dict]:
        return self._data.get("models", [])

    def get_model(self, model_id: str) -> Optional[dict]:
        for m in self.get_all_models():
            if m.get("id") == model_id:
                return m
        return None

    def get_policies(self) -> List[dict]:
        return self._data.get("policies", [])

    def evaluate_policies(self, model_id: str,
                          live_metrics: Optional[dict] = None) -> List[PolicyResult]:
        """
        Evalúa todas las policies sobre un modelo.
        live_metrics: métricas en tiempo real (sobreescribe las del YAML).
        """
        model = self.get_model(model_id)
        if not model:
            return []

        # Merge métricas del YAML con live si las hay
        metrics = dict(model.get("metrics", {}))
        if live_metrics:
            metrics.update(live_metrics)

        results = []
        policies = self.get_policies()

        for policy in policies:
            if model_id not in policy.get("applies_to", []):
                continue

            rule  = policy.get("rule", "")
            sev   = policy.get("severity", "warn")
            pid   = policy.get("id", "")
            desc  = policy.get("description", "")

            passed, msg = self._evaluate_rule(rule, model, metrics)

            results.append(PolicyResult(
                policy_id=pid,
                severity=sev if not passed else "pass",
                model_id=model_id,
                message=msg if not passed else f"✓ {desc}",
                passed=passed,
            ))

        return results

    def _evaluate_rule(self, rule: str, model: dict,
                       metrics: dict) -> tuple[bool, str]:
        """Evalúa una regla simple contra las métricas del modelo."""
        try:
            # Construir namespace con todas las variables
            ns = {}
            ns.update(metrics)
            ns.update(model.get("fairness", {}))
            ns.update(model.get("validation", {}))
            ns.update({"tier": model.get("tier", "")})

            # Normalizar nombres (guiones → guiones bajos)
            ns_clean = {k.replace("-", "_"): v for k, v in ns.items()}

            result = eval(rule, {"__builtins__": {}}, ns_clean)
            return bool(result), ""
        except Exception as e:
            return True, f"Rule evaluation error: {e}"

    def get_model_summary(self, model_id: str) -> Optional[ModelSummary]:
        model = self.get_model(model_id)
        if not model:
            return None

        policy_results = self.evaluate_policies(model_id)

        return ModelSummary(
            id=model_id,
            name=model.get("name", ""),
            type=model.get("type", ""),
            tier=model.get("tier", ""),
            status=model.get("status", ""),
            owner=model.get("owner", ""),
            sr117_score=model.get("validation", {}).get("sr117_score", 0),
            last_validation=model.get("validation", {}).get("last_date", ""),
            next_validation=model.get("validation", {}).get("next_date", ""),
            psi=model.get("metrics", {}).get("psi", 0),
            psi_status=model.get("metrics", {}).get("psi_status", ""),
            regulations=model.get("regulations", []),
            policy_results=policy_results,
        )

    def get_dashboard_data(self) -> dict:
        """Genera todos los datos necesarios para el dashboard ejecutivo."""
        models  = self.get_all_models()
        summaries = []
        all_policies = []
        blocks = 0
        warns  = 0

        for m in models:
            s = self.get_model_summary(m["id"])
            if s:
                summaries.append(s)
                for p in s.policy_results:
                    all_policies.append(p)
                    if not p.passed:
                        if p.severity == "block": blocks += 1
                        elif p.severity == "warn": warns  += 1

        return {
            "summaries":       summaries,
            "all_policies":    all_policies,
            "n_models":        len(models),
            "n_approved":      sum(1 for m in models if m.get("status") == "approved"),
            "n_blocks":        blocks,
            "n_warns":         warns,
            "metadata":        self._data.get("metadata", {}),
        }

    def export_json(self, path: str = "reports/mrr_snapshot.json"):
        """Exporta un snapshot del MRR en JSON para el audit trail."""
        data = self.get_dashboard_data()
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "n_models": data["n_models"],
            "n_approved": data["n_approved"],
            "n_policy_blocks": data["n_blocks"],
            "n_policy_warns": data["n_warns"],
            "models": [
                {
                    "id": s.id,
                    "name": s.name,
                    "status": s.status,
                    "tier": s.tier,
                    "sr117_score": s.sr117_score,
                    "psi": s.psi,
                    "next_validation": s.next_validation,
                    "days_to_next_validation": s.days_to_next_validation,
                    "n_policy_issues": sum(1 for p in s.policy_results if not p.passed),
                }
                for s in data["summaries"]
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
        return snapshot
