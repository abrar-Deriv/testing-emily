from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.models.schemas import CostRecord

# Approximate per-token costs in USD for common models.
# These are estimates; adjust if your LiteLLM proxy uses different pricing.
_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o": {"input": 0.002500, "output": 0.010000},
    "gpt-4-turbo": {"input": 0.010000, "output": 0.030000},
}
_DEFAULT_COST = {"input": 0.001000, "output": 0.003000}


def _cost_per_token(model: str) -> dict[str, float]:
    for key, rates in _COST_PER_1K.items():
        if key in model:
            return rates
    return _DEFAULT_COST


@dataclass
class CostTracker:
    records: list[CostRecord] = field(default_factory=list)

    def record(
        self,
        *,
        stage: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        source_name: str = "",
        entity_id: str = "",
    ) -> CostRecord:
        rates = _cost_per_token(model)
        cost = (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]
        rec = CostRecord(
            call_id=str(uuid.uuid4()),
            stage=stage,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=round(cost, 8),
            source_name=source_name,
            entity_id=entity_id,
        )
        self.records.append(rec)
        return rec

    @property
    def total_cost(self) -> float:
        return round(sum(r.estimated_cost_usd for r in self.records), 6)

    def summary_by_stage(self) -> dict[str, float]:
        stages: dict[str, float] = {}
        for r in self.records:
            stages[r.stage] = round(stages.get(r.stage, 0.0) + r.estimated_cost_usd, 8)
        return stages

    def summary_by_source(self) -> dict[str, float]:
        sources: dict[str, float] = {}
        for r in self.records:
            if r.source_name:
                sources[r.source_name] = round(sources.get(r.source_name, 0.0) + r.estimated_cost_usd, 8)
        return sources
