from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExtractedNumber(BaseModel):
    value: float
    unit: str = ""
    context_span: str = ""


class ContentItem(BaseModel):
    id: str
    source_url: str
    source_name: str
    title: str
    body: str
    published_at: datetime | None = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    numbers: list[ExtractedNumber] = Field(default_factory=list)
    content_hash: str = ""


class EntityMention(BaseModel):
    surface_form: str
    canonical_id: str | None = None
    entity_type: str  # currency_pair, index, central_bank, indicator, company, person
    span_start: int = 0
    span_end: int = 0
    source_content_id: str
    resolution_confidence: float = 1.0


class CanonicalEntity(BaseModel):
    canonical_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)


class EntitySentiment(BaseModel):
    canonical_id: str
    canonical_name: str
    label: Literal["bullish", "bearish", "neutral"]
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence_span: str
    source_content_id: str
    source_name: str


class QAFlag(BaseModel):
    flag_type: Literal["sentiment_conflict", "unresolved_entity", "numeric_inconsistency"]
    entity_id: str | None = None
    details: str
    sources: list[str] = Field(default_factory=list)


class CostRecord(BaseModel):
    call_id: str
    stage: str  # extraction, resolution, sentiment, briefing
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    source_name: str = ""
    entity_id: str = ""


class PipelineResult(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime
    content_items: list[ContentItem]
    entities: list[CanonicalEntity]
    sentiments: list[EntitySentiment]
    qa_flags: list[QAFlag]
    cost_records: list[CostRecord]
    total_cost_usd: float
    briefings: dict[str, str] = Field(default_factory=dict)
