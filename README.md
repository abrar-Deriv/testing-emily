# Financial Content Intelligence Pipeline

An end-to-end pipeline that ingests content from multiple financial web sources, extracts and resolves financial entities, performs per-entity sentiment analysis, and generates structured intelligence briefings for multiple audiences.

## Sources

| Source | Extractor key | Method |
|--------|--------------|--------|
| Yahoo Finance EUR/USD | `yahoo_eurusd` | HTTP |
| Yahoo Finance Markets | `yahoo_markets` | HTTP |
| Federal Reserve Press Releases | `fed` | HTTP |
| IMF News | `imf` | HTTP → Playwright fallback |
| Reuters FX Markets | `reuters` | HTTP → Playwright fallback |

## Setup

```bash
# Install dependencies (requires uv)
uv sync

# Install Playwright browser (for JS-rendered sources)
uv run playwright install chromium

# Configure credentials
cp .env.example .env
# Edit .env with your LiteLLM API key and base URL
```

## Usage

```bash
# Full pipeline (all sources, all LLM stages)
uv run python main.py run

# Specific sources only
uv run python main.py run --sources fed,imf

# Extract only — no LLM calls (useful for testing/dev)
uv run python main.py run --dry-run

# List available source keys
uv run python main.py sources

# Verbose logging
uv run python main.py --verbose run --dry-run
```

## Output

All artifacts are written to `output/`:

| File | Contents |
|------|----------|
| `pipeline_result.json` | Full structured result: content items, entities, sentiments, QA flags, cost records |
| `briefing_trader.md` | Trader-focused briefing: bullet points, entity sentiment table, conflict flags |
| `briefing_analyst.md` | Analyst narrative: cross-source themes, confidence caveats, source attribution |
| `briefing_executive.md` | 3–5 sentence executive summary: top risk, top opportunity |
| `cost_report.json` | Token usage and estimated cost by stage and source |

## Architecture

```
Ingest (parallel) → Dedup → Entity Recognition (LLM) → Entity Resolution (deterministic + LLM)
    → Per-Entity Sentiment (batched LLM) → QA Checks → Audience Briefings → Write Outputs
```

### Cost optimization strategies

1. **Content deduplication** — duplicate bodies (same `content_hash`) are skipped before any LLM call.
2. **Tiered model routing** — cheap model (`MODEL_EXTRACTION`) for entity recognition and resolution; stronger model (`MODEL_SENTIMENT`) only for sentiment scoring.
3. **Batching** — all entities from the same article are sent in a single sentiment call, reducing API calls by ~3–5×.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_API_KEY` | Your LiteLLM / OpenAI API key | — |
| `LITELLM_BASE_URL` | LiteLLM proxy base URL | `https://api.openai.com/v1` |
| `MODEL_EXTRACTION` | Model for entity recognition + resolution | `gpt-4o-mini` |
| `MODEL_SENTIMENT` | Model for per-entity sentiment | `gpt-4o` |
| `MODEL_BRIEFING` | Model for audience briefings | `gpt-4o-mini` |
