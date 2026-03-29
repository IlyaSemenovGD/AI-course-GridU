# Data Assistant – Phases 1 & 2

A Streamlit application that generates realistic synthetic datasets from SQL schemas
and lets you query them conversationally using natural language.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Gemini 2.5 Flash (Google AI Studio or Vertex AI) |
| SDK | `google-genai` (unified Google GenAI SDK) |
| UI | Streamlit |
| Database | PostgreSQL 16 |
| Visualizations | Seaborn / Matplotlib |
| Containers | Docker / Docker Compose |
| Observability | Langfuse |

---

## Features

### Phase 1 — Data Generation
- Upload any `.sql` / `.ddl` / `.txt` DDL schema (up to ~7 tables)
- Three sample schemas included: Company/Employees, Restaurants, Library Management
- Configurable rows per table (1–1000), temperature, and model
- **Streaming** generation with live per-table progress
- **Structured JSON output** ensuring schema compliance
- Respects ENUMs, CHECK constraints, NOT NULL, FK relationships
- Circular FK detection and deferred resolution
- Per-table **function-calling**-based modification:
  - Set numeric column ranges
  - Rebalance ENUM distributions
  - Conditional updates
  - Full column regeneration
- Download tables as individual CSV or full ZIP archive
- Save datasets to PostgreSQL for access in Talk to Your Data tab

### Phase 2 — Talk to Your Data
- Conversational chat interface over saved datasets
- Natural language → SQL via Gemini **function calling**
- **Streaming** text responses with live token output
- SQL query shown in expandable code block alongside tabular results
- Automatic **Seaborn chart** generation (bar, line, scatter, hist, box, heatmap)
- Chart type auto-detected from data shape or specified by the model
- Guardrails:
  - Topic blocklist (off-topic requests rejected without LLM call)
  - Prompt injection / jailbreak regex detection
  - SQL safety validation (read-only SELECT enforced)
  - LLM-based jailbreak classifier (structured output, confidence score)
  - Optional PII masking on query results (`ENABLE_PII_MASKING=true`)
- Langfuse traces per conversation turn with jailbreak confidence scores

---

## Quick Start

### Local (without Docker)

**Step 1 — Start PostgreSQL**

Pick whichever option suits you:

*Option A – Docker (just the database):*
```bash
docker run -d \
  --name datagen-postgres \
  -e POSTGRES_DB=datagen \
  -e POSTGRES_USER=datagen \
  -e POSTGRES_PASSWORD=datagen \
  -p 5432:5432 \
  postgres:16-alpine
```

*Option B – Homebrew (macOS):*
```bash
brew install postgresql@16
brew services start postgresql@16
psql postgres -c "CREATE USER datagen WITH PASSWORD 'datagen';"
psql postgres -c "CREATE DATABASE datagen OWNER datagen;"
```

*Option C – Postgres.app (macOS, GUI):*
Download from [postgresapp.com](https://postgresapp.com), start it, then in the psql shell:
```sql
CREATE USER datagen WITH PASSWORD 'datagen';
CREATE DATABASE datagen OWNER datagen;
```

**Step 2 — Install & configure**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Required: set GOOGLE_API_KEY (or GCP_PROJECT for Vertex AI)
# Postgres defaults (host=localhost, db/user/password=datagen) need no changes
# if you used Option A or B above.
```

**Step 3 — Run**

```bash
streamlit run app/main.py
```

### Docker Compose

```bash
cp .env.example .env
# Set GOOGLE_API_KEY or GCP_PROJECT; optionally add Langfuse keys

docker compose up --build
```

App available at <http://localhost:8501>.

> **Vertex AI auth inside Docker**: the compose file bind-mounts `~/.config/gcloud`
> into the container. Run `gcloud auth application-default login` on the host first.

---

## Typical Workflow

1. **Data Generation tab** → upload a DDL schema (or pick a sample) → set instructions + row count → click **Generate**
2. Review per-table previews; use the quick-edit box to refine any table
3. Click **Save to Database** to persist the dataset
4. **Talk to Your Data tab** → select the saved dataset → ask questions in natural language

Example questions:
- *"How many employees are in each department?"*
- *"Show me the top 5 restaurants by average rating as a bar chart"*
- *"What is the salary distribution across job titles?"*

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ (or GCP_PROJECT) | Google AI Studio API key |
| `GCP_PROJECT` | ✅ (or GOOGLE_API_KEY) | GCP project for Vertex AI |
| `GCP_LOCATION` | | Vertex AI region (default `us-central1`) |
| `POSTGRES_HOST` | | DB host (default `localhost`) |
| `POSTGRES_PORT` | | DB port (default `5432`) |
| `POSTGRES_DB` | | DB name (default `datagen`) |
| `POSTGRES_USER` | | DB user (default `datagen`) |
| `POSTGRES_PASSWORD` | | DB password (default `datagen`) |
| `CHAT_MODEL` | | Model for chat (default `gemini-2.5-flash`) |
| `ENABLE_PII_MASKING` | | Set `true` to redact PII in query results |
| `LANGFUSE_PUBLIC_KEY` | | Langfuse public key (optional) |
| `LANGFUSE_SECRET_KEY` | | Langfuse secret key (optional) |
| `LANGFUSE_HOST` | | Langfuse host (default `https://cloud.langfuse.com`) |

---

## Project Structure

```
phase1/
├── app/
│   ├── main.py                   # Streamlit entry point + sidebar navigation
│   ├── core/
│   │   ├── ddl_parser.py         # DDL → TableDef objects + topological sort
│   │   ├── data_generator.py     # Gemini streaming + structured output + function calling
│   │   ├── database.py           # PostgreSQL persistence + read-only query execution
│   │   ├── sql_agent.py          # NL-to-SQL agentic loop (Phase 2)
│   │   ├── guardrails.py         # Input validation, jailbreak detection, PII masking
│   │   └── visualizer.py         # Seaborn chart renderer
│   ├── utils/
│   │   └── observability.py      # Langfuse tracing + scoring helpers
│   └── pages/
│       ├── data_generation.py    # Data Generation tab
│       └── talk_to_data.py       # Talk to Your Data tab
├── company_employee_schema.ddl
├── library_mgm_schema.ddl
├── restrurants_schema.ddl
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Gemini API Features Used

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| **Streaming** (`generate_content_stream`) | Live progress during data generation | Streaming chat responses |
| **Structured output** (`response_mime_type="application/json"`) | All data generation calls | Jailbreak classifier |
| **Function calling** (`Tool` / `FunctionDeclaration`) | Per-table data modification | NL-to-SQL query execution + chart hints |

---

## Langfuse Observability

When `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are configured:

**Phase 1 traces** — `generate_table`, `modify_table`

**Phase 2 traces** — `chat_turn` with nested spans:
- `guardrail_check` — input validation result
- `sql_generation` — Turn 1: model call that produces the SQL tool call
- `sql_execution` — query execution metadata
- `answer_generation` — Turn 2: streamed explanation

**Scores emitted per turn:**

| Score | Value | Purpose |
|-------|-------|---------|
| `jailbreak` | 0.0–1.0 | Jailbreak confidence from classifier |
| `guardrail_passed` | 0 or 1 | Whether the message was allowed |
| `sql_success` | 0 or 1 | Whether the SQL executed successfully |
| `jailbreak_alert` | 1.0 | Emitted only when jailbreak score ≥ 0.7 — use as Langfuse alert filter |

If credentials are absent the app runs without any observability overhead.
