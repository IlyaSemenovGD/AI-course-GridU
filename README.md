# Phase 1 – Synthetic Data Generation

A Streamlit application that interprets SQL DDL schemas and generates realistic synthetic datasets using **Gemini 2.0 Flash** via Vertex AI.

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Gemini 2.0 Flash (Vertex AI) |
| SDK | `google-genai` (new unified SDK) |
| UI | Streamlit |
| Database | PostgreSQL 16 |
| Containers | Docker / Docker Compose |
| Observability | Langfuse |

## Features

- Upload any `.sql` / `.ddl` / `.txt` schema file (up to ~7 tables)
- Three sample schemas included: Company/Employees, Restaurants, Library
- Configurable rows per table (1–1000), temperature, and model
- **Streaming** generation with live progress
- **Structured JSON output** ensuring schema compliance
- Respects ENUMs, CHECK constraints, NOT NULL, FK relationships
- Circular FK detection and deferred resolution
- Per-table **function-calling**-based modification (set ranges, replace enum distributions, conditional updates, column regeneration)
- Download tables as individual CSV or full ZIP
- Save datasets to PostgreSQL for later access in "Talk to your data" tab
- Langfuse traces for every LLM call (optional)

## Quick Start

### Local (without Docker)

**Step 1 — Start PostgreSQL**

Pick whichever option suits you:

*Option A – Docker (just the database, no full compose):*
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
# then create the database and user:
psql postgres -c "CREATE USER datagen WITH PASSWORD 'datagen';"
psql postgres -c "CREATE DATABASE datagen OWNER datagen;"
```

*Option C – Postgres.app (macOS, GUI):*
Download from [postgresapp.com](https://postgresapp.com), start it, then open the psql shell and run:
```sql
CREATE USER datagen WITH PASSWORD 'datagen';
CREATE DATABASE datagen OWNER datagen;
```

**Step 2 — Install & configure the app**

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Required: set GOOGLE_API_KEY (or GCP_PROJECT for Vertex AI)
# Postgres defaults (host=localhost, db/user/password=datagen) work out of the box
#   if you used Option A or B above — no changes needed.
```

**Step 3 — Run**

```bash
streamlit run app/main.py
```

### Docker Compose

```bash
cp .env.example .env
# fill in GCP_PROJECT (and optionally Langfuse keys)

docker compose up --build
```

The app will be available at <http://localhost:8501>.

> **Vertex AI auth inside Docker**: the compose file bind-mounts
> `~/.config/gcloud` into the container so your Application Default
> Credentials are available.  Run `gcloud auth application-default login`
> on the host first.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GCP_PROJECT` | ✅ | Google Cloud project with Vertex AI enabled |
| `GCP_LOCATION` | | Vertex AI region (default `us-central1`) |
| `POSTGRES_HOST` | | DB host (default `localhost`) |
| `POSTGRES_PORT` | | DB port (default `5432`) |
| `POSTGRES_DB` | | DB name (default `datagen`) |
| `POSTGRES_USER` | | DB user (default `datagen`) |
| `POSTGRES_PASSWORD` | | DB password (default `datagen`) |
| `LANGFUSE_PUBLIC_KEY` | | Langfuse public key (observability optional) |
| `LANGFUSE_SECRET_KEY` | | Langfuse secret key |
| `LANGFUSE_HOST` | | Langfuse host (default `https://cloud.langfuse.com`) |

## Project Structure

```
phase1/
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── core/
│   │   ├── ddl_parser.py        # DDL → TableDef objects + topological sort
│   │   ├── data_generator.py    # Gemini streaming + function-calling generator
│   │   └── database.py          # PostgreSQL persistence
│   ├── utils/
│   │   └── observability.py     # Langfuse tracing helpers
│   └── pages/
│       ├── data_generation.py   # Data Generation tab
│       └── talk_to_data.py      # Talk to Data tab (Phase 2 placeholder)
├── company_employee_schema.ddl
├── library_mgm_schema.ddl
├── restrurants_schema.ddl
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Gemini API Features Used

| Feature | Where |
|---------|-------|
| **Streaming** (`generate_content_stream`) | Table data generation – live progress in UI |
| **Structured output** (`response_mime_type="application/json"`) | All data generation calls |
| **Function calling** (`Tool` / `FunctionDeclaration`) | Per-table data modification |

## Langfuse Traces

When `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set, every LLM call
is traced under the `generate_table` or `modify_table` trace name.  If
credentials are absent the app runs normally without any observability
overhead.
