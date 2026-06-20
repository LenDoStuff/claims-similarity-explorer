# Claims Similarity Explorer

Python + Streamlit MVP for local semantic search and exploratory clustering of insurance claims.

Snowflake is the source for ingestion only. After indexing, the app reads claim documents, embeddings, and metadata from local persistent ChromaDB.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The public repository includes the smallest local embedding model and the local reranker:

```text
models/embeddings/multilingual-e5-small/
models/rerankers/mmarco-mMiniLMv2-L12-H384-v1/
```

The app scans `models/embeddings/` and `models/rerankers/` at startup. Add another local model folder there and it appears in Streamlit without changing code. Model scanning does not load weights; weights are loaded only for indexing, search, or reranking. The app expects local Sentence Transformers models and does not call an external embedding or reranking API at runtime.

## Snowflake Environment

Copy `.env.example` to `.env` or set these variables in your shell:

```text
SNOWFLAKE_ACCOUNT
SNOWFLAKE_USER
SNOWFLAKE_PASSWORD
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
SNOWFLAKE_TABLE
SNOWFLAKE_ROLE
```

Optional column override variables are available for the Snowflake source columns. Defaults are defined in `src/config.py`.

## Build the Local Index

```powershell
python scripts/build_chroma_index.py
```

This loads Snowflake rows, hashes the prepared records and selected model folder, then reuses an existing matching Chroma collection or builds a new versioned collection:

- Loads selected columns from Snowflake.
- Cleans claim text.
- Builds event-focused embedding text.
- Embeds with `models/embeddings/multilingual-e5-small/` by default.
- Persists records to `chroma_db/`.
- Writes the active per-model manifest, for example `artifacts/index_manifest_multilingual_e5_small.json`.

To build the index with another local embedding model, place it under `models/embeddings/` and pass the folder name:

```powershell
python scripts/build_chroma_index.py --model-key multilingual-e5-base
python scripts/build_chroma_index.py --model-key multilingual-e5-large
```

Search uses the active manifest for the selected embedding model. If the same Snowflake rows and model fingerprint are indexed again, embedding is skipped and the existing Chroma collection is reused.

The same indexing flow is available in Streamlit on the `Index Setup` tab. Indexing only runs after clicking `Load or refresh index`; the app does not rebuild Chroma on startup.

## Build Clusters

```powershell
python scripts/build_clusters.py --clusters 12
```

This reads embeddings from the active Chroma collection for the selected embedding model, assigns KMeans cluster IDs, updates Chroma metadata, and writes per-model cluster artifacts.

## Seed Dummy Demo Data

For a local demo without Snowflake, seed ChromaDB with synthetic German and English claims:

```powershell
python scripts/seed_dummy_chroma.py --all-models
```

This rebuilds one local collection per discovered embedding model in `chroma_db/`, embeds 32 dummy claims with each model, assigns KMeans clusters, and writes per-model demo artifacts:

```text
claims_multilingual_e5_small_<hash>

artifacts/index_manifest_multilingual_e5_small.json
```

A single dummy index can also be rebuilt:

```powershell
python scripts/seed_dummy_chroma.py --model-key multilingual-e5-small
```

## Run the App

```powershell
streamlit run app/streamlit_app.py
```

Pages:

- Similar Claims Search
- Cluster Explorer
- Data & Embedding Diagnostics

The embedding model dropdown switches between the prebuilt local indexes. The search page also supports semantic, BM25, and hybrid retrieval modes, plus optional local cross-encoder reranking. The diagnostics page includes a local smoke-test button for the active embedding model.

## Tests

```powershell
pytest
```

## Notes

- Claims with blank descriptions are excluded from semantic indexing and counted in diagnostics.
- Claim ID, handler-like fields, payment amounts, reserve amounts, and statuses are metadata, not embedding text.
- BM25 and cross-encoder reranking are local experiment options for comparing search setups.
- Cross-encoder reranking may be slower on CPU.
- KMeans clusters are exploratory and should not be treated as authoritative business classifications.
