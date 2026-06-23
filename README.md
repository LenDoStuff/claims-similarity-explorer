# Claims Similarity Explorer

Streamlit app for initializing and searching insurance-claim embeddings stored in Snowflake.

Snowpark handles source loading, text preparation, Cortex embedding generation, table persistence, filtering, and vector similarity ranking.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Snowflake Connection

The app uses Snowpark's default local Snowflake connection. Keep credentials outside the repository:

```toml
# C:\Users\<username>\AppData\Local\Snowflake\connections.toml
[myconnection]
account = "myorganization-myaccount"
user = "jdoe"
password = "..."
warehouse = "COMPUTE_WH"
database = "CLAIMS"
schema = "CLAIMS_META"
role = "..."
```

```toml
# C:\Users\<username>\AppData\Local\Snowflake\config.toml
default_connection_name = "myconnection"
```

The role needs source-table `SELECT`, permission to create or replace the derived table, and either `SNOWFLAKE.CORTEX_EMBED_USER` or `SNOWFLAKE.CORTEX_USER`.

## Configuration

The source table and column mapping are configured in `app_config.toml`:

```toml
[snowflake]
table = "SCHADEN"
row_limit = 1000

[columns]
claim_id = "claim_id"
description = "claim_description"
line_of_business = "line_of_business"
claim_type = "claim_type"
cause_of_loss = "cause_of_loss"
damaged_object = "damaged_object"
country = "country"
claim_status = "claim_status"
loss_date = "loss_date"
reserve_amount = "reserve_amount"
paid_amount = "paid_amount"
currency = "currency"
policy_type = "policy_type"
```

Every mapping key must be present. `claim_id` and `description` are required; set optional mappings to `""` to skip them. `row_limit` is optional.

## Initialize Embeddings

Use the `Index Setup` tab or run:

```powershell
python scripts/build_snowflake_embeddings.py --models voyage-multilingual-2 multilingual-e5-large
```

Initialization:

- Loads the source with `session.table()`.
- Builds the embedding text with Snowpark expressions.
- Calls Snowflake Cortex `AI_EMBED` for every selected model.
- Creates or replaces `<SOURCE_TABLE>_EMBEDDINGS` in the source table's schema.
- Stores one typed `VECTOR` column per selected model.

Running initialization again replaces the derived table and its previous model columns.
The CLI requires an explicit model list; the Streamlit multiselect starts empty.

## Search

```powershell
streamlit run app/streamlit_app.py
```

The search page lists only models present in the derived table. Queries are embedded with the selected model and can be ranked with cosine similarity, inner product, Manhattan distance, or Euclidean distance. Metadata filters, metric ordering, and limits are applied in Snowflake before final rows are converted to pandas for rendering.

## Tests

```powershell
pytest
```
