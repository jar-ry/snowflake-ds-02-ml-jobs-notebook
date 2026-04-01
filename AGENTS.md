# ML Jobs Notebook — Agent Guide

## What This Repo Is

A local-IDE ML pipeline that submits training to Snowflake via the `@remote` decorator. The notebook (`CUSTOMER_VALUE_MODEL_NOTEBOOK.ipynb`) mirrors the Snowflake Notebook version but runs locally with modular `.py` helper files.

**Use case:** Customer value regression (predict `MONTHLY_CUSTOMER_VALUE`).

## Repo Structure

```
├── CUSTOMER_VALUE_MODEL_NOTEBOOK.ipynb   # Main pipeline notebook
├── feature_engineering_fns.py            # uc01_load_data, uc01_pre_process
├── useful_fns.py                         # Session creation, Registry/FeatureStore helpers, versioning
├── conda.yml                             # Conda environment
├── connection.json.example               # Snowflake credentials template
└── connection.json                       # Actual credentials (gitignored)
```

## Environment

```bash
conda env create -f conda.yml
conda activate snowflake_ds
```

Python 3.10, key packages: `snowflake-ml-python>=1.30.0`, `xgboost`, `scikit-learn`, `altair`, `matplotlib`.

## How to Run

```bash
conda activate snowflake_ds
jupyter lab CUSTOMER_VALUE_MODEL_NOTEBOOK.ipynb
```

Run cells top-to-bottom. The `train_remote()` cell submits an ML Job — use `results.wait()` to block, `results.show_logs()` for output, `results.result()` for return value.

## Snowflake Connection

The notebook connects via `connection.json` using `useful_fns.create_SF_Session()`. Copy `connection.json.example` and fill in credentials.

Environment variable override: set `SNOWFLAKE_CONNECTION_NAME` to use a named Snowflake connection instead.

## Key Snowflake Objects

- **Database:** `RETAIL_REGRESSION_DEMO`
- **Schemas:** `DS` (raw data), `MODELLING` (Model Registry), `FEATURE_STORE`
- **Compute Pool:** `CUSTOMER_VALUE_MODEL_POOL_CPU`
- **Warehouse:** `RETAIL_REGRESSION_DEMO_WH`
- **Model:** `UC01_SNOWFLAKEML_RF_REGRESSOR_MODEL` (in `MODELLING` schema)
- **Stage:** `payload_stage` (ML Job artifacts)

## Pipeline Steps (Notebook Cells)

1. **Setup** — imports, session, database parameters
2. **Feature Store** — register `CUSTOMER` entity, create FeatureView backed by Dynamic Table
3. **Feature Engineering** — join `CUSTOMERS` + `PURCHASE_BEHAVIOR`, derive features via `feature_engineering_fns.py`
4. **Dataset** — generate versioned Snowflake Dataset from FeatureView using spine DataFrame
5. **HPO Training** — `@remote` ships `train_remote()` to compute pool; Tuner runs 10 RandomSearch trials across 3 nodes; pre-creates model in Registry before HPO to avoid race condition
6. **Explainability** — run `best_version.run(X_explain, function_name="explain")` for SHAP values, visualise with `plot_violin`
7. **Promotion** — best trial → default version, `PROD` alias, tag, copy to `PROD_SCHEMA`
8. **Inference** — deploy model as SPCS container service
9. **Monitoring** — `ModelMonitor` for drift detection

## Architecture Notes

- `train_remote()` is decorated with `@remote("CUSTOMER_VALUE_MODEL_POOL_CPU", stage_name="payload_stage", target_instances=3)` — the entire function (including nested `train()`, `build_pipeline()`, `evaluate_model()`) is serialised and shipped to the compute pool
- The `train()` function inside `train_remote()` is the per-trial HPO function executed by Ray workers
- `SnowflakeXgboostCallback` is commented out — it doesn't support `target_platforms` or `enable_explainability`. Use `exp.log_model()` instead
- Models are logged with `target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"]` and `options={"enable_explainability": True}`
- Before HPO, the code pre-creates the model in the Registry with a dummy version to avoid "Object already exists" race conditions from parallel trials

## Common Modifications

- **Change model type:** Edit `build_pipeline()` in cell 28 and update HPO search space in `train_remote()`
- **Change features:** Edit `feature_engineering_fns.py` (data loading and preprocessing)
- **Change HPO:** Modify `search_space` dict and `num_trials` in `train_remote()`
- **Change compute:** Adjust `target_instances` in the `@remote` decorator
