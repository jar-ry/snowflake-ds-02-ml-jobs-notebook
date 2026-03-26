# 02 - ML Jobs Notebook

**Maturity Level:** ⭐⭐ Intermediate | **Runs In:** Local IDE → Snowflake Container Runtime

## Overview

Develop locally in your favorite IDE and submit compute-heavy work (HPO, training) to Snowflake as an ML Job. The `@remote` decorator ships your function and its nested dependencies to a Snowflake compute pool, so you get local debugging flexibility with cloud-scale execution.

## When to Use

- ✅ Local IDE development with full debugging, linting, and extensions
- ✅ Modular code split across `.py` helper files and notebooks
- ✅ Heavy training offloaded to a dedicated SPCS compute pool
- ✅ Teams that version-control notebooks and modules in Git
- ⚠️ Requires managing a local Python/conda environment
- ⚠️ Requires a `connection.json` file for Snowflake credentials
- ⚠️ Requires a compute pool provisioned in Snowflake

## What the Notebook Does

The pipeline (`CLV_MODEL_NOTEBOOK.ipynb`) mirrors `01_snowflake_notebooks` end-to-end, but executes training remotely via ML Jobs:

| Step | What Happens |
|------|--------------|
| **Session creation** | Connects from local Python using `connection.json` via `create_SF_Session()` |
| **Feature Store setup** | Creates `MODELLING` and `FEATURE_STORE` schemas, registers a `CUSTOMER` entity |
| **Feature engineering** | Uses imported functions from `feature_engineering_fns.py` (`uc01_load_data`, `uc01_pre_process`) |
| **FeatureView creation** | Registers a managed FeatureView backed by a Dynamic Table |
| **Dataset generation** | Builds a versioned Snowflake Dataset from the FeatureView |
| **HPO + Training (ML Job)** | The `train_remote()` function is decorated with `@remote("CLV_MODEL_POOL_CPU")` — the entire function (including nested `train()`, `build_pipeline()`, `evaluate_model()`) is shipped to the compute pool as an ML Job. The Tuner runs 10 `RandomSearch` trials across 3 target instances. |
| **Experiment tracking** | Each trial logs params, metrics, and model artifacts via `ExperimentTracking` |
| **Model promotion** | Selects best trial, sets default version, alias, tag, and copies to `PROD_SCHEMA` |
| **Inference service** | Deploys the model as a container service on SPCS |
| **Model monitoring** | Configures a `ModelMonitor` for ongoing tracking |

## Contents

```
02_ml_jobs_notebook/
├── README.md                        # This file
├── CLV_MODEL_NOTEBOOK.ipynb         # Main pipeline notebook (run locally)
├── feature_engineering_fns.py       # uc01_load_data, uc01_pre_process
├── useful_fns.py                    # Session creation, Registry/FeatureStore helpers,
                                     #   version utilities, SQL formatting
```

## How It Works

```
┌──────────────────────────┐          ┌──────────────────────────────────────┐
│      Local Machine       │          │            Snowflake                 │
│                          │          │                                      │
│  CLV_MODEL_NOTEBOOK.ipynb│          │  ┌──────────────────────────────┐    │
│  feature_engineering_fns │  ──────► │  │  Warehouse (Feature Store,   │    │
│  useful_fns.py           │  Session │  │  Dataset, FeatureViews)      │    │
│                          │          │  └──────────────────────────────┘    │
│  @remote decorator       │          │                                      │
│  ───────────────────     │  ML Job  │  ┌──────────────────────────────┐    │
│  train_remote() ─────────┼────────► │  │  Compute Pool (SPCS)         │    │
│                          │          │  │  - 3 target instances        │    │
│                          │          │  │  - Tuner + 10 HPO trials     │    │
│                          │          │  │  - Experiment Tracking       │    │
│                          │          │  └──────────────┬───────────────┘    │
│                          │          │                 │                    │
│  results.wait()    ◄─────┼──────────┤                 ▼                    │
│  results.show_logs()     │          │          Model Registry              │
└──────────────────────────┘          └──────────────────────────────────────┘
```

## Key Difference from 01_snowflake_notebooks

| | 01 Snowflake Notebooks | 02 ML Jobs Notebook |
|-|------------------------|---------------------|
| **Where code lives** | Inline in Snowflake Notebook | Local `.ipynb` + `.py` helper files |
| **Session** | `get_active_session()` (automatic) | `Session.builder.configs(...)` via `connection.json` |
| **HPO compute** | `scale_cluster(5)` — scales the notebook cluster | `@remote("CLV_MODEL_POOL_CPU", target_instances=3)` — submits an ML Job |
| **Code organization** | Self-contained single notebook | Modular: notebook imports from `feature_engineering_fns.py` and `helper/` |
| **Job lifecycle** | Synchronous (cells block) | Asynchronous: `results.wait()`, `results.show_logs()`, `results.result()` |
| **Best for** | Learning, demos, fast iteration | IDE-first teams who want SPCS-backed distributed training |

## Prerequisites

- Completed `Step01_Setup.ipynb` (creates database, tables, mock data)
- Local conda environment (`conda activate snowflake_ds`) with `snowflake-ml-python>=1.30.0`
- A `connection.json` file in the notebook's parent directory (or adjust the path in cell 6)
- A CPU compute pool (e.g. `CLV_MODEL_POOL_CPU`) provisioned in Snowflake
- A stage named `payload_stage` for ML Job artifacts

## Quick Start

```bash
conda activate snowflake_ds
jupyter lab implementations/02_ml_jobs_notebook/CLV_MODEL_NOTEBOOK.ipynb
```

Run cells top-to-bottom. The `train_remote()` call submits the job — use `results.wait()` to block until complete, then inspect with `results.show_logs()` and `results.result()`.

## Snowflake Services Used

- Feature Store (Entity, FeatureView, Dynamic Tables)
- Model Registry (versioning, aliases, tags)
- Datasets & DataConnectors
- Experiment Tracking
- ML Jobs (`@remote` decorator → Snowflake Container Runtime)
- Tuner / HPO (`tune.Tuner`, `RandomSearch`, 3 target instances)
- SPCS Model Service (real-time inference)
- Model Monitoring

## Related Repos

| Repo | Description |
|------|-------------|
| [snowflake-ds-setup](https://github.com/jar-ry/snowflake-ds-setup) | Environment setup, data generation, and helper utilities (run this first) |
| [snowflake-ds-01-notebooks](https://github.com/jar-ry/snowflake-ds-01-notebooks) | Same pipeline running entirely in Snowflake UI (no local setup) |
| [snowflake-ds-03-ml-jobs-framework](https://github.com/jar-ry/snowflake-ds-03-ml-jobs-framework) | Production-grade modular framework using `submit_directory` |
| [snowflake-ds-04-feature-store](https://github.com/jar-ry/snowflake-ds-04-feature-store) | Split repo: Feature Store with FeatureViews and Versioned Datasets |
| [snowflake-ds-04-ml-training](https://github.com/jar-ry/snowflake-ds-04-ml-training) | Split repo: ML Training with training, promotion, inference, monitoring |
