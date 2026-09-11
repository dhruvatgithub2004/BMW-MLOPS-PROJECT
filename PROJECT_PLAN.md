# BMW Price Prediction — MLOps Execution Plan

## 1. Objective

Build an end-to-end, reproducible MLOps pipeline that trains a regression model to predict
the resale **price** of a used BMW car from its listing attributes, and serves that model
through a Flask API — with every step (data, code, params, model, metrics) versioned and
reproducible via DVC + MLflow.

## 2. Dataset

Source: `bmw.csv` (Kaggle used-car listings), currently referenced from the exploratory
notebook at `notebooks/BMW car prediction.ipynb`.

| Column         | Type        | Notes                                  |
|----------------|-------------|-----------------------------------------|
| `model`        | categorical | BMW model name (e.g. "3 Series")        |
| `year`         | numeric     | registration year                       |
| `price`        | numeric     | **target**                              |
| `transmission` | categorical | Manual / Automatic / Semi-Auto          |
| `mileage`      | numeric     |                                          |
| `fuelType`     | categorical | Petrol / Diesel / Hybrid / Electric     |
| `tax`          | numeric     | road tax                                |
| `mpg`          | numeric     | fuel economy                            |
| `engineSize`   | numeric     | litres                                  |

Task type: **regression**. Evaluation metrics: R², MAE, RMSE.

## 3. Tech stack (already scaffolded in this repo)

| Concern              | Tool                                             |
|-----------------------|--------------------------------------------------|
| Project layout        | cookiecutter-data-science (`src/`, `data/`, `models/`, `notebooks/`) |
| Data/model versioning | DVC — remote `mylocal` → `../local_s3` (`.dvc/config`) |
| Experiment tracking    | MLflow, tracking server on DagsHub (`dhruvatgithub2004/BMW-MLOPS-PROJECT`) |
| Modeling               | scikit-learn (`ColumnTransformer` + `OneHotEncoder`/`StandardScaler`, `LinearRegression`, `RandomForestRegressor`) |
| Config                 | `python-box` + `PyYAML` (`params.yaml`, `config.yaml`) |
| Serving                | Flask + Flask-Cors |
| Logging                | `src/__init__.py` → `logger` (writes to `logs/running_logs.log` + stdout) |

## 4. Repo status check (what exists vs. what's a placeholder)

- ✅ Folder structure, `setup.py`, `requirements.txt`, DVC initialized, logging in `src/__init__.py`.
- ✅ EDA + a first modeling pass done in `notebooks/BMW car prediction.ipynb` (LinearRegression
  vs RandomForestRegressor, feature importance, MLflow/DagsHub logging prototype).
- ✅ `src/data/data_ingestion.py` rewritten for the BMW dataset: loads `data/raw/bmw.csv`,
  validates expected columns, splits train/test per `params.yaml`, writes to `data/interim/`.
- ✅ `src/features/build_features.py` builds the `ColumnTransformer` (OneHotEncoder + StandardScaler),
  fits it on train, saves `models/preprocessor.pkl` and the transformed arrays to `data/processed/`.
- ✅ `src/model/train_model.py` trains LinearRegression + RandomForestRegressor on the processed
  arrays, logs params/metrics/model to MLflow (DagsHub-backed), and saves the best model to
  `models/model.pkl`. **Not yet run** — needs a one-time DagsHub login (`dagshub.init(...)` opens
  a browser auth flow the first time, or set a `DAGSHUB_USER_TOKEN` env var) before it can push runs.
- ✅ `config.yaml` / `params.yaml` added at repo root, read via `src/utils/common.py` (`read_yaml`,
  `create_directories`) using `python-box`'s `ConfigBox` + `ensure_annotations`.
- ⚠️ `src/model/predict_model.py`, `src/visualization/visualize.py` are still **empty**.
- ⚠️ No `dvc.yaml` yet — the DVC pipeline stages haven't been wired up.
- ⚠️ No `app.py` / Flask entrypoint yet.

## 5. Execution phases

### Phase 0 — Housekeeping
- [x] Delete/rewrite the mismatched boilerplate in `data_ingestion.py`.
- [x] Add `config.yaml` (paths: raw/interim/processed data, model dir, MLflow/DagsHub settings) and
      `params.yaml` (test_size, random_state, model hyperparameters) — read via `python-box` + `PyYAML`.
- [ ] Confirm `.env` holds any secrets needed (DagsHub token, S3 creds) and stays git-ignored (it already is).

### Phase 1 — Data ingestion (`src/data/data_ingestion.py`)
- [x] Load `bmw.csv` from `data/raw/` (copied in from the local Kaggle download).
- [x] Basic schema/sanity checks (expected columns present).
- [x] Train/test split (`test_size` from `params.yaml`), written to `data/interim/train.csv` and
      `data/interim/test.csv`.
- [x] `dvc add data/raw/bmw.csv` — raw data is now DVC-tracked (`data/raw/bmw.csv.dvc`), and the
      blanket `/data/` rule was removed from `.gitignore` in favor of the per-directory
      `.gitignore` files DVC manages itself (see Phase 6).

### Phase 2 — EDA (already mostly done in the notebook)
- [ ] Port the key findings from `notebooks/BMW car prediction.ipynb` into short notes here or
      in `reports/`: which features correlate with price, categorical cardinality, outliers in
      `price`/`mileage`, and any rows worth dropping (e.g. implausible `year` or `mpg` values).
- [ ] Save the important plots (`price` vs `engineSize`, feature importance, actual-vs-predicted)
      into `reports/figures/` via `src/visualization/visualize.py`.

### Phase 3 — Feature engineering (`src/features/build_features.py`)
- [x] Build the `ColumnTransformer` (OneHotEncoder for `model`/`transmission`/`fuelType`,
      StandardScaler for numeric columns) as a reusable function, not inline notebook code.
- [x] Persist the fitted transformer (`joblib`) to `models/preprocessor.pkl` so training and
      inference use the identical transform.
- [x] Write transformed train/test arrays (`X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`)
      to `data/processed/`.

  > Design note: kept the preprocessor and estimator as two separate artifacts
  > (`preprocessor.pkl` + `model.pkl`) rather than one bundled sklearn `Pipeline`, so the
  > feature-engineering DVC stage and the training DVC stage stay independently re-runnable.
  > `predict_model.py` (Phase 7) will need to load and apply both.

### Phase 4 — Model training (`src/model/train_model.py`)
- [x] Train candidate models: `LinearRegression` (baseline) and `RandomForestRegressor`
      (primary candidate) on the processed arrays; hyperparameters sourced from `params.yaml`.
- [x] Compute R², MAE, RMSE on the held-out test set.
- [x] Log params, metrics, and the fitted model to MLflow (tracking URI + DagsHub project wired
      up via `config.yaml`'s `mlflow` section and `dagshub.init(...)`).
- [x] Save the best-scoring model to `models/model.pkl` (`joblib.dump`).
- [x] **Executed** on 2026-09-11 — authenticated via DagsHub OAuth (`dagshub.init(...)`), both runs
      logged successfully:

      | Model | R² | MAE | RMSE |
      |---|---|---|---|
      | Linear Regression | 0.860 | 2822 | 4258 |
      | **Random Forest Regressor** | **0.943** | **1608** | **2724** |

      Runs viewable at https://dagshub.com/dhruvatgithub2004/BMW-MLOPS-PROJECT.mlflow/#/experiments/0.
      Random Forest won and was saved to `models/model.pkl`.
- [ ] Log the actual-vs-predicted plot as an MLflow artifact (currently only in the notebook).
- [ ] Register the saved model in the MLflow Model Registry.

### Phase 5 — Model evaluation & selection
- [x] Compare runs in the MLflow/DagsHub UI; Random Forest Regressor wins on all three metrics
      (R²/MAE/RMSE) — no tradeoff to reason about, it's a strict improvement over Linear Regression.
- [ ] Promote the winning run's model version to a "staging"/"production" alias in the registry.

### Phase 6 — DVC pipeline (`dvc.yaml`)
- [x] Define stages: `data_ingestion → build_features → train`, each stage's `cmd`, `deps`, `params`,
      and `outs` wired to the scripts above.

  > Design note: no separate `evaluate` stage — `train_model.py` already evaluates both candidates
  > on the test set as part of training, so the `train` stage writes `reports/metrics.json`
  > (`cache: false`, tracked in git directly) as its `metrics` output instead of a redundant stage
  > that would reload the model and recompute the same numbers.

- [x] `dvc repro` reproduces the whole pipeline deterministically — verified: ran it twice back to
      back, second run reported "Data and pipelines are up to date" for all 3 stages.
- [x] `dvc metrics show` renders `reports/metrics.json` as a table (best model + both candidates'
      R²/MAE/RMSE), so metrics are diffable across commits without opening DagsHub.
- [ ] Commit `dvc.lock` alongside code changes so pipeline state is reproducible from git history.

  > Found and fixed two real bugs while building this:
  > 1. `dvc.yaml`'s output directories (`data/interim/`, `data/processed/`) weren't getting their
  >    `.gitignore` entries written because those stages had already been run manually (outside
  >    DVC) before `dvc.yaml` existed, so DVC's first `dvc repro` saw matching output hashes and
  >    treated the stages as already-satisfied without ever "checking them in." Fixed by forcing
  >    `dvc repro -f data_ingestion build_features` once, which made DVC do its normal bookkeeping.
  >    Verified with `git add -A --dry-run`: only `.gitignore`/`.gitkeep`/`.dvc` pointer files and
  >    `dvc.lock`/`dvc.yaml`/`reports/metrics.json` would be staged — no raw/interim/processed data.
  > 2. `dvc repro` crashed running the `train` stage: mlflow writes an emoji (`🏃`) to stdout, which
  >    crashes on Windows' default `cp1252` console codepage. Fixed permanently in `src/__init__.py`
  >    by reconfiguring `sys.stdout`/`sys.stderr` to UTF-8 on import, rather than relying on
  >    `PYTHONUTF8=1` being set by whoever runs the pipeline.
  > 3. The root `.gitignore`'s blanket `/data/` rule was silently swallowing DVC's own `.dvc`
  >    pointer files (`dvc add` failed with "bad DVC file name ... is git-ignored"). Replaced it
  >    with a comment — DVC now manages exclusion per-directory itself.

### Phase 7 — Inference (`src/model/predict_model.py`)
- [x] Load `models/model.pkl` and the fitted `models/preprocessor.pkl` (lazily cached module-level).
- [x] Expose `predict(raw_input: dict) -> float` that takes raw feature values (same shape as the
      training columns minus `price`) and returns a predicted price.

  > Found and fixed a real bug while wiring this up: `data/raw/bmw.csv`'s `model` column has a
  > leading space (`" 3 Series"`, not `"3 Series"`). The preprocessor's `OneHotEncoder` was fit on
  > those un-stripped strings, so any API caller sending the "normal" spelling would've silently
  > hit `handle_unknown="ignore"` and gotten a degraded, wrong prediction with no error. Added
  > `clean_categoricals()` to `data_ingestion.py` (strips whitespace on all string columns) and
  > **re-ran ingestion → features → training** so the artifacts are consistent. Same metrics as
  > before (Random Forest R²=0.943), just built on cleaned category strings.

### Phase 8 — Serving (Flask API)
- [x] Flesh out `main.py` with a `/predict` POST endpoint that accepts JSON matching the raw
      feature schema and returns `{"price": ...}`.
- [x] Add `/health` endpoint for liveness checks.
- [x] Enable CORS via `Flask-Cors`.
- [x] Basic input validation — missing fields return a 400 with the list of missing field names.
- [x] Manually tested locally: `/health` → `{"status": "ok"}`; a full `/predict` payload → a sane
      price (~$21.5k for a 2018 diesel automatic 3 Series); a partial payload → 400 with the
      missing-fields list.

### Phase 9 — Testing
- [x] Added `pytest` + `pytest.ini` (`pythonpath = .`, `testpaths = tests`) so `import src`/`import main`
      resolve without an editable install.
- [x] `tests/test_build_features.py` — `build_preprocessor` routes categorical/numeric columns
      correctly, output shape matches row count, and unseen categories at transform time don't raise.
- [x] `tests/test_predict_model.py` — `predict()` is deterministic for a fixed input and raises
      `KeyError` on missing fields.
- [x] `tests/test_app.py` — Flask smoke test via `app.test_client()`: `/health`, a full `/predict`
      payload, a payload missing fields (400), and a non-JSON body (400).

  > Design note: `tests/conftest.py` fits a tiny synthetic preprocessor+model in-memory and
  > monkeypatches `predict_model._preprocessor`/`_model` rather than loading the real
  > `models/*.pkl`. Those are DVC-tracked, not git-tracked — a fresh clone won't have them without
  > `dvc pull`/`dvc repro` first, which would make the test suite depend on pipeline state instead
  > of just the code. Verified by moving `models/` aside entirely: all 9 tests still pass.
  > `test_environment.py` (Python-version check) was left as-is — different purpose, not a pytest suite.

- [x] Ran `python -m pytest -v`: **9 passed**.

### Phase 10 — CI (optional but recommended)
- [ ] GitHub Actions workflow: install deps, run tests, run `dvc repro --dry` (or a lightweight
      subset) on PRs to catch pipeline breakage before merge.

### Phase 11 — Deployment
- [x] Dockerize the Flask app: `Dockerfile` (`python:3.13-slim`, gunicorn, listens on port 7860),
      `.dockerignore`. Kept — still usable for local/self-hosted/other-platform deployment even
      though it's not what ended up on Hugging Face (see below).
- [x] Target platform decided: **Hugging Face Spaces**, but the **SDK flipped from Docker to
      Gradio partway through** — the user's HF account shows Docker Spaces gated behind a paid
      plan (Static/Gradio free, Docker marked "Paid"). Gradio doesn't change what "end-to-end
      MLOps" means here: ingestion/features/training/tracking (DVC+MLflow/DagsHub)/tests all stay
      exactly as built — only the serving presentation layer changed. Gradio also auto-generates
      a REST API endpoint alongside the UI, so this isn't even a capability downgrade.

  > Design note: baking `models/model.pkl` (145MB) + `models/preprocessor.pkl` into the image via
  > `COPY` rather than pulling from the MLflow registry at container start. Reason: our DVC remote
  > (`local_s3/`) is a local folder, not reachable from HF's build servers, and MLflow model-registry
  > pull-at-startup adds a DagsHub auth dependency to the serving container for no real benefit at
  > this project's scale. Tradeoff: redeploying after retraining means rebuilding the image (or
  > later wiring a registry pull — noted as a fast-follow, not done now).
  > Also pinned `scikit-learn==1.8.0` and `joblib==1.5.3` in `requirements-serving.txt` — the
  > versions the model was actually pickled with — so the container can't silently drift onto an
  > incompatible sklearn version at build time.
  > Split out `requirements-serving.txt` (Flask/gunicorn/pandas/numpy/scikit-learn/joblib/
  > python-box/pyYAML/ensure only) from the full `requirements.txt`, after the first Docker build
  > attempt spent ~5 minutes installing the entire training/dev toolchain (mlflow, dagshub, dvc,
  > matplotlib, pytest, celery, fastapi, sqlalchemy...) into what should be a small serving image.
  > The slim build finished dependency install in ~99s.

- [x] Built and tested the image locally end to end: `docker build` → `docker run -p 7860:7860` →
      `/health` returned `{"status":"ok"}`, `/predict` returned the same price ($21,551.11) as the
      non-containerized test earlier, and a missing-fields request correctly 400'd. Final image
      size: ~1.04GB (dominated by the 145MB model + scipy/numpy/pandas/scikit-learn wheels).
- [x] Built `space/` — a self-contained Gradio app for the HF Space: `space/app.py` (loads
      `models/*.pkl` relative to itself rather than importing `src`, so the folder can be copied
      standalone into a fresh Space repo), `space/requirements.txt` (gradio + the same pinned
      `scikit-learn==1.8.0`/`joblib==1.5.3`), `space/README.md` (Spaces frontmatter, `sdk: gradio`).
      Dropdown choices, year range, engine-size range etc. pulled from the actual training data,
      not guessed.
- [x] Tested locally end to end: installed `gradio`, called `predict_price()` directly (same
      £21,551.10 as every earlier test), then launched the real Gradio server and hit its
      auto-generated REST endpoint (`POST /gradio_api/call/predict_price` → poll by `event_id`) —
      also returned £21,551.10. Confirmed `python -m pytest` (9 passed) still green after the
      gradio install touched shared `fastapi`/`starlette` versions.
- [ ] Push `space/` to an actual Hugging Face Space (needs the user's HF login — not something
      done from here): create a Space with SDK **Gradio**, then either copy `space/`'s contents
      into that Space's git repo and push, or `huggingface-cli upload`. Confirm it builds and the
      public URL serves predictions.

### Phase 12 — Monitoring & maintenance
- [ ] Log incoming prediction requests/responses for later drift analysis.
- [ ] Periodically re-run the pipeline on fresh data and compare metrics against the current
      production model before promoting a replacement.

## 6. Immediate next steps

1. ~~Rewrite `src/data/data_ingestion.py` for the BMW dataset (Phase 1).~~ Done — verified it
   runs end-to-end against `data/raw/bmw.csv`.
2. ~~Add `params.yaml` / `config.yaml`.~~ Done.
3. ~~Move the notebook's feature engineering into `src/features/build_features.py`.~~ Done — verified
   it runs and produces a (8624, 37) / (2157, 37) train/test transform.
4. ~~Move the notebook's training + MLflow logging into `src/model/train_model.py`.~~ Done and
   **run** on 2026-09-11 — Random Forest Regressor won (R²=0.943) and is saved to `models/model.pkl`;
   both runs are logged at https://dagshub.com/dhruvatgithub2004/BMW-MLOPS-PROJECT.mlflow.
5. **Next up:** `src/model/predict_model.py` (Phase 7) — load `models/preprocessor.pkl` +
   `models/model.pkl` and expose a `predict(raw_input: dict) -> float` function.
6. Then: the Flask `app.py` (Phase 8) with a `/predict` endpoint wrapping that function.
