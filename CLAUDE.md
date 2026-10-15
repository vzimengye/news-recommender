# Project instructions

## Project goal

Build a reproducible two-stage news recommendation system using Microsoft News Dataset (MIND-small): retrieval, ranking, FastAPI serving, experiment tracking, and basic monitoring. Implement the smallest end-to-end version first (data pipeline, then a popularity retrieval + logistic regression baseline with offline evaluation, then serving/tracking/monitoring, then retrieval and ranking upgrades), then improve one component at a time and record measured results. Do not claim metric improvements without running the experiment.

## Data and licensing

- Use the official MIND-small training and validation (`dev`) archives. Read and comply with the Microsoft Research License Terms before downloading; use the data only within the granted research purposes.
- Never commit source archives, extracted records, processed datasets, user behavior logs, model artifacts derived from the dataset, or request logs. Keep them under ignored `data/`, `models/`, `mlruns/`, and `logs/` paths as appropriate.
- Preserve the original archives under `data/raw/MINDsmall_train/` and `data/raw/MINDsmall_dev/` (or document an equivalent stable layout). Do not rename source files unnecessarily.
- MIND-small has train and dev splits, not a public labeled test split. Treat `dev` as a held-out offline evaluation set and say so accurately. If a train/validation split is needed for tuning, make a chronological split within train after inspecting timestamps; do not assume a particular day boundary without checking the data. Keep all learned statistics and features causal to each impression time.
- Keep dataset download/extraction reproducible (document official URLs, expected filenames, and commands); do not hard-code credentials or silently fetch data during import/model serving.

## Engineering conventions

- Python 3.11. Put reusable implementation in `src/`, runnable entry points in `scripts/`, configuration in `configs/`, and exploratory work in `notebooks/`.
- Keep training and serving feature construction on one shared code path.
- Use deterministic seeds where supported. Keep paths and model parameters in configuration rather than scattering literals through code.
- Respect temporal ordering and impression grouping in data processing and evaluation. Prevent future information from entering features, retrieval statistics, or model fitting.
- Keep dependencies explicit and document setup and commands in `README.md`.
- Add focused tests for data parsing, temporal leakage, ranking metrics, and serving contracts as these components are implemented. Do not fabricate expected metrics; derive them from runs.
- Prefer small, reviewable changes that match the current milestone. Do not implement later milestones prematurely unless asked.

## Repository hygiene

- Ignore downloaded and derived data, environments, caches, MLflow runs, logs, and generated model/index files.
- Never commit secrets, private credentials, or dataset contents.
- When changing behavior, update the relevant README/configuration and report what was changed and how it was checked.
