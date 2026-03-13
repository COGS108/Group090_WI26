"""Shared constants: data paths and benchmark names used across all modules."""

RAW_DIR       = "data/00-raw"
INTERIM_DIR   = "data/01-interim"
PROCESSED_DIR = "data/02-processed"

HF_BENCHMARKS: tuple[str, ...] = (
    "bbh",
    "gpqa",
    "ifeval",
    "math",
    "mmlu_pro",
    "musr",
)

EPOCH_RAW   = f"{RAW_DIR}/epoch_ai_raw/all_ai_models.csv"
HF_RAW      = f"{RAW_DIR}/huggingface/formatted.json"
HELM_ZIP    = f"{RAW_DIR}/helm/run_stats.zip"

EPOCH_OUT   = f"{PROCESSED_DIR}/epoch_ai_parameters.ndjson"
HF_OUT      = f"{PROCESSED_DIR}/hf_parameters_co2.ndjson"
HELM_OUT    = f"{PROCESSED_DIR}/helm_leaderboard_all_results_clean.parquet"
HELM_INTERIM = f"{INTERIM_DIR}/helm_temp"
