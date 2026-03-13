"""Analysis helpers for the HuggingFace dataset (load + normalize only; no plots)."""

import polars as pl

from modules.constants import HF_BENCHMARKS
from modules.get_data import load_hf_for_eda


def load_hf_with_benchmarks() -> tuple[pl.DataFrame, list[str]]:
    """Load cleaned HF data with all benchmark columns unnested and nulls dropped.

    Returns the same (hf, x_vars) tuple as load_hf_for_eda so notebooks can
    swap in this call without changing any downstream variable names.

    This is the single authoritative place for HF unnesting; hf_EDA.py used
    to duplicate this logic inline.
    """
    return load_hf_for_eda(benchmarks=HF_BENCHMARKS)
