"""Reusable data-loading and preprocessing helpers for EDA notebooks."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from modules.constants import HF_BENCHMARKS as DEFAULT_HF_BENCHMARKS


def load_epoch_processed(path: str = "data/02-processed/epoch_ai_parameters.ndjson") -> pd.DataFrame:
    """Load the cleaned Epoch dataset."""
    return pd.read_json(path, lines=True)


def prepare_epoch_log_views(
    epoch_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float, int]:
    """Prepare log-scale Epoch subsets and fit summary used across multiple plots."""
    epoch = epoch_df.loc[epoch_df["Parameters"].gt(0)].copy()
    epoch["log10_params"] = np.log10(epoch["Parameters"])

    epoch_compute = epoch.loc[epoch["Training compute (FLOP)"].gt(0)].copy()
    epoch_compute["log10_train_flop"] = np.log10(epoch_compute["Training compute (FLOP)"])

    corr_log = epoch_compute["log10_params"].corr(epoch_compute["log10_train_flop"])
    slope, intercept = np.polyfit(epoch_compute["log10_params"], epoch_compute["log10_train_flop"], 1)
    n_pair = len(epoch_compute)

    return epoch, epoch_compute, corr_log, float(slope), float(intercept), n_pair


def load_hf_for_eda(
    path: str = "data/02-processed/hf_parameters_co2.ndjson",
    benchmarks: Sequence[str] = DEFAULT_HF_BENCHMARKS,
) -> tuple[pl.DataFrame, list[str]]:
    """Load and normalize benchmark columns for HuggingFace EDA cells."""
    hf = pl.scan_ndjson(path)

    for bench in benchmarks:
        hf = hf.with_columns(pl.col(bench).list.first().alias(bench))
        hf = hf.unnest(bench, separator="_")

    hf = hf.drop([col for col in hf.collect_schema().names() if "_name" in col])
    hf = hf.drop_nulls(pl.selectors.matches("co2_cost|_value")).collect()

    x_vars = ["parameters"] + [f"{bench}_value" for bench in benchmarks]
    return hf, x_vars
