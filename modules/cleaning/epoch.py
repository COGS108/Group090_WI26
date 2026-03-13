"""Cleaning pipeline for the Epoch AI dataset."""

import polars as pl

from modules.constants import EPOCH_RAW, EPOCH_OUT


def clean_epoch(
    input_path: str = EPOCH_RAW,
    output_path: str = EPOCH_OUT,
) -> pl.DataFrame:
    """Load raw Epoch AI CSV, filter to language models, drop unused columns,
    write NDJSON to *output_path*, and return the cleaned DataFrame."""

    epoch = pl.scan_csv(input_path, schema_overrides={
        "Batch size": pl.Float64,
        "Finetune compute (FLOP)": pl.Float64,
        "Last modified": pl.Datetime,
        "Parameters": pl.Float64,
        "Publication Date": pl.Date,
        "Training chip-hours": pl.Float64,
        "Training compute (FLOP)": pl.Float64,
        "Training dataset size (total)": pl.Utf8,
    })

    epoch = epoch.filter(pl.col("Domain").str.contains("Language"))

    epoch = epoch.with_columns(pl.col("Notability criteria").is_not_null().alias("Notable"))

    epoch = epoch.drop([
        "Abstract",
        "Accessibility notes",
        "Approach",
        "Archived links",
        "Authors",
        "Base model",
        "Citations",
        "Dataset size notes",
        "Finetune compute notes",
        "Hugging Face developer id",
        "Link",
        "Model accessibility",
        "Notability criteria notes",
        "Notability criteria",
        "Numerical format",
        "Post-training compute (FLOP)",
        "Post-training compute notes",
        "Reference",
        "Task",
        "Training cloud compute vendor",
        "Training compute cost (2023 USD)",
        "Training compute estimation method",
        "Training compute notes",
        "Training data center",
        "Training time notes",
        "Utilization notes",
        "WikiText and Penn Treebank data",
        "Batch size notes",
        "Confidence",
        "Country (of organization)",
        "Epochs",
        "Hardware quantity",
        "Hardware utilization (HFU)",
        "Hardware utilization (MFU)",
        "Organization categorization",
        "Organization",
        "Parameters notes",
        "Possibly over 1e23 FLOP",
        "Training code accessibility",
        "Training compute lower bound",
        "Training compute upper bound",
        "Training power draw (W)",
        "Training time (hours)",
    ])

    epoch = epoch.with_columns(
        pl.col("Training dataset size (total)").str.split(",").list.first().cast(pl.Float64)
    )
    epoch = epoch.with_columns(pl.col("Frontier model").is_not_null())
    epoch = epoch.with_columns(pl.col("Foundation model").is_not_null())

    epoch = epoch.drop_nulls(pl.col(["Parameters"]))

    epoch.sink_ndjson(output_path)
    return epoch.collect()


if __name__ == "__main__":
    clean_epoch()
