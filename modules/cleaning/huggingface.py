"""Cleaning pipeline for the HuggingFace Open LLM Leaderboard dataset."""

import polars as pl

from modules.constants import HF_RAW, HF_OUT, HF_BENCHMARKS


def clean_huggingface(
    input_path: str = HF_RAW,
    output_path: str = HF_OUT,
) -> pl.DataFrame:
    """Unnest, clean, and aggregate HuggingFace leaderboard data.
    Writes NDJSON to *output_path* and returns the cleaned DataFrame."""

    hf = pl.read_json(input_path).lazy()

    hf = hf.unnest(["model", "evaluations", "features", "metadata"])

    hf = hf.drop([
        "base_model",
        "has_chat_template",
        "hub_license",
        "id",
        "is_flagged",
        "is_merged",
        "is_not_available_on_hub",
        "sha",
        "generation",
        "hub_hearts",
        "is_moe",
        "is_official_provider",
        "submission_date",
        "type",
        "upload_date",
        "weight_type",
    ])

    hf = hf.with_columns(
        (pl.col("params_billions") * 1_000_000_000).alias("parameters")
    ).drop("params_billions")

    hf = hf.with_columns(
        pl.col("name").str.split("/").list.get(-1).alias("model")
    )

    hf = hf.group_by("model").agg([
        pl.col("architecture").first(),
        pl.col("average_score").mean(),
        pl.col("co2_cost").mean(),
        pl.col("parameters").mean(),
        pl.col("precision").first(),
        pl.col("name"),
        *[pl.col(b) for b in HF_BENCHMARKS],
    ])

    hf.sort(["architecture", "model"]).sink_ndjson(output_path)
    return hf.collect()


if __name__ == "__main__":
    clean_huggingface()
