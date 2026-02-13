import polars as pl
import os

# Directories need closing slashes /
generate_files = False
src = "./data/Input/huggingface/formatted.json"
temp_dir = "./data/temp/"
output_dir = "./data/Output/"
individual_scores = False # This keeps dictionaries of each test, e.g. ifeval, gqpa, mmlu, math. Average score kept regardless.

# Open raw from src
hf = pl.read_json(src).lazy()

# Seperate model into individual columns. Polars interprets dictionaries as "struct" dtype.
hf = hf.unnest(["model", "evaluations", "features", "metadata"])

# Drop columns
hf = hf.drop([
    "base_model",
    "has_chat_template",
    "hub_license",
    "id",
    "is_flagged",
    "is_merged",
    "is_not_available_on_hub",
    "precision",
    "sha",


])


# Potentially dropped columns (we can discuss)
hf = hf.drop([
    "generation",
    "hub_hearts",
    "is_moe", # Unknown var
    "is_official_provider",
    "submission_date",
    "type", # Type of LLM?
    "upload_date",
    "weight_type", # if using adapter https://huggingface.co/docs/hub/adapters

])

# creating analysis files
if generate_files:
    for cols in hf.drop(pl.col("name")).collect().columns:
        hf.select([cols, "name"]).unique().sort(by=cols, descending=True).sink_ndjson(f"{temp_dir}hf_{cols}.ndjson")

# expand params_billions to parameters (to match epoch dataset)
hf = hf.with_columns((pl.col("params_billions") * 1000000000).alias("parameters")).drop("params_billions")

# Drop individual score columns if individual_scores is False
if not individual_scores:
    hf = hf.drop([
        "bbh",
        "gpqa",
        "ifeval",
        "math",
        "mmlu_pro",
        "musr",
    ])

# Add column with model name without contributor
hf = hf.with_columns((pl.col("name").str.split("/").list.get(-1)).alias("model"))

# duplicates exist, collapse by model
hf = hf.group_by("model").agg([
    pl.col("architecture").first(), 
    pl.col("average_score").mean(), 
    pl.col("co2_cost").mean(),
    pl.col("parameters").mean(),
    pl.col("name"), 
])

hf.sort(["architecture", "model"]).sink_ndjson(f"{output_dir}hf_parameters_co2.ndjson")