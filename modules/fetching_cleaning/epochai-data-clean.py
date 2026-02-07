import polars as pl
import os

# Directories need closing slashes /
generate_files = False
src = "./data/Input/epoch_ai_raw/all_ai_models.csv"
temp_dir = "./data/temp/"
output_dir = "./data/Output/"

# Reads epoch.ai data, defines schema for columns
epoch = pl.scan_csv(src, schema_overrides={
    "Batch size":pl.Float64,
    "Finetune compute (FLOP)":pl.Float64,
    "Last modified":pl.Datetime,
    "Parameters":pl.Float64, 
    "Publication Date":pl.Date,
    "Training chip-hours":pl.Float64,
    "Training compute (FLOP)":pl.Float64,
    "Training dataset size (total)":pl.Utf8, # Casted to numeric later.


})

# Filter epoch.ai data for only language models
epoch = epoch.filter(pl.col("Domain").str.contains("Language"))

# Create Bool indicator for model notability - Done here to drop other notability columns
epoch = epoch.with_columns(pl.col("Notability criteria").is_not_null().alias("Notable"))

# Discard columns
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

])

# Potentially Discarded columns (we can discuss)
epoch = epoch.drop([
    "Batch size notes", # duplicate data?
    "Confidence", # unsure what it is
    "Country (of organization)",
    "Epochs", # unsure what it is
    "Hardware quantity", # unsure
    "Hardware utilization (HFU)", # unsure
    "Hardware utilization (MFU)", # unsure
    "Organization categorization",
    "Organization",
    "Parameters notes",
    "Possibly over 1e23 FLOP", # duplicate data? Is a Bool
    "Training code accessibility", # could compare open-source to closed-source models?
    "Training compute lower bound", # unknown how many have this var
    "Training compute upper bound", # same as lower bound
    "Training hardware", # Additional stat?
    "Training power draw (W)", # No unit given, if we can determine.
    "Training time (hours)", # No metric

])

# Variable specific changes
epoch = epoch.with_columns(pl.col("Training dataset size (total)").str.split(",").list.first().cast(pl.Float64))
epoch = epoch.with_columns(pl.col("Frontier model").is_not_null())
epoch = epoch.with_columns(pl.col("Foundation model").is_not_null())


# Generate files for each column (if generate_files = True)
if generate_files:
    for cols in epoch.drop(pl.col("Model")).collect().columns:
        epoch.select([cols, "Model"]).unique().sort(by=cols).sink_ndjson(f"{temp_dir}epoch_data_{cols}.ndjson")


# Drop nulls in "Parameters"
epoch = epoch.drop_nulls(pl.col(["Parameters"]))


# Export
epoch.sink_ndjson(f"{output_dir}epoch_ai_parameters.ndjson")