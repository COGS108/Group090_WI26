import polars as pl
import os


# Extract epoch-ai zip file data into new folder /data/ai_models/
# Generates file data_test 
os.chdir("./results")
pl.scan_csv("../data/epoch_ai_raw/notable_ai_models.csv", schema_overrides={"Training dataset size (total)":pl.Utf8}).sink_ndjson("epoch_ai.ndjson")


