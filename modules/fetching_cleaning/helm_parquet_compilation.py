# based on helm_csv_compilation.ipynb
# takes ~5min + 4gb ram to run. 

import os
import json
import pandas as pd
from zipfile import ZipFile

src = "./data/Input/helm/run_stats.zip"
temp_dir = "./data/temp/"
output_dir = "./data/Output/"


# Check if the data is extracted, if not, extract the zip into /temp, make sure it has enough space. ~1.3 gb (time intensive, would rather only do once.)
if not os.path.exists(f"{temp_dir}helm_temp/"):
    with ZipFile(src, "r") as zip_ref:
        zip_ref.extractall(f"{temp_dir}helm_temp/")

rows = []

# access all subdirectories
for root, dirs, files in os.walk(f"{temp_dir}helm_temp/"):
    for file in files:
        if file.endswith(".json"):
            filepath = os.path.join(root, file)
            
            with open(filepath, "r") as f:
                data = json.load(f)
            
            flat = pd.json_normalize(data)
            rows.append(flat)

data = pd.concat(rows, ignore_index=True)
drop_cols = [c for c in data.columns if c.startswith(("scenario","name","adapter","data_","metric","groups"))]
drop_cols.remove("name.name")
drop_cols.remove("adapter_spec.model")
data_clean = data.drop(columns=drop_cols)
data_clean["model"] = data["adapter_spec.model"].bfill()
del data
data_clean = data_clean.drop("adapter_spec.model", axis=1)
data_clean = data_clean.dropna()
data_clean.insert(0, "name.name", data_clean.pop("name.name"))
data_clean = data_clean.rename(columns={'name.name':'metric'})
data_clean.insert(0, "model", data_clean.pop("model"))
data_clean.to_parquet(f"{output_dir}helm_leaderboard_all_results_clean.parquet", compression = "zstd", index=False)