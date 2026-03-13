"""Cleaning pipeline for Stanford HELM benchmark results."""

import json
import os
from zipfile import ZipFile

import pandas as pd

from modules.constants import HELM_ZIP, HELM_OUT, HELM_INTERIM


def build_helm_parquet(
    zip_path: str = HELM_ZIP,
    interim_dir: str = HELM_INTERIM,
    output_path: str = HELM_OUT,
) -> pd.DataFrame:
    """Extract HELM zip, flatten JSON stats, write Parquet, and return the DataFrame.

    Note: takes ~5 min and ~4 GB RAM to run.
    The zip is only extracted once; subsequent calls reuse the interim directory."""

    if not os.path.exists(interim_dir):
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(interim_dir)

    rows = []
    for root, _dirs, files in os.walk(interim_dir):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    data = json.load(f)
                rows.append(pd.json_normalize(data))

    data = pd.concat(rows, ignore_index=True)

    drop_cols = [
        c for c in data.columns
        if c.startswith(("scenario", "name", "adapter", "data_", "metric", "groups"))
    ]
    drop_cols.remove("name.name")
    drop_cols.remove("adapter_spec.model")

    data_clean = data.drop(columns=drop_cols)
    data_clean["model"] = data["adapter_spec.model"].bfill()
    del data
    data_clean = data_clean.drop("adapter_spec.model", axis=1)
    data_clean = data_clean.dropna()
    data_clean.insert(0, "name.name", data_clean.pop("name.name"))
    data_clean = data_clean.rename(columns={"name.name": "metric"})
    data_clean.insert(0, "model", data_clean.pop("model"))

    data_clean.to_parquet(output_path, compression="zstd", index=False)
    return data_clean


if __name__ == "__main__":
    build_helm_parquet()
