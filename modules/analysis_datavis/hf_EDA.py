import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Plot if true, makes the benchmark plots stacked if true
make_plots = True

# load hf dataset
src = "./data/Output/hf_parameters_co2.ndjson"
hf = pl.scan_ndjson(src)

# unnest specific benchmarks (json origin)
benchmarks = ["bbh", "gpqa", "ifeval", "math", "mmlu_pro", "musr",]
for bench in benchmarks:
    hf = hf.with_columns(pl.col(bench).list.first().alias(bench))
    hf = hf.unnest(bench, separator="_")

# drop unnecesscary name cols
hf = hf.drop([col for col in hf.collect_schema().names() if "_name" in col])

# drop nulls for data we want to look at
hf = hf.drop_nulls(pl.selectors.matches("co2_cost|_value")).collect()
x_vars=[
    "parameters",
    "bbh_value",
    "gpqa_value",
    "ifeval_value",
    "math_value",
    "mmlu_pro_value",
    "musr_value",
]

# Describe variables
with pl.Config(tbl_cols=100):
    print("Summary Statistics for HuggingFace Dataset")
    print(hf.drop([
        "architecture",
        "model",
        "precision",
        "name",

    ]).describe())

normalized_numeric = [
    "co2_cost",
    "parameters",
    "average_score",
    "bbh_normalized_score",
    "gpqa_normalized_score",
    "ifeval_normalized_score",
    "math_normalized_score",
    "mmlu_pro_normalized_score",
    "musr_normalized_score",
]


# Correlation Matrix
corr_df = hf.with_columns([
    pl.col('co2_cost').log1p().alias('log_co2'),
])
corr_df = corr_df.select(["log_co2"] + [col for col in normalized_numeric if col not in ["co2_cost"]]).corr()
with pl.Config(tbl_cols=100):
    print("Correlation Matrix for Numerical Variables")
    print(corr_df)

co2 = hf.select("co2_cost").to_numpy().flatten()

# generate plots to look at co2_cost
fig, axs = plt.subplots(1, 4, figsize=(12, 6))
axs[0].hist(co2, edgecolor="black")
axs[0].set_title("co2_cost distribution")
axs[0].set_yscale("log")

log = np.log1p(co2)
axs[1].hist(log, edgecolor="black")
axs[1].set_title("co2_cost distribution log")
axs[1].set_yscale("log")

axs[2].boxplot(co2, vert=True)
axs[2].set_title('co2_cost outliers')

axs[3].boxplot(log, vert=True)
axs[3].set_title('log_co2 outliers')


# This code below generates benchmark to co2 subplots, vertically.
fig, axs = plt.subplots(len(x_vars), 2, figsize=(12, 24))
axs = axs.flatten(order="F")
axs[0].set_title("benchmarks : co2_cost")
axs[7].set_title("benchmarks : log_co2")
for var in range(len(x_vars)):
    x = hf.select(x_vars[var]).to_numpy().flatten()
    m, b = np.polyfit(x, co2, 1)
    regline = np.linspace(x.min(), x.max())
    axs[var].scatter(hf.select(x_vars[var]).to_numpy(), co2, alpha=0.5)
    axs[var].plot(regline, m*regline+b, color="black")
    axs[var].set_xlabel(x_vars[var])
    axs[var].set_ylabel(f"co2_cost")

for var in range(len(x_vars)):
    x = hf.select(x_vars[var]).to_numpy().flatten()
    m, b = np.polyfit(x, log, 1)
    regline = np.linspace(x.min(), x.max())
    axs[var+len(x_vars)].scatter(hf.select(x_vars[var]).to_numpy(), log, alpha=0.5)
    axs[var+len(x_vars)].plot(regline, m*regline+b, color="black")
    axs[var+len(x_vars)].set_xlabel(x_vars[var])
    axs[var+len(x_vars)].set_ylabel(f"log_co2")

print("List of architectures in HuggingFace Dataset")
print(hf.select(pl.col("architecture")).unique().sort(by="architecture").to_numpy())

# Catagorical Variable analysis
fig, axs = plt.subplots(1, 6, figsize=(12, 6))
log = hf.with_columns([
    pl.col("co2_cost").log1p().alias("log_co2")
])
precision = hf.select("precision").unique().sort(by="precision").to_numpy().flatten()

axs[0].set_ylabel("co2_cost")
axs[0].set_title("Precision : co2_cost")

for cat in range(len(precision)):
    axs[cat].boxplot(hf.filter(pl.col("precision")==precision[cat]).select("co2_cost"))
    axs[cat].set_title(f"{precision[cat]} : co2_cost")

for cat in range(len(precision)):
    axs[cat+3].boxplot(log.filter(pl.col("precision")==precision[cat]).select("log_co2"))
    axs[cat+3].set_title(f"{precision[cat]} : log_co2")


plt.tight_layout()
plt.show()