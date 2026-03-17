import sys
sys.path.append("./modules")
from get_data import load_hf_for_analysis
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

# Load data
hf, x_vars = load_hf_for_analysis()
df = hf.to_pandas()
df["log_co2"] = np.log1p(df["co2_cost"])
 
#  outputs
def print_diagnostics(model):
    print(f"  AIC: {model.aic:.1f}")
    print(f"  R²: {model.rsquared:.3f}")
    print(f"  Residual std err: {model.resid.std():.3f}")
    
    # Breusch-Pagan test
    bp_p = het_breuschpagan(model.resid, model.model.exog)[1]
    print(f"  Breusch-Pagan p: {bp_p:.3f}")
    
    # Shapiro-Wilk test (sample large residuals for speed)
    resid_sample = model.resid.sample(min(5000, len(model.resid)))
    sw_p = shapiro(resid_sample)[1]
    print(f"  Shapiro-Wilk p: {sw_p:.3f}")
    
    # Robust p-values for each predictor (skip intercept)
    robust = model.get_robustcov_results(cov_type="HC3")
    robust_pvalues = pd.Series(robust.pvalues, index=model.params.index)
    
    for var in model.params.index:
        if var != 'Intercept':
            print(f"  {var}: β={model.params[var]:.3f}, robust p={robust_pvalues[var]:.3f}")



# Parameters (power transform grid search)
# model comparison
models = {
    "logX": smf.ols("log_co2 ~ np.log1p(parameters)", data=df).fit(),
    "loglog": smf.ols("log_co2 ~ np.log1p(np.log1p(parameters))", data=df).fit(),
    "power": smf.ols("log_co2 ~ I(parameters**0.5)", data=df).fit(),
}
print("Model comparison AIC:", {name: mod.aic for name, mod in models.items()})

# Grid search
best_aic, best_b = np.inf, None
df["param_trans"] = 0.0
for b in np.linspace(0.1, .99, 100):
    df["param_trans"] = df["parameters"] ** b
    model = smf.ols("log_co2 ~ param_trans", data=df).fit()
    if model.aic < best_aic:
        best_aic, best_b = model.aic, b

print(f"Optimal exponent: b={best_b:.2f}, AIC={best_aic:.1f}")

# Final model
df["param_trans"] = df["parameters"] ** best_b
model_params = smf.ols("log_co2 ~ param_trans", data=df).fit()
print_diagnostics(model_params)


# Average_score 

print("\nlog_co2 ~ avg_score")

best_aic_avg, best_b_avg = np.inf, None
df["avg_trans"] = 0.0
for b in np.linspace(0.1, 5, 100):
    df["avg_trans"] = df["average_score"] ** b
    model = smf.ols("log_co2 ~ avg_trans", data=df).fit()
    if model.aic < best_aic_avg:
        best_aic_avg, best_b_avg = model.aic, b

print(f"Optimal exponent: b={best_b_avg:.2f}, AIC={best_aic_avg:.1f}")

# Final model
df["avg_trans"] = df["average_score"] ** best_b_avg
model_avg = smf.ols("log_co2 ~ avg_trans", data=df).fit()
print_diagnostics(model_avg)


# Controlled (average_score + parameters)
print("\nlog_co2 ~ parameters + average_score")

df["param_trans_ctrl"] = df["parameters"] ** best_b
df["avg_trans_ctrl"] = df["average_score"] ** best_b_avg

model_ctrl = smf.ols("log_co2 ~ param_trans_ctrl + avg_trans_ctrl", data=df).fit()
print_diagnostics(model_ctrl)