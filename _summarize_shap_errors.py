#!/usr/bin/env python3
# ---------------------------------------------------------------
#  Summarise SHAP‑error metrics  (Cosine‑Error & NRMSE_max)
# ---------------------------------------------------------------
import os, warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = "./interim_results/_9_shap_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 1) Load CSV  +  ensure nrmse_max exists
# ────────────────────────────────────────────────────────────────
df = pd.read_csv("shap_errors_summary.csv").drop_duplicates()

if "nrmse_max" not in df.columns:
    if "max_abs_shap" not in df.columns:
        raise ValueError(
            "CSV lacks both 'nrmse_max' and 'max_abs_shap'.\n"
            "Provide at least one (max_abs_shap = largest |actual SHAP| per row)."
        )
    df["nrmse_max"] = (
        df["rmse"] / df["max_abs_shap"].abs().replace(0, pd.NA)
    ).fillna(0)

# ────────────────────────────────────────────────────────────────
# 2) Aggregate  (mean across environments)
# ────────────────────────────────────────────────────────────────
summary = (
    df.groupby(["variant", "KPM"])
      .agg(cosine_error=("cosine_error", "mean"),
           nrmse_max   =("nrmse_max",    "mean"))
      .reset_index()
)

print("\n📊  Avg. Cosine‑Error & NRMSE_max per Variant + KPM\n")
print(summary.to_string(index=False, float_format="%.4f"))

# ────────────────────────────────────────────────────────────────
# 3) Save .dat tables  (Variant  CosErr  NRMSE_max)
# ────────────────────────────────────────────────────────────────
for kpm, sub in summary.groupby("KPM"):
    out_dat = f"{OUT_DIR}/summary_{kpm}_nrmse.dat"
    sub[["variant", "cosine_error", "nrmse_max"]] \
        .rename(columns={"variant":     "Variant",
                         "cosine_error":"CosineError",
                         "nrmse_max":   "NRMSE_max"}) \
        .to_csv(out_dat, sep="\t", index=False)
    print(f"💾  Saved → {out_dat}")

# ────────────────────────────────────────────────────────────────
# 4) Bar‑plots  (CosErr vs NRMSE_max) for two KPIs
# ────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
melted = pd.melt(
    summary,
    id_vars=["variant", "KPM"],
    value_vars=["cosine_error", "nrmse_max"],
    var_name="Metric",
    value_name="Error"
)

palette = {"cosine_error": "Blues_d", "nrmse_max": "Oranges_d"}

def plot_kpm(kpm, fname_stub):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, metric in zip(axes, ["cosine_error", "nrmse_max"]):
        sns.barplot(
            data=melted[(melted["KPM"] == kpm) & (melted["Metric"] == metric)],
            x="variant", y="Error",
            palette=palette[metric], width=0.6, ax=ax)
        ax.set_title(f"{kpm} — {metric.replace('_', ' ').title()}")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel("Variant")

    plt.suptitle(f"{kpm} — Cosine Error vs NRMSE$_{{max}}$", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = f"{OUT_DIR}/{fname_stub}.png"
    plt.savefig(out_png)
    plt.close()
    print(f"🖼️  Plot saved → {out_png}")

# create plots for the two main KPIs
plot_kpm("Avg_Delay_ms",    "avg_delay_cosine_nrmse")
plot_kpm("Throughput_Mbps", "throughput_cosine_nrmse")
