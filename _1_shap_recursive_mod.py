import os
import glob
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# === Feature classification ===
control_features = ['ant_tilt_deg', 'CIO', 'TxPower', 'PRB_num', 'Scheduling']
non_control_features = []
#non_control_features = ['Users', 'DL_Buffer', 'Avg_SNR_dB']
all_base_features = control_features + non_control_features

# === Target configuration ===
kpm_targets = ['Throughput_Mbps', 'Avg_Delay_ms', 'user_throughput']
recursive_targets = non_control_features

output_dir = "shap_outputs"
os.makedirs(output_dir, exist_ok=True)

file_paths = sorted(glob.glob("dataset/ORAN_log*.csv"))

for fp in file_paths:
    print(f"\n📁 Processing: {fp}")
    df = pd.read_csv(fp)

    # Ensure unique columns (safety)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required_columns = all_base_features + kpm_targets
    if not all(col in df.columns for col in required_columns):
        print("❌ Missing required columns — skipping")
        continue

    df = df[required_columns].dropna()
    if len(df) < 2:
        print("❌ Skipping (not enough data)")
        continue

    base = os.path.splitext(os.path.basename(fp))[0]

    # Filter out non-numeric features before computing correlation
    numeric_feats = [f for f in all_base_features if df[f].dtype in [np.float64, np.int64, np.float32, np.int32]]

    if numeric_feats:
        corr_matrix = df[numeric_feats].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Map: {base}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base}_corr_map.png")
        plt.close()

        print("📊 Correlation heatmap saved.")
    else:
        print("⚠️ No numeric features available for correlation heatmap.")


    for target in kpm_targets + recursive_targets:
        print(f"\n🔎 SHAP for target: {target}")
        y = df[target]

        # Determine SHAP input features
        if target in recursive_targets:
            target_features = control_features
        else:
            target_features = control_features + non_control_features

        X = df[target_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), [f for f in target_features if f != "Scheduling"]),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ["Scheduling"]),
            ],
            verbose_feature_names_out=False          # <── this kills "num__"/"cat__"
        )
        X_encoded = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        print(f"✅ RMSE ({target}): {rmse:.4f}")

        # SHAP explanation
        background = shap.utils.sample(X_train, 100, random_state=42)
        explainer = shap.Explainer(model, background)
        shap_values = explainer(X_test).values

        feature_names = preprocessor.get_feature_names_out()

        # === Aggregate Scheduling importance ===
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        norm_mean_abs = mean_abs / np.sum(mean_abs)  # Normalize so they sum to 1
        feat = list(feature_names)
        vals = list(mean_abs)
        #vals = list(norm_mean_abs)
        sched_idx = [i for i, n in enumerate(feat) if n.startswith('cat__Scheduling_')]
        if sched_idx:
            sched_imp = sum(vals[i] for i in sched_idx)
            for i in sorted(sched_idx, reverse=True):
                del feat[i]; del vals[i]
            feat.append('Scheduling')
            vals.append(sched_imp)

        # --- Save outputs ---
        np.save(f"{output_dir}/{base}_{target}_features.npy", feature_names)
        np.save(f"{output_dir}/{base}_{target}_shap.npy", shap_values)
        np.save(f"{output_dir}/{base}_{target}_mean_abs.npy", np.array([feat, vals], dtype=object))

        if target in recursive_targets:
            np.save(f"{output_dir}/{base}_{target}_Xtest.npy", X_test)
        else:
            np.save(f"{output_dir}/{base}_Xtest.npy", X_test)

        np.save(f"{output_dir}/{base}_{target}_ytest.npy", y_test.to_numpy())
        # Print top features for quick check
        print("Top |SHAP| features:")
        for f, v in sorted(zip(feat, vals), key=lambda x: -x[1])[:8]:
            print(f"  • {f:18s}: {v:.4f}")