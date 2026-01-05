import pandas as pd
import json

DRIFT_THRESHOLD = 0.2  # 20%

# Load baseline
with open("models/training_feature_baseline.json") as f:
    baseline = json.load(f)

# Load current data
df = pd.read_csv("data/processed/ml_features.csv")
current_features = df.drop(columns=["risk_label"]).select_dtypes(include=["int64", "float64"])

current_means = current_features.mean()

print("\n📊 Drift Report:")
for feature, train_mean in baseline.items():
    curr_mean = current_means.get(feature, None)

    if curr_mean is not None and train_mean != 0:
        drift = abs(curr_mean - train_mean) / abs(train_mean)

        if drift > DRIFT_THRESHOLD:
            print(f"⚠️ Drift detected in {feature}: {drift:.2%}")
        else:
            print(f"✅ {feature}: stable")
