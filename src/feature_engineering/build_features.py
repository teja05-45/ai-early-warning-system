import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv("data/raw/operational_risk_data.csv", parse_dates=["date"])

# Sort for time-based operations
df = df.sort_values(["ticket_id", "date"])

# -----------------------------
# 1. Encode Priority (Ordinal)
# -----------------------------
priority_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4
}
df["priority_encoded"] = df["priority"].map(priority_map)

# -----------------------------
# 2. Rolling Features (7 days)
# -----------------------------
df["avg_engineer_load_7d"] = (
    df.groupby("ticket_id")["assigned_engineer_load"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

df["escalations_7d"] = (
    df.groupby("ticket_id")["customer_escalations"]
    .transform(lambda x: x.rolling(7, min_periods=1).sum())
)

# -----------------------------
# 3. Trend Features
# -----------------------------
df["engineer_load_trend"] = (
    df.groupby("ticket_id")["assigned_engineer_load"]
    .diff()
    .fillna(0)
)

df["resolution_delay_trend_7d"] = (
    df.groupby("ticket_id")["resolution_delay_trend"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# -----------------------------
# 4. Lag Features (Early Signals)
# -----------------------------
df["risk_label_lag_1"] = (
    df.groupby("ticket_id")["risk_label"]
    .shift(1)
    .fillna(0)
)
# -----------------------------
# 6. Risk Trend Feature (7-day)
# -----------------------------
df["risk_score_proxy"] = (
    0.4 * (df["priority_encoded"] >= 3).astype(int) +
    0.3 * (df["assigned_engineer_load"] > 15).astype(int) +
    0.2 * (df["resolution_delay_trend_7d"] > 0.3).astype(int) +
    0.1 * df["customer_escalations"]
)

df["risk_trend_7d"] = (
    df.groupby("ticket_id")["risk_score_proxy"]
    .diff(7)
    .fillna(0)
)


# -----------------------------
# 5. Drop Non-ML Columns
# -----------------------------
ml_df = df.drop(columns=["priority", "date", "ticket_id"])

# Save processed data
ml_df.to_csv("data/processed/ml_features.csv", index=False)

print("✅ Feature engineering completed")
print(ml_df.head())
