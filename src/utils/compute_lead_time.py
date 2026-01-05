import pandas as pd
import joblib

# Load data and model
df = pd.read_csv("data/processed/ml_features.csv")
model = joblib.load("models/risk_model.joblib")

# Predict risk scores
X = df.drop("risk_label", axis=1)
df["risk_score"] = model.predict_proba(X)[:, 1]

# Thresholds
WARNING_THRESHOLD = 0.7

# Track lead time per ticket
lead_times = []

for ticket_id in df.index.unique():
    ticket_df = df.loc[[ticket_id]]

    # First warning date
    warnings = ticket_df[ticket_df["risk_score"] > WARNING_THRESHOLD]
    failures = ticket_df[ticket_df["risk_label"] == 1]

    if len(warnings) > 0 and len(failures) > 0:
        first_warning_idx = warnings.index.min()
        failure_idx = failures.index.min()

        lead_time = failure_idx - first_warning_idx
        lead_times.append(lead_time)

# Summary
if lead_times:
    avg_lead_time = sum(lead_times) / len(lead_times)
    print(f"⏱️ Average Early-Warning Lead Time: {avg_lead_time:.2f} days")
else:
    print("⚠️ No valid lead-time cases found")
