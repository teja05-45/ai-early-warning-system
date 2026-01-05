import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("models/risk_model.joblib")
df = pd.read_csv("data/processed/ml_features.csv")

X = df.drop("risk_label", axis=1)

# Create SHAP explainer
explainer = shap.Explainer(model, X)

# Calculate SHAP values
shap_values = explainer(X)

# -----------------------------
# Global Feature Importance
# -----------------------------
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png")
plt.close()

print("✅ SHAP explainability generated")
print("📊 Saved: reports/shap_summary.png")
