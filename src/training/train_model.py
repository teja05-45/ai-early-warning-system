import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Load processed features
df = pd.read_csv("data/processed/ml_features.csv")

# Split features & target
X = df.drop("risk_label", axis=1)
y = df["risk_label"]

# Train-test split (time-agnostic for now)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model (industry favorite)
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"✅ Model trained successfully | ROC-AUC: {roc_auc:.4f}")

# Save model
joblib.dump(model, "models/risk_model.joblib")
print("📦 Model saved to models/risk_model.joblib")
