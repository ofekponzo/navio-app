"""
NavIO — scripts/make_mock_artifacts.py
Development-only stand-ins for the trained artifacts in artifacts/.

The real artifacts (risk_regressor, dynamic_classifier, crisis_classifier,
engineered_scaler, dataset_embeddings.npy) are stored in Git LFS and are not
available from the GitHub remote. This script fits tiny placeholder models on
random hybrid vectors so recsys.py can import and the dashboard can run.
Predictions are NOT meaningful — replace with the real artifacts to restore
the validated Part 3 behaviour.

Usage: NAVIO_ARTIFACT_DIR=/some/dir python scripts/make_mock_artifacts.py
"""

import os
import sys

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.environ.get("NAVIO_ARTIFACT_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
REQUIRED = [
    "dataset_embeddings.npy", "risk_regressor.joblib", "dynamic_classifier.joblib",
    "crisis_classifier.joblib", "engineered_scaler.joblib",
]
EMBED_DIM = 768   # all-mpnet-base-v2
N_ENGINEERED = 9  # recsys.FEATURE_NAMES

if all(os.path.exists(os.path.join(OUT_DIR, f)) for f in REQUIRED):
    print(f"[mock-artifacts] All artifacts already present in {OUT_DIR}; nothing to do.")
    sys.exit(0)

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(44)

print("[mock-artifacts] Loading dataset to size embeddings / pick label sets...")
df = load_dataset("ofekponzo/navio-synthetic-dataset", split="train", token=os.environ.get("HF_TOKEN")).to_pandas()
n_rows = len(df)

embeddings = rng.standard_normal((n_rows, EMBED_DIM)).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
np.save(os.path.join(OUT_DIR, "dataset_embeddings.npy"), embeddings)

n_fit = 400
raw_features = rng.standard_normal((n_fit, N_ENGINEERED)) * 10
scaler = StandardScaler().fit(raw_features)
X = np.concatenate([embeddings[:n_fit], scaler.transform(raw_features)], axis=1)

risk_regressor = Ridge().fit(X, df["Risk_Score"].values[:n_fit].astype(float))
dynamic_classifier = LogisticRegression(max_iter=200).fit(X, df["Dynamic"].values[:n_fit])
crisis_classifier = LogisticRegression(max_iter=200).fit(X, df["Crisis_Flag"].values[:n_fit].astype(bool))

joblib.dump(risk_regressor, os.path.join(OUT_DIR, "risk_regressor.joblib"))
joblib.dump(dynamic_classifier, os.path.join(OUT_DIR, "dynamic_classifier.joblib"))
joblib.dump(crisis_classifier, os.path.join(OUT_DIR, "crisis_classifier.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "engineered_scaler.joblib"))
print(f"[mock-artifacts] Wrote MOCK artifacts to {OUT_DIR} ({n_rows} x {EMBED_DIM} embeddings).")
