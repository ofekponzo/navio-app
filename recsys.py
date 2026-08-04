"""
NavIO — recsys.py
Retrieval, feature engineering, and classification/regression pipeline.

Ported from Part 3 (navio_part3_final.py). Two Space-specific changes from
the Colab version:

1. Loads the dataset and the mpnet embedding model directly from the
   Hugging Face Hub at import time (course constraints 1 & 2) instead of
   Part 3's Colab train/test split, which existed only for offline
   evaluation. In production there's nothing to hold out — the full 11,000
   rows are the "known patient" pool.
2. Adds `analyze_new_session()`, a thin wrapper around the untouched
   `process_new_session()` that accepts a duration in minutes (what the
   dashboard actually collects) instead of two timestamps, deriving
   start/end from "now" purely for display. process_new_session() itself is
   byte-for-byte the validated Part 3 function.

Everything else — the empirically-tuned 56-term crisis lexicon, the hybrid
feature construction, the deterministic decision tables, the dedicated
crisis classifier + 0.2 probability threshold — is unchanged from the
pipeline validated in Part 3 (Risk_Level Acc 0.621, Dynamic Acc 0.516,
crisis recall 0.753 @ 5.8% false-alarm rate, overall CPT accuracy 0.917).

Everything in this module loads ONCE at import time. app.py imports this
module exactly once at startup; per-request calls only hit
`analyze_new_session()` / `retrieval_index.get_patient_history()`, never
the loading code above them.
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from transformers import pipeline as hf_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

HF_DATASET_REPO_ID = "ofekponzo/navio-synthetic-dataset"
MPNET_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
K_NEIGHBORS = 5
CRISIS_THRESHOLD = 0.2  # tuned empirically in Part 3 — see module docstring
HF_TOKEN = os.environ.get("HF_TOKEN")  # only required if the dataset/Space is private


# --- Step 1: load the dataset directly from the HF dataset repo (constraint 1) --
print("[recsys] Loading NavIO dataset from the Hugging Face Hub...")
df = load_dataset(HF_DATASET_REPO_ID, split="train", token=HF_TOKEN).to_pandas()
assert len(df) == 11000 and df["Patient_ID"].nunique() == 2200, (
    f"[recsys] Unexpected dataset shape: {len(df)} rows, "
    f"{df['Patient_ID'].nunique()} patients — expected 11,000 / 2,200."
)
print(f"[recsys] Loaded {len(df):,} rows / {df['Patient_ID'].nunique():,} patients.")


# --- Step 2: load the winning embedding model directly from HF (constraint 2) --
print("[recsys] Loading all-mpnet-base-v2 embedding model...")
embedder = SentenceTransformer(MPNET_MODEL_ID)


# --- Step 3: load the precomputed dataset embeddings (uploaded to this repo) --
_embeddings_path = os.path.join(ARTIFACT_DIR, "dataset_embeddings.npy")
dataset_embeddings = np.load(_embeddings_path)
assert dataset_embeddings.shape[0] == len(df), (
    f"[recsys] Embeddings ({dataset_embeddings.shape[0]} rows) don't match the "
    f"loaded dataset ({len(df)} rows). The dataset on the Hub may have changed "
    f"since navio_part5_export_artifacts.py was last run — re-run the export "
    f"script to regenerate dataset_embeddings.npy."
)
print(f"[recsys] Loaded precomputed embeddings: {dataset_embeddings.shape}")


# --- Step 4: load the trained sklearn artifacts ------------------------------
risk_regressor = joblib.load(os.path.join(ARTIFACT_DIR, "risk_regressor.joblib"))
dynamic_classifier = joblib.load(os.path.join(ARTIFACT_DIR, "dynamic_classifier.joblib"))
crisis_classifier = joblib.load(os.path.join(ARTIFACT_DIR, "crisis_classifier.joblib"))
engineered_scaler = joblib.load(os.path.join(ARTIFACT_DIR, "engineered_scaler.joblib"))
print("[recsys] Loaded trained artifacts: risk_regressor, dynamic_classifier, crisis_classifier, engineered_scaler.")


# --- Sentiment pipeline (free Spaces are CPU-only) ---------------------------
_sentiment_pipeline = hf_pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1, truncation=True, max_length=512,
)


# --- Engineered features (verbatim from Part 3 — empirically-tuned lexicon) --
TIMESTAMP_PATTERN = re.compile(r"\[\d{1,2}:\d{2}\]")
SPEAKER_LINE_PATTERN = re.compile(r"(?m)^\s*([A-Za-z][A-Za-z .]{0,25}):\s*(.*)$")

# Original 22 hand-picked terms + 34 data-driven terms mined from frequency-
# ratio analysis in Part 3 (words >=5x more frequent in crisis transcripts,
# >=15 occurrences, excluding overly generic tokens). Do not edit this list
# without retraining crisis_classifier — the model was fit on features
# computed from exactly this lexicon.
CRISIS_KEYWORDS = [
    "anxiety", "anxious", "panic", "helpless", "helplessness", "hopeless",
    "hopelessness", "overwhelmed", "overwhelming", "suicidal", "suicide",
    "self-harm", "crisis", "unsafe", "worthless", "danger", "scared",
    "afraid", "trapped", "can't cope", "give up", "giving up",
    "additional", "between", "breakdown", "breaks", "complete", "concerned",
    "crying", "current", "desperate", "distressed", "emergency", "ensure",
    "escalating", "harm", "harming", "hotline", "immediate", "intervention",
    "lose", "panicked", "provide", "resigned", "resources", "risk", "safe",
    "safety", "severe", "sniffling", "sobbing", "sobs", "stabilize", "state",
    "tearful", "tears",
]

FEATURE_NAMES = [
    "session_duration_min", "word_count", "crisis_keyword_count", "sentiment_score",
    "patient_turn_count", "therapist_turn_count", "avg_patient_utterance_len",
    "avg_therapist_utterance_len", "turn_length_ratio",
]


def compute_session_duration_minutes(start_time, end_time):
    start, end = pd.to_datetime(start_time), pd.to_datetime(end_time)
    return max(1, round((end - start).total_seconds() / 60))


def extract_turns(transcript):
    clean = TIMESTAMP_PATTERN.sub(" ", transcript) if isinstance(transcript, str) else ""
    turns = []
    for match in SPEAKER_LINE_PATTERN.finditer(clean):
        speaker_name, utterance = match.group(1).strip(), match.group(2).strip()
        role = "therapist" if "dr" in speaker_name.lower() else "patient"
        if utterance:
            turns.append((role, utterance))
    return turns


# --- Intra-session timeline parsing (Graph 2 — UI-facing, not part of the ---
# --- trained feature pipeline above, which deliberately strips timestamps) --
# extract_turns() above throws the [MM:SS] markers away on purpose, because
# the trained models were fit on timestamp-free text. Graph 2 needs the
# opposite: it plots turns BY their timestamp, so this is a parallel parser
# that keeps them. Reuses the same empirically-tuned CRISIS_KEYWORDS lexicon
# and the already-loaded sentiment pipeline so "risk spikes" on the timeline
# stay consistent with what the model itself considers crisis language,
# rather than inventing a separate ad hoc vocabulary for the chart.
INTRASESSION_TURN_PATTERN = re.compile(
    r"^\s*\[(\d{1,2}):(\d{2})\]\s*([A-Za-z][A-Za-z .]{0,25}):\s*(.*)$", re.MULTILINE
)


def parse_intrasession_timeline(transcript):
    """Returns a list of per-turn dicts, ordered by time, for Graph 2:
      time_min      — minutes into the session (float, e.g. 12.5)
      role          — "therapist" | "patient"
      speaker       — raw speaker label as written in the transcript
      utterance     — the turn's text
      crisis_hits   — CRISIS_KEYWORDS terms found in this turn (possibly [])
      intensity     — a 0-100 heuristic "arousal" score for this turn, built
                       from sentiment polarity plus a crisis-keyword bonus.
                       This is a presentation-layer heuristic for visualizing
                       within-session shifts — NOT a trained model output,
                       and it plays no role in Risk_Score / Risk_Level /
                       Crisis_Flag, which come entirely from the models above.
    Turns without a leading [MM:SS] marker are skipped (nothing to place on
    the timeline's x-axis for them).
    """
    text = transcript if isinstance(transcript, str) else ""
    turns = []
    for match in INTRASESSION_TURN_PATTERN.finditer(text):
        minute_str, second_str, speaker, utterance = match.groups()
        utterance = utterance.strip()
        if not utterance:
            continue
        time_min = int(minute_str) + int(second_str) / 60.0
        role = "therapist" if "dr" in speaker.lower() else "patient"
        lower_utterance = utterance.lower()
        crisis_hits = [kw for kw in CRISIS_KEYWORDS if kw in lower_utterance]

        try:
            sentiment_result = _sentiment_pipeline(utterance[:512])[0]
        except Exception:
            sentiment_result = {"label": "NEUTRAL", "score": 0.5}
        if sentiment_result["label"] == "NEGATIVE":
            base_intensity = sentiment_result["score"] * 65.0
        else:
            base_intensity = (1 - sentiment_result["score"]) * 30.0
        crisis_bonus = min(len(crisis_hits) * 14.0, 40.0)
        intensity = float(np.clip(base_intensity + crisis_bonus, 0, 100))

        turns.append({
            "time_min": round(time_min, 2),
            "role": role,
            "speaker": speaker.strip(),
            "utterance": utterance,
            "crisis_hits": crisis_hits,
            "intensity": round(intensity, 1),
        })
    turns.sort(key=lambda t: t["time_min"])
    return turns


def engineered_features(transcript, session_duration_minutes):
    text = transcript if isinstance(transcript, str) else ""
    words = text.split()
    word_count = len(words)
    lower_text = text.lower()
    crisis_count = sum(lower_text.count(kw) for kw in CRISIS_KEYWORDS)

    sentiment_result = _sentiment_pipeline(text[:2000])[0] if text.strip() else {"label": "NEUTRAL", "score": 0.5}
    sentiment_score = sentiment_result["score"] if sentiment_result["label"] == "POSITIVE" else -sentiment_result["score"]

    turns = extract_turns(text)
    patient_turns = [u for role, u in turns if role == "patient"]
    therapist_turns = [u for role, u in turns if role == "therapist"]
    avg_patient_len = np.mean([len(u.split()) for u in patient_turns]) if patient_turns else 0.0
    avg_therapist_len = np.mean([len(u.split()) for u in therapist_turns]) if therapist_turns else 0.0
    turn_ratio = avg_patient_len / avg_therapist_len if avg_therapist_len > 0 else 0.0

    return np.array([
        session_duration_minutes, word_count, crisis_count, sentiment_score,
        len(patient_turns), len(therapist_turns), avg_patient_len, avg_therapist_len, turn_ratio,
    ], dtype=np.float32)


# --- Deterministic decision tables (verbatim from Part 1 / Part 3) ----------
def bucket_risk_level(score):
    """Single source of truth for Risk_Score -> Risk_Level, identical to Part 1."""
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Medium"
    return "High"


def derive_cpt_code(duration_minutes, crisis_flag):
    """Crisis_Flag comes from the dedicated classifier, never the regression."""
    if crisis_flag:
        return "90839", 60
    duration_bucket = min([30, 45, 60], key=lambda d: abs(d - duration_minutes))
    return {30: "90832", 45: "90834", 60: "90837"}[duration_bucket], duration_bucket


DYNAMIC_RANK = {"Open": 1, "Dependent": 0, "Defensive": -1}


def bucket_direction(delta, improve_threshold=-2, worsen_threshold=2):
    if delta <= improve_threshold:
        return "Improving"
    elif delta >= worsen_threshold:
        return "Worsening"
    return "Stable"


def derive_progress_combined(current_risk_score, previous_risk_score, current_dynamic, previous_dynamic):
    if previous_risk_score is None:
        return "Baseline"
    risk_direction = bucket_direction(current_risk_score - previous_risk_score)
    dynamic_shift = DYNAMIC_RANK.get(current_dynamic, 0) - DYNAMIC_RANK.get(previous_dynamic, 0)
    dynamic_direction = "Improving" if dynamic_shift > 0 else ("Worsening" if dynamic_shift < 0 else "Stable")

    if risk_direction == "Improving" and dynamic_direction == "Improving":
        return "Breakthrough"
    if risk_direction == "Worsening" and dynamic_direction == "Worsening":
        return "Deteriorating"
    if {risk_direction, dynamic_direction} == {"Improving", "Worsening"}:
        return "Mixed Signal — Clinical Review Recommended"
    if "Improving" in (risk_direction, dynamic_direction):
        return "Gradual Improvement"
    if "Worsening" in (risk_direction, dynamic_direction):
        return "Deteriorating"
    return "Stagnant"


# --- Retrieval / RAG index (verbatim from Part 3) ---------------------------
class NavioRetrievalIndex:
    def __init__(self, embeddings, metadata_df):
        self.embeddings = normalize(embeddings)
        self.metadata = metadata_df.reset_index(drop=True)

    def query_similar(self, query_embedding, k=K_NEIGHBORS, exclude_patient_id=None):
        query_norm = normalize(query_embedding.reshape(1, -1))
        similarities = (query_norm @ self.embeddings.T).flatten()
        order = np.argsort(-similarities)
        results = []
        for idx in order:
            row = self.metadata.iloc[idx]
            if exclude_patient_id is not None and row["Patient_ID"] == exclude_patient_id:
                continue
            results.append({**row.to_dict(), "similarity": float(similarities[idx])})
            if len(results) >= k:
                break
        return results

    def get_patient_history(self, patient_id):
        history = self.metadata[self.metadata["Patient_ID"] == patient_id].sort_values("Session_Number")
        return history.to_dict("records")


retrieval_index = NavioRetrievalIndex(dataset_embeddings, df)
print(f"[recsys] Retrieval index built: {len(retrieval_index.metadata):,} sessions indexed.")


# --- End-to-end orchestrator (verbatim from Part 3) -------------------------
def process_new_session(transcript, start_time, end_time, patient_id):
    duration_minutes = compute_session_duration_minutes(start_time, end_time)

    history = retrieval_index.get_patient_history(patient_id)
    is_new_patient = len(history) == 0
    previous_session = history[-1] if history else None

    embedding = embedder.encode([transcript], convert_to_numpy=True, normalize_embeddings=True)[0]
    features = engineered_features(transcript, duration_minutes)
    features_scaled = engineered_scaler.transform(features.reshape(1, -1))[0]
    hybrid_vector = np.concatenate([embedding, features_scaled]).reshape(1, -1)

    predicted_score = float(np.clip(risk_regressor.predict(hybrid_vector)[0], 0, 10))
    predicted_dyn = dynamic_classifier.predict(hybrid_vector)[0]

    # Crisis: dedicated classifier + tuned threshold, NOT the regression.
    crisis_probability = float(crisis_classifier.predict_proba(hybrid_vector)[0, 1])
    crisis_flag = bool(crisis_probability >= CRISIS_THRESHOLD)

    risk_level = bucket_risk_level(predicted_score)
    cpt_code, billed_duration = derive_cpt_code(duration_minutes, crisis_flag)

    if is_new_patient:
        progress = "Baseline"
    else:
        progress = derive_progress_combined(
            predicted_score, previous_session["Risk_Score"], predicted_dyn, previous_session["Dynamic"],
        )

    similar_cases = retrieval_index.query_similar(embedding, k=K_NEIGHBORS, exclude_patient_id=patient_id)

    return {
        "Patient_ID": patient_id,
        "session_context": "intake" if is_new_patient else "follow_up",
        "Session_Duration_minutes": duration_minutes,
        "Predicted_Risk_Score": round(predicted_score, 2),
        "Risk_Level": risk_level,
        "Crisis_Flag": crisis_flag,
        "Crisis_Probability": round(crisis_probability, 3),
        "Predicted_Dynamic": predicted_dyn,
        "Progress": progress,
        "Target_CPT_Code": cpt_code,
        "Billed_Duration_minutes": billed_duration,
        "Similar_Cases": similar_cases,
        "Previous_Session_Summary": previous_session,
    }


def analyze_new_session(transcript, patient_id, duration_minutes):
    """UI-facing wrapper: the dashboard collects one duration value, not two
    timestamps — nobody wants to pick two datetimes for a session that just
    happened. Derives start/end from 'now' purely so the metric cards have
    something to display; the ML pipeline itself only ever consumed the
    duration, so this changes nothing about prediction behavior."""
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=max(1, int(duration_minutes)))
    result = process_new_session(transcript, start_time, end_time, patient_id)
    result["Session_Start"] = start_time
    result["Session_End"] = end_time
    return result


def get_all_patient_ids():
    return sorted(df["Patient_ID"].unique().tolist())
