from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / 'model' / 'sentiment_model.joblib'

if not MODEL_PATH.exists():
    print(f'Model not found at {MODEL_PATH}. Please run train.py first.')
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)

samples = [
    "I absolutely loved the movie, it was fantastic!",
    "This was the worst experience, I hated it.",
    "It was okay, not great but not terrible.",
]

print('Running smoke test predictions:')
for s in samples:
    pred = model.predict([s])[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = max(model.predict_proba([s])[0])
    label = 'Positive' if int(pred) == 1 else 'Negative'
    conf = f"{proba:.2%}" if proba is not None else 'N/A'
    print(f'Input: {s}\n -> Prediction: {label} (confidence: {conf})\n')
