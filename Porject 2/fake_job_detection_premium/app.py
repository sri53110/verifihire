import os
import re
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "job_risk_model.joblib"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request limit

_model = None

def load_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model

def warning_signals(text: str):
    """Transparent heuristic indicators; these are clues, not proof of fraud."""
    checks = [
        ("Upfront payment request", r"\b(registration fee|pay (?:a |the )?(?:fee|deposit)|upfront payment|processing fee|security deposit)\b"),
        ("Unusually high or guaranteed earnings", r"\b(guaranteed income|earn.{0,20}(?:per day|per week)|easy money|no experience.{0,20}high pay)\b"),
        ("Pressure or urgency", r"\b(apply immediately|limited slots|urgent hiring|act now|today only)\b"),
        ("Unverifiable contact method", r"\b(telegram|whatsapp only|message me privately)\b"),
        ("Requests sensitive personal information", r"\b(bank password|otp|one.time password|send your aadhaar|social security number)\b"),
        ("No interview or selection process", r"\b(no interview|required interview waived|hired instantly|instant joining)\b"),
    ]
    found = []
    for label, pattern in checks:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(label)
    return found

def combine_text(payload):
    fields = ["title", "company_profile", "description", "requirements", "benefits", "employment_type", "location"]
    return " ".join(str(payload.get(k, "") or "").strip() for k in fields).strip()

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    model = load_model()
    return jsonify({
        "status": "ok",
        "model_ready": model is not None,
        "message": "Model loaded" if model is not None else "Model not trained yet. Run: python train_model.py"
    })

@app.post("/api/analyze")
def analyze():
    if not request.is_json:
        return jsonify({"error": "Send the job details as JSON."}), 400
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    if not title and not description:
        return jsonify({"error": "Enter a job title or job description."}), 400
    if len(title) > 300 or len(description) > 20000:
        return jsonify({"error": "Title or description is too long."}), 400

    model = load_model()
    if model is None:
        return jsonify({
            "error": "The ML model is not trained yet. Add a labeled dataset and run `python train_model.py`."
        }), 503

    text = combine_text(payload)
    prediction = int(model.predict([text])[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = list(model.classes_)
        if prediction in classes:
            confidence = round(float(probs[classes.index(prediction)]) * 100, 1)

    signals = warning_signals(text)
    # Risk score is the model's probability for the fraudulent class, not a verified scam probability.
    risk_score = None
    if hasattr(model, "predict_proba") and 1 in list(model.classes_):
        probs = model.predict_proba([text])[0]
        risk_score = round(float(probs[list(model.classes_).index(1)]) * 100)

    result = {
        "label": "Potentially fraudulent" if prediction == 1 else "Likely legitimate",
        "prediction": prediction,
        "confidence": confidence,
        "risk_score": risk_score,
        "warning_signals": signals,
        "explanation": (
            "The model found patterns associated with fraudulent postings. Verify the employer independently."
            if prediction == 1 else
            "The model found more patterns associated with legitimate postings. This is not a guarantee of safety."
        ),
        "disclaimer": "Automated screening only. Verify the employer, official domain, and hiring process independently."
    }
    return jsonify(result)

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Request too large. Keep the job description under 20,000 characters."}), 413

if __name__ == "__main__":
    # For local development only. Use a production WSGI server when deployed.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
