import joblib

from apps.ml_admin.models import ModelVersion

# Loads the model
MODEL_PATH = 'ml_pipeline/toy_models/LRmodel.pkl'
MODEL = joblib.load(MODEL_PATH)


def analyze_text(analyzed_text):
    prediction = MODEL.predict([analyzed_text])

    label = prediction[0]
    proba = MODEL.predict_proba([analyzed_text])[0]
    confidence = max(proba)
    model_version = ModelVersion.objects.first()

    return (label, confidence, model_version)
