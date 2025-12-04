# Author: Karl Byland

import joblib

from apps.ml_admin.models import ModelVersion
from apps.predictions.models import PredictionResult, TextSubmission

# Loads the model
MODEL_PATH = 'ml_pipeline/toy_models/LRmodel.pkl'  # Path to model used to predict
MODEL = joblib.load(MODEL_PATH)


def analyze_text(analyzed_text):
    prediction = MODEL.predict([analyzed_text])

    label = prediction[0]
    proba = MODEL.predict_proba([analyzed_text])[0]
    confidence = max(proba)
    model_version = (
        ModelVersion.objects.first()
    )  # Placeholder, just uses the first model in the database

    return (label, confidence, model_version)


def get_prediction_result(user, user_text):
    prediction, confidence, model_version = analyze_text(user_text)

    if user:
        save_prediction_to_database(
            user, user_text, prediction, confidence, model_version
        )

    confidence_percentage = round(confidence * 100)

    return prediction, confidence_percentage


def save_prediction_to_database(user, user_text, prediction, confidence, model_version):
    template_texts = [
        (
            'I feel so empty inside. Nothing brings me joy anymore.'
            " I wake up each day wondering what's the point."
            " I used to love painting but now I can't even pick up a brush."
            ' My friends invite me out but I just make excuses.'
            " I'm tired all the time but can't sleep properly. Everything feels gray and meaningless."
        ),
        (
            'I have so much on my plate right now.'
            ' Work deadlines are piling up, bills need to be paid, and I barely have time to breathe.'
            " I feel overwhelmed and like I'm drowning."
            ' My body feels tense all the time and I get headaches every day.'
            " I snap at people I care about because I'm so on edge."
        ),
    ]
    if user_text not in template_texts:
        submission = TextSubmission.objects.create(user=user, text_content=user_text)
        # Save to database
        PredictionResult.objects.create(
            submission=submission,
            model_version=model_version,
            mental_state=prediction,
            confidence=confidence,
            recommendations='Follow recommended steps',  # placeholder text
        )
