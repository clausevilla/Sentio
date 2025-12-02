# Author: Karl Byland, Claudia Sevilla

import joblib

from apps.ml_admin.models import ModelVersion
from apps.predictions.models import PredictionResult, TextSubmission
import re

NEGATIVE_WORDS = ['sad', 'angry', 'bad', 'mad', 'upset', 'frustrated', 'terrible', 'hopeless', 'pessimistic', 'worried', 'anxious', 'nervous', 'fear', 'awful', 'panic', 'stress']

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

    # Calculate metrics
    anxiety_level, negativity_level, emotional_intensity, word_count, char_count = calculate_metrics(user_text)

    if user:
        save_prediction_to_database(
            user, user_text, prediction, confidence, model_version
        )

    confidence_percentage = round(confidence * 100)

    return prediction, confidence_percentage, anxiety_level, negativity_level, emotional_intensity, word_count, char_count

def calculate_metrics(text: str):
    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)
    char_count = len(text)

    # Negativity level based on key words
    negativity_count = sum(words.count(w) for w in NEGATIVE_WORDS)
    negativity_level = min(int((negativity_count / max(word_count, 1)) * 100), 100)

    # Extract general text features (punctuation, use of caps, word density)
    punctuation_count = len(re.findall(r'[!?]', text))
    capital_count = sum(1 for c in text if c.isupper())
    word_density = word_count / max(char_count, 1)  # words per character

    anxiety_score = word_density * 50 + punctuation_count * 30 + capital_count * 0.5 # Anxiety: more punctuation and word density
    emotional_score = word_density * 20 + punctuation_count * 10 + capital_count * 5 # Emotional: higher use of capitals

    # Calculate anxiety level and emotional intensity based on text features
    anxiety_level = min(int(anxiety_score), 100) # cap to percentage
    emotional_intensity = min(int(emotional_score), 100)

    return anxiety_level, negativity_level, emotional_intensity, word_count, char_count


def save_prediction_to_database(user, user_text, prediction, confidence, model_version):
    if user:
        submission = TextSubmission.objects.create(user=user, text_content=user_text)
        # Save to database
        PredictionResult.objects.create(
            submission=submission,
            model_version=model_version,
            mental_state=prediction,
            confidence=confidence,
            recommendations='Follow recommended steps',  # placeholder text
        )
