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

    # Generate recommendations
    recommendations = get_recommendations(prediction, confidence, anxiety_level)

    if user:
        save_prediction_to_database(
            user, user_text, prediction, confidence, model_version, recommendations
        )

    confidence_percentage = round(confidence * 100)

    return prediction, confidence_percentage, recommendations, anxiety_level, negativity_level, emotional_intensity, word_count, char_count

def get_recommendations (prediction, confidence, anxiety_level):
    recommendations = []

    if confidence > 70:             # high confidence results
        if prediction == 'normal':
            recommendations.append("Congratulations! You seem to present an undoubtedly healthy mental state.")
        elif prediction == 'stress':
            recommendations.append("You seem to present strong indicators of stress, try to limit caffeine and consider mindfulness meditation for 10 minutes daily. Delegate tasks where possible and practice setting boundaries. Reduce multitasking and practice single-tasking focus.")
        elif prediction == 'suicidal':
            recommendations.append("You are not alone. If you are having thoughts of taking your life or if you know anyone who does, contact Mind Suicide Line dialing 90101 (open 24/7) or contact them on their online chat. For immediate crisis, call 988 (Suicide & Crisis Lifeline)")
        elif prediction == 'depression':
            recommendations.append("You seem to present strong indicators of depression, consider speaking with a mental health professional this week. Establish a consistent sleep schedule and try behavioral activation: schedule one pleasant activity each day.")

    elif confidence >= 50 and confidence <= 70:     # medium confidence
        if prediction == 'normal':
            recommendations.append("Good job! You seem to have a healthy mental state.")
        elif prediction == 'stress':
            recommendations.append("You seem to present medium indicators of stress, try to limit caffeine and consider mindfulness meditation for 10 minutes daily. Delegate tasks where possible and practice setting boundaries. Reduce multitasking and practice single-tasking focus.")
            if anxiety_level >= 50:
                recommendations.append("Additionally, your text presents symptoms of anxiety, consider practicing daily grounding techniques (5-4-3-2-1 method). Remember no problem lasts forever, try to see things from an outside perspective and put your mental health before anything else.")
        elif prediction == 'suicidal':
            recommendations.append("The analysis suggests you may have thoughts of taking your own life. If that is true or you know anyone who does contact Mind Suicide Line dialing 90101 (open 24/7) or contact them on their online chat. For immediate crisis, call 988 (Suicide & Crisis Lifeline)")
        elif prediction == 'depression':
            recommendations.append("You seem to present mild indicators of depression, consider speaking with a mental health professional if you feel your situation requires professional help. Establish a consistent sleep schedule and try behavioral activation: schedule one pleasant activity each day.")

    else:                               # low confidence
        recommendations.append("Model confidence is low. Consider these general wellness suggestions:")

        if prediction == 'suicidal':
            recommendations.append("The analysis suggests you may have thoughts of taking your own life. If that is true or you know anyone who does contact Mind Suicide Line dialing 90101 (open 24/7) or contact them on their online chat. Even with low confidence, this is important: For immediate crisis, call 988 (Suicide & Crisis Lifeline).")
        else:
            recommendations.append("Maintain regular sleep, nutrition, and physical activity. Check in with yourself regularly about emotional wellbeing, and do not hesitate to reach out for help.")

    if anxiety_level >= 50:
        recommendations.append("Additionally, your text presents a high anxiety level, consider practicing daily grounding techniques (5-4-3-2-1 method). Try to see things from a positive perspective and prioritize your mental health.")

    return recommendations

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


def save_prediction_to_database(user, user_text, prediction, confidence, recommendations, model_version):
    if user:
        submission = TextSubmission.objects.create(user=user, text_content=user_text)
        # Save to database
        PredictionResult.objects.create(
            submission=submission,
            model_version=model_version,
            mental_state=prediction,
            confidence=confidence,
            recommendations=recommendations,
        )
