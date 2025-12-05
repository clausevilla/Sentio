# Author: Karl Byland, Claudia Sevilla

import joblib

from apps.ml_admin.models import ModelVersion
from apps.predictions.models import PredictionResult, TextSubmission
import re

NEGATIVE_WORDS = [
    # Sadness / Depression
    'sad',
    'depressed',
    'hopeless',
    'empty',
    'numb',
    'lonely',
    'miserable',
    'worthless',
    'useless',
    'helpless',
    'despair',
    'grief',
    'heartbroken',
    'unhappy',
    'down',
    'low',
    'blue',
    'gloomy',
    'melancholy',
    'dejected',
    # Anger / Frustration
    'angry',
    'mad',
    'furious',
    'rage',
    'hate',
    'frustrated',
    'irritated',
    'annoyed',
    'bitter',
    'resentful',
    'hostile',
    'agitated',
    'outraged',
    # Anxiety / Fear
    'anxious',
    'worried',
    'nervous',
    'scared',
    'afraid',
    'fear',
    'panic',
    'terrified',
    'dread',
    'uneasy',
    'tense',
    'restless',
    'overwhelmed',
    'paranoid',
    'insecure',
    'uncertain',
    # Stress / Exhaustion
    'stress',
    'stressed',
    'exhausted',
    'tired',
    'drained',
    'burned',
    'burnout',
    'overwhelmed',
    'pressured',
    'overloaded',
    'fatigued',
    # General negative
    'bad',
    'terrible',
    'awful',
    'horrible',
    'worst',
    'painful',
    'suffering',
    'struggling',
    'failing',
    'broken',
    'damaged',
    'ruined',
    'destroyed',
    'pessimistic',
    'negative',
    'dark',
    'lost',
    'stuck',
    'trapped',
    # Self-critical
    'stupid',
    'dumb',
    'idiot',
    'failure',
    'loser',
    'pathetic',
    'weak',
    'ugly',
    'fat',
    'disgusting',
    'ashamed',
    'embarrassed',
    'guilty',
    # Crisis indicators (important for mental health)
    'suicide',
    'suicidal',
    'die',
    'dying',
    'death',
    'kill',
    'end',
    'goodbye',
    'harm',
    'hurt',
    'cutting',
    'selfharm',
]


# Loads the model
MODEL_PATH = 'ml_pipeline/toy_models/LRmodel.pkl'  # Path to model used to predict
MODEL = joblib.load(MODEL_PATH)


def analyze_text(analyzed_text):
    prediction = MODEL.predict([analyzed_text])

    label = prediction[0]
    proba = MODEL.predict_proba([analyzed_text])[0]
    confidence = max(proba)
    model_version = ModelVersion.objects.filter(
        is_active=True
    ).first()  # Placeholder, just uses the first model in the database

    return (label, confidence, model_version)


def get_prediction_result(user, user_text):
    prediction, confidence, model_version = analyze_text(user_text)

    # Calculate metrics
    anxiety_level, negativity_level, emotional_intensity, word_count, char_count = (
        calculate_metrics(user_text)
    )

    # Generate recommendations
    recommendations = get_recommendations(prediction, confidence, anxiety_level)

    if user:
        save_prediction_to_database(
            user, user_text, prediction, confidence, model_version, recommendations
        )

    confidence_percentage = round(confidence * 100)

    return (
        prediction,
        confidence_percentage,
        recommendations,
        anxiety_level,
        negativity_level,
        emotional_intensity,
        word_count,
        char_count,
    )


def get_recommendations(prediction, confidence, anxiety_level):
    recommendations = []

    if confidence > 0.70:  # high confidence results
        if prediction == 'normal':
            recommendations.append(
                'You seem to present an undoubtedly healthy mental state. Remember to maintain regular sleep, nutrition, and physical activity. Check in with yourself regularly about emotional wellbeing, and do not hesitate to reach out for help in case of need.'
            )
        elif prediction == 'stress':
            recommendations.append(
                'You seem to present strong indicators of stress, try to limit caffeine and consider mindfulness meditation for 10 minutes daily. Delegate tasks where possible and practice setting boundaries, focusing on one task at a time can make a big difference.'
            )
        elif prediction == 'suicidal':
            recommendations.append(
                'You are not alone. If you are having thoughts of taking your life or if you know anyone who does, contact Mind Suicide Line dialing 90101 (open 24/7) or contact them on their online chat. For immediate crisis, call 988 (Suicide & Crisis Lifeline).'
            )
        elif prediction == 'depression':
            recommendations.append(
                'You seem to present strong indicators of depression, consider speaking with a mental health professional this week. Establish a consistent sleep schedule and try behavioral activation: schedule one pleasant activity each day.'
            )

    elif confidence >= 0.50 and confidence <= 0.70:  # medium confidence
        if prediction == 'normal':
            recommendations.append(
                'You seem to have a healthy mental state. Try to maintain regular sleep, nutrition, and physical activity. Check in with yourself regularly about emotional wellbeing, and do not hesitate to reach out for help in case of need.'
            )
        elif prediction == 'stress':
            recommendations.append(
                'You seem to present medium indicators of stress, try to limit caffeine and consider mindfulness meditation for 10 minutes daily. Delegate tasks where possible and practice setting boundaries, focusing on one task at a time can make a big difference.'
            )
        elif prediction == 'suicidal':
            recommendations.append(
                'The analysis suggests you may have thoughts of taking your own life. If that is true or you know anyone who does, contact Mind Suicide Line dialing 90101 (open 24/7) or contact them on their online chat. For immediate crisis, call 988 (Suicide & Crisis Lifeline).'
            )
        elif prediction == 'depression':
            recommendations.append(
                'You seem to present mild indicators of depression, consider speaking with a mental health professional if you feel your situation requires professional help. Establish a consistent sleep schedule and try behavioral activation: schedule one pleasant activity each day.'
            )

    else:  # low confidence (< 0.50)
        recommendations.append(
            'Model confidence is low. Consider these general wellness suggestions:'
        )
        if prediction == 'suicidal':
            recommendations.append(
                'The analysis suggests you may have thoughts of taking your own life. If that is true or you know anyone who does, contact Mind Suicide Line dialing 90101 (open 24/7) or contact them on their online chat. Even with low confidence, this is important: For immediate crisis, call 988 (Suicide & Crisis Lifeline).'
            )
        elif prediction == 'depression':
            recommendations.append(
                'Check in with yourself regularly about emotional wellbeing. In case you feel symptoms of depression, do not hesitate to reach out for professional help.'
            )
        elif prediction == 'stress':
            recommendations.append(
                'In case you feel stressed, try to limit caffeine and consider mindfulness meditation for 10 minutes daily. Delegate tasks where possible and practice setting boundaries, focusing on one task at a time can make a big difference.'
            )

        else:
            recommendations.append(
                'Your mental state seems normal. Ensure to maintain regular sleep, nutrition, and physical activity. Check in with yourself regularly about emotional wellbeing, and do not hesitate to reach out for help in case of need.'
            )

    if anxiety_level >= 50:
        recommendations.append(
            'Additionally, your text presents a high anxiety level, consider practicing daily grounding techniques (5-4-3-2-1 method). Try to take a step back and think long-term; it might feel less overwhelming. Remember to always prioritize your mental health.'
        )

    return recommendations


def calculate_metrics(text: str):
    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)
    char_count = len(text)

    # Negativity level based on key words
    negativity_count = sum(words.count(w) for w in NEGATIVE_WORDS)
    negativity_level = min(int((negativity_count / max(word_count, 1)) * 100), 100)

    # Extract general text features (punctuation, use of caps, word density)
    period_count = len(re.findall(r'(?<!\.)\.(?!\.)', text))  # Single periods only
    ellipsis_count = len(re.findall(r'\.{2,}', text))  # Two or more dots

    question_count = len(re.findall(r'\?', text))
    exclamation_count = len(re.findall(r'!', text))

    capital_count = sum(1 for c in text if c.isupper())
    word_density = word_count / max(char_count, 1)  # words per character

    # Anxiety: Ellipses weighted heavier than periods (uncertainty, trailing thoughts)
    anxiety_score = (
        (ellipsis_count * 50)
        + (period_count * 20)
        + (word_density * 25)
        + (capital_count * 0.5)
    )

    # Emotional intensity: Exclamations > Questions (strong emotion > curiosity)
    emotional_score = (
        (exclamation_count * 40)
        + (question_count * 30)
        + (capital_count * 20)
        + (word_density * 10)
    )

    anxiety_level = min(int(anxiety_score), 100)
    emotional_intensity = min(int(emotional_score), 100)

    return anxiety_level, negativity_level, emotional_intensity, word_count, char_count


def save_prediction_to_database(
    user, user_text, prediction, confidence, model_version, recommendations
):
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
