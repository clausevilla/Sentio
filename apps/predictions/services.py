# Author: Karl Byland, Claudia Sevilla

import json
import re

import joblib
import pandas as pd

from apps.ml_admin.models import ModelVersion
from apps.predictions.models import PredictionResult, TextSubmission
from ml_pipeline.data_cleaning.cleaner import DataCleaningPipeline
from ml_pipeline.preprocessing.preprocessor import DataPreprocessingPipeline

# Loads the model
MODEL_PATH = 'ml_pipeline/toy_models/LRmodel.pkl'  # Path to model used to predict
MODEL = joblib.load(MODEL_PATH)
DATA_PATH = 'apps/predictions/data/strings.json'


def load_json():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    return text_data


def analyze_text(analyzed_text):
    predictions = MODEL.predict([analyzed_text])

    prediction = predictions[0]
    proba = MODEL.predict_proba([analyzed_text])[0]
    confidence = max(proba)

    return (prediction, confidence)


def clean_user_input(text):
    data = {'text': [text]}
    df = pd.DataFrame(data)
    pipeline = DataCleaningPipeline()
    df = pipeline.fix_encoding(df)
    return df


def preprocess_user_input(df, model_type):
    pipeline = DataPreprocessingPipeline()
    pipeline_version = ''
    model_to_pipeline = {
        'lstm': 'rnn',
        'random_forest': 'rnn',
        'transformer': 'transformer',
        'logistic_regression': 'traditional',
    }

    try:
        pipeline_version = model_to_pipeline[model_type]
    except KeyError:
        raise ValueError(
            f"Invalid model type '{model_type}'. Must be one of: {', '.join(model_to_pipeline.keys())}"
        )

    processed_tuple = pipeline.preprocess_dataframe(df, pipeline_version)
    return processed_tuple[0]['text_preprocessed']


def get_prediction_result(user, user_text):
    model_version = ModelVersion.objects.filter(
        is_active=True
    ).first()  # Takes the active model

    df = clean_user_input(user_text)
    processed_text = preprocess_user_input(df, model_version.model_type)
    prediction, confidence = analyze_text(processed_text.iloc[0])

    text_data = load_json()

    # Calculate metrics
    anxiety_level, negativity_level, emotional_intensity, word_count, char_count = (
        calculate_metrics(user_text, text_data['negative_words'])
    )

    # Generate recommendations
    recommendations = get_recommendations(
        prediction, confidence, anxiety_level, text_data['recommendations']
    )

    if user:
        save_prediction_to_database(
            user,
            user_text,
            prediction,
            confidence,
            model_version,
            recommendations,
            text_data,
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


def get_recommendations(prediction, confidence, anxiety_level, recommendations_strings):
    recommendations = []

    if confidence > 0.70:  # high confidence results
        if prediction == 'normal':
            recommendations.append(  # Index 0 because maybe we want to add random recs in future
                recommendations_strings['normal']['high_confidence'][0]
            )
        elif prediction == 'stress':
            recommendations.append(
                recommendations_strings['stress']['high_confidence'][0]
            )
        elif prediction == 'suicidal':
            recommendations.append(
                recommendations_strings['suicidal']['high_confidence'][0]
            )
        elif prediction == 'depression':
            recommendations.append(
                recommendations_strings['depression']['high_confidence'][0]
            )

    elif confidence >= 0.50 and confidence <= 0.70:  # medium confidence
        if prediction == 'normal':
            recommendations.append(
                recommendations_strings['normal']['medium_confidence'][0]
            )
        elif prediction == 'stress':
            recommendations.append(
                recommendations_strings['stress']['medium_confidence'][0]
            )
        elif prediction == 'suicidal':
            recommendations.append(
                recommendations_strings['suicidal']['medium_confidence'][0]
            )
        elif prediction == 'depression':
            recommendations.append(
                recommendations_strings['depression']['medium_confidence'][0]
            )

    else:  # low confidence (< 0.50)
        recommendations.append(recommendations_strings['low_model_confidence'])
        if prediction == 'suicidal':
            recommendations.append(
                recommendations_strings['suicidal']['low_confidence'][0]
            )
        elif prediction == 'depression':
            recommendations.append(
                recommendations_strings['depression']['low_confidence'][0]
            )
        elif prediction == 'stress':
            recommendations.append(
                recommendations_strings['stress']['low_confidence'][0]
            )

        else:
            recommendations.append(
                recommendations_strings['normal']['low_confidence'][0]
            )

    if anxiety_level >= 50:
        recommendations.append(recommendations_strings['anxiety'][0])

    return recommendations


def calculate_metrics(text: str, negative_words):
    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)
    char_count = len(text)

    # Negativity level based on key words
    negativity_count = sum(words.count(w) for w in negative_words)
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
    user, user_text, prediction, confidence, model_version, recommendations, text_data
):
    template_texts = text_data['example_texts']
    if user_text not in template_texts:
        submission = TextSubmission.objects.create(user=user, text_content=user_text)
        # Save to database
        PredictionResult.objects.create(
            submission=submission,
            model_version=model_version,
            mental_state=prediction,
            confidence=confidence,
            recommendations=recommendations,
        )
