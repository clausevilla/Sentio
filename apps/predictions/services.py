# Author: Karl Byland, Claudia Sevilla, Lian Shi

import json
import logging
import re

import pandas as pd
from django.conf import settings

from apps.ml_admin.models import ModelVersion
from apps.predictions.models import PredictionResult, TextSubmission
from ml_pipeline.data_cleaning.cleaner import DataCleaningPipeline
from ml_pipeline.inference.predictor import Predictor
from ml_pipeline.preprocessing.preprocessor import DataPreprocessingPipeline
from ml_pipeline.storage.handler import StorageHandler

logger = logging.getLogger(__name__)

_predictor = None
_loaded_model_id = None


def load_json():
    DATA_PATH = 'apps/predictions/data/strings.json'
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    return text_data


def get_predictor(active_model):
    """
    Get or create predictor with the active model loaded.

    Caches the predictor to avoid reloading on every request.
    Reloads if the active model changes.
    """
    global _predictor, _loaded_model_id

    if active_model is None:
        raise RuntimeError('No active model configured')

    if _predictor is None or _loaded_model_id != active_model.id:
        storage = StorageHandler(
            model_dir=settings.MODEL_DIR,
            # Determined by runtime settings
            gcs_bucket=getattr(settings, 'GCS_BUCKET', None),
            use_gcs=getattr(settings, 'USE_GCS', False),
        )
        _predictor = Predictor(storage)
        _predictor.load(active_model.model_file_path)
        _loaded_model_id = active_model.id
        logger.info(f'Loaded model: {active_model.version_name}')

    return _predictor


def clean_user_input(text):
    data = {'text': [text]}
    df = pd.DataFrame(data)
    pipeline = DataCleaningPipeline()
    df = pipeline.fix_encoding(df)
    return df


def analyze_text(text, model_version):
    """
    Analyze text and return prediction results.

    Returns:
        Tuple of (label, confidence, model_version, all_confidences)
    """
    predictor = get_predictor(model_version)
    result = predictor.predict(text)

    return result['label'], result['confidence'], result['probabilities'], model_version


def preprocess_user_input(df, model_type):
    pipeline = DataPreprocessingPipeline()
    pipeline_version = ''
    model_to_pipeline = {
        'lstm': 'traditional',  # Seems to work better with full preprocessing
        'random_forest': 'traditional',
        'transformer': 'traditional',  # Seems to work better with full preprocessing
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
    """
    Get prediction and optionally save to database.

    Returns:
        Tuple of (prediction_label, confidence_percentage, recommendations,
                  anxiety_level, negativity_level, emotional_intensity,
                  word_count, char_count)
    """

    model_version = ModelVersion.objects.filter(
        is_active=True
    ).first()  # First and only available model

    df = clean_user_input(user_text)
    processed_text = preprocess_user_input(df, model_version.model_type)

    label, confidence, all_confidences_dict, model_version = analyze_text(
        processed_text.iloc[0], model_version
    )

    text_data = load_json()

    # Calculate metrics
    anxiety_level, negativity_level, emotional_intensity, word_count, char_count = (
        calculate_metrics(user_text, text_data['negative_words'])
    )

    # Generate recommendations
    recommendations = get_recommendations(
        label, confidence, anxiety_level, text_data['recommendations']
    )

    if user:
        save_prediction_to_database(
            user=user,
            user_text=user_text,
            prediction=label,
            confidence=confidence,
            example_texts=text_data['example_texts'],
            model_version=model_version,
            recommendations=recommendations,
            anxiety_level=anxiety_level,
            negativity_level=negativity_level,
            emotional_intensity=emotional_intensity,
        )

    confidence_percentage = round(confidence * 100)

    all_confidences_percentage = {}
    for class_name, prob in all_confidences_dict.items():
        all_confidences_percentage[class_name] = round(prob * 100)

    return (
        label,
        confidence_percentage,
        recommendations,
        anxiety_level,
        negativity_level,
        emotional_intensity,
        word_count,
        char_count,
        all_confidences_percentage,
    )


def get_recommendations(prediction, confidence, anxiety_level, recommendations_strings):
    recommendations = []
    prediction_lower = str(prediction).lower()
    confidence_percentage = confidence * 100 if confidence < 1 else confidence

    if confidence_percentage > 70:  # high confidence results
        if prediction_lower == 'normal':
            recommendations.append(  # Index 0 because maybe we want to add random recs in future
                recommendations_strings['normal']['high_confidence'][0]
            )
        elif prediction_lower == 'stress':
            recommendations.append(
                recommendations_strings['stress']['high_confidence'][0]
            )
        elif prediction_lower == 'suicidal':
            recommendations.append(
                recommendations_strings['suicidal']['high_confidence'][0]
            )
        elif prediction_lower == 'depression':
            recommendations.append(
                recommendations_strings['depression']['high_confidence'][0]
            )

    elif (
        confidence_percentage >= 50 and confidence_percentage < 70
    ):  # medium confidence
        if prediction_lower == 'normal':
            recommendations.append(
                recommendations_strings['normal']['medium_confidence'][0]
            )
        elif prediction_lower == 'stress':
            recommendations.append(
                recommendations_strings['stress']['medium_confidence'][0]
            )
        elif prediction_lower == 'suicidal':
            recommendations.append(
                recommendations_strings['suicidal']['medium_confidence'][0]
            )
        elif prediction_lower == 'depression':
            recommendations.append(
                recommendations_strings['depression']['medium_confidence'][0]
            )

    else:  # low confidence (< 0.50)
        recommendations.append(recommendations_strings['low_model_confidence'])
        if prediction_lower == 'suicidal':
            recommendations.append(
                recommendations_strings['suicidal']['low_confidence'][0]
            )
        elif prediction_lower == 'depression':
            recommendations.append(
                recommendations_strings['depression']['low_confidence'][0]
            )
        elif prediction_lower == 'stress':
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
        (ellipsis_count * 20)
        + (period_count * 3)
        + (word_density * 15)
        + (capital_count * 0.5)
    )

    # Emotional intensity: Exclamations > Questions (strong emotion > curiosity)
    emotional_score = (
        (exclamation_count * 8)
        + (question_count * 5)
        + (capital_count * 4)
        + (word_density * 10)
    )

    anxiety_level = min(int(anxiety_score), 100)
    emotional_intensity = min(int(emotional_score), 100)

    return anxiety_level, negativity_level, emotional_intensity, word_count, char_count


def save_prediction_to_database(
    user,
    user_text,
    prediction,
    confidence,
    model_version,
    recommendations,
    example_texts,
    anxiety_level=None,
    negativity_level=None,
    emotional_intensity=None,
):
    """
    Save the prediction result and associated metrics to the database.

    Excludes template/example texts from being saved to avoid cluttering
    the user's history with demonstration data.

    Args:
        user: The authenticated user making the prediction
        user_text: The original text submitted for analysis
        prediction: The predicted mental state label
        confidence: Model confidence score (0-1)
        model_version: The ModelVersion instance used for prediction
        recommendations: List of recommendation strings
        anxiety_level: Calculated anxiety metric (0-100)
        negativity_level: Calculated negativity metric (0-100)
        emotional_intensity: Calculated emotional intensity metric (0-100)
    """

    if user_text not in example_texts:
        # Create the text submission record
        submission = TextSubmission.objects.create(user=user, text_content=user_text)

        # Convert recommendations list to string for storage
        recommendations_text = (
            '\n'.join(recommendations)
            if isinstance(recommendations, list)
            else recommendations
        )

        # Create the prediction result with all metrics
        PredictionResult.objects.create(
            submission=submission,
            model_version=model_version,
            mental_state=prediction,
            confidence=confidence,
            recommendations=recommendations_text,
            anxiety_level=anxiety_level,
            negativity_level=negativity_level,
            emotional_intensity=emotional_intensity,
        )
