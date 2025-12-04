# Author: Karl Byland

import logging

from django.conf import settings

from apps.ml_admin.models import ModelVersion
from apps.predictions.models import PredictionResult, TextSubmission
from ml_pipeline.inference.predictor import Predictor
from ml_pipeline.storage.handler import StorageHandler

logger = logging.getLogger(__name__)

_predictor = None
_loaded_model_id = None


def get_predictor():
    """
    Get or create predictor with the active model loaded.

    Caches the predictor to avoid reloading on every request.
    Reloads if the active model changes.
    """
    global _predictor, _loaded_model_id

    active_model = ModelVersion.objects.filter(
        is_active=True
    ).first()  # First and only available model

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

    return _predictor, active_model


def analyze_text(text):
    """
    Analyze text and return prediction results.

    Returns:
        Tuple of (label, confidence, model_version)
    """
    predictor, model_version = get_predictor()
    result = predictor.predict(text)

    return result['label'], result['confidence'], model_version


def get_prediction_result(user, user_text):
    """
    Get prediction and optionally save to database.

    Returns:
        Tuple of (prediction_label, confidence_percentage)
    """
    label, confidence, model_version = analyze_text(user_text)

    if user:
        save_prediction_to_database(user, user_text, label, confidence, model_version)

    confidence_percentage = round(confidence * 100)

    return label, confidence_percentage


def save_prediction_to_database(user, user_text, prediction, confidence, model_version):
    """Save submission and prediction result to database."""
    submission = TextSubmission.objects.create(user=user, text_content=user_text)
    # Save to database
    PredictionResult.objects.create(
        submission=submission,
        model_version=model_version,
        mental_state=prediction,
        confidence=confidence,
        recommendations='Follow recommended steps',  # placeholder text
    )
