"""
Google Cloud Storage utilities for ML models
"""

import os

import joblib
from google.cloud import storage

BUCKET_NAME = os.environ.get('GCS_BUCKET', 'sentio-m_l-models')
LOCAL_MODELS_DIR = '/app/ml-models'


def upload_model(model, model_name):
    """
    Save model locally and upload to GCS

    Args:
        model: The trained model object
        model_name: Name for the model file (e.g., 'logistic_v1.pkl')
    """
    # Ensure local directory exists
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

    # Save locally first
    local_path = f'{LOCAL_MODELS_DIR}/{model_name}'
    joblib.dump(model, local_path)
    print(f'Saved model locally: {local_path}')

    # Upload to GCS
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f'models/{model_name}')
        blob.upload_from_filename(local_path)
        print(f'Uploaded to GCS: gs://{BUCKET_NAME}/models/{model_name}')
    except Exception as e:
        print(f'Warning: Could not upload to GCS: {e}')


def download_model(model_name):
    """
    Download model from GCS (or use local cache)

    Args:
        model_name: Name of the model file

    Returns:
        The loaded model object
    """
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    local_path = f'{LOCAL_MODELS_DIR}/{model_name}'

    # Check if already downloaded
    if os.path.exists(local_path):
        print(f'Using cached model: {local_path}')
        return joblib.load(local_path)

    # Download from GCS
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f'models/{model_name}')
        blob.download_to_filename(local_path)
        print(f'Downloaded from GCS: {model_name}')
        return joblib.load(local_path)
    except Exception as e:
        print(f'Error downloading model: {e}')
        raise


def list_models():
    """List all models in GCS"""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix='models/')
        models = [
            blob.name for blob in blobs if blob.name.endswith(('.pkl', '.joblib'))
        ]
        return models
    except Exception as e:
        print(f'Error listing models: {e}')
        return []
