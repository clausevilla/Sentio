# Author: Karl Byland

import joblib
import pandas as pd

from apps.ml_admin.models import ModelVersion
from apps.predictions.models import PredictionResult, TextSubmission
from ml_pipeline.data_cleaning.cleaner import DataCleaningPipeline
from ml_pipeline.preprocessing.preprocessor import DataPreprocessingPipeline

# Loads the model
MODEL_PATH = 'ml_pipeline/toy_models/LRmodel.pkl'  # Path to model used to predict
MODEL = joblib.load(MODEL_PATH)


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


def preprocess_user_input(df, version_name):
    pipeline = DataPreprocessingPipeline()
    df = pipeline.preprocess_dataframe(df, version_name)
    return df['processed_text']


def get_prediction_result(user, user_text):
    model_version = (
        ModelVersion.objects.first()
    )  # !!!Placeholder, just uses the first model in the database

    df = clean_user_input(user_text)
    processed_text = preprocess_user_input(df, model_version.version_name)
    prediction, confidence = analyze_text(processed_text.iloc[0])

    if user:
        save_prediction_to_database(
            user, user_text, prediction, confidence, model_version
        )

    confidence_percentage = round(confidence * 100)

    return prediction, confidence_percentage


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
