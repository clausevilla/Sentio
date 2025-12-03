# Author: Marcus Berggren
import numpy as np
import pandas as pd
import pytest

from ..logistic_regression import LogisticRegressionModel

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


# Inherits from conftest.py
@pytest.fixture
def trained_model(sample_data):
    X_train, y_train, _, _ = sample_data
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)
    return model


class TestLogisticRegressionInitialization:
    def test_pipeline_created(self):
        model = LogisticRegressionModel()
        assert model.pipeline is not None
        assert 'tfidf' in model.pipeline.named_steps
        assert 'classifier' in model.pipeline.named_steps

    def test_default_config(self):
        model = LogisticRegressionModel()
        assert model.config['max_iter'] == 1000
        assert model.config['C'] == 1.0
        assert model.config['solver'] == 'lbfgs'

    def test_custom_config_overrides_defaults(self):
        config = {'max_iter': 2000, 'C': 0.5}
        model = LogisticRegressionModel(config)

        assert model.config['max_iter'] == 2000
        assert model.config['C'] == 0.5
        assert model.config['solver'] == 'lbfgs'  # default preserved

    def test_tfidf_config_override(self):
        config = {'tfidf': {'ngram_range': (1, 3)}}
        model = LogisticRegressionModel(config)

        assert model.config['tfidf']['ngram_range'] == (1, 3)
        assert model.config['tfidf']['max_df'] == 0.95  # default preserved


class TestLogisticRegressionTraining:
    def test_fit_returns_self(self, sample_data):
        X_train, y_train, _, _ = sample_data
        model = LogisticRegressionModel()
        result = model.fit(X_train, y_train)
        assert result is model

    def test_fit_creates_coefficients(self, trained_model):
        classifier = trained_model.pipeline.named_steps['classifier']
        assert hasattr(classifier, 'coef_')

    def test_fit_learns_classes(self, trained_model):
        assert len(trained_model.pipeline.classes_) == 5


class TestLogisticRegressionPrediction:
    def test_predict_returns_array(self, trained_model, sample_data):
        _, _, X_test, _ = sample_data
        predictions = trained_model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)

    def test_predict_returns_valid_classes(self, trained_model, sample_data):
        _, _, X_test, _ = sample_data
        predictions = trained_model.predict(X_test)
        valid_classes = set(trained_model.pipeline.classes_)

        assert all(p in valid_classes for p in predictions)

    def test_predict_proba_shape(self, trained_model, sample_data):
        _, _, X_test, _ = sample_data
        proba = trained_model.predict_proba(X_test)

        assert proba.shape == (len(X_test), 5)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_handles_mixed_case_input(self, trained_model):
        X_mixed = pd.Series(['HAPPY HAPPY', 'sad sad'])
        predictions = trained_model.predict(X_mixed)
        assert predictions.shape == (2,)


class TestLogisticRegressionEvaluation:
    def test_evaluate_returns_expected_keys(self, trained_model, sample_data):
        _, _, X_test, y_test = sample_data
        metrics = trained_model.evaluate(X_test, y_test)

        expected_keys = [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'auc',
            'confusion_matrix',
            'classification_report',
        ]
        for key in expected_keys:
            assert key in metrics

    def test_evaluate_metrics_in_valid_range(self, trained_model, sample_data):
        _, _, X_test, y_test = sample_data
        metrics = trained_model.evaluate(X_test, y_test)

        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0


class TestLogisticRegressionFeatureImportance:
    def test_feature_importance_structure(self, trained_model):
        importance = trained_model.get_feature_importance(top_n=10)

        assert importance['type'] == 'per_class'
        assert isinstance(importance['data'], dict)
        assert len(importance['data']) == 5  # one per class

    def test_feature_importance_per_class_limit(self, trained_model):
        importance = trained_model.get_feature_importance(top_n=5)

        for class_features in importance['data'].values():
            assert len(class_features) <= 5

    def test_feature_importance_values_are_floats(self, trained_model):
        importance = trained_model.get_feature_importance(top_n=5)

        for class_features in importance['data'].values():
            for value in class_features.values():
                assert isinstance(value, float)
