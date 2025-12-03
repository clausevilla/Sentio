# Author: Marcus Berggren
import numpy as np
import pytest

from ..random_forest import RandomForestModel

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


# Inherits from conftest.py
@pytest.fixture
def trained_model(sample_data):
    """Pre-trained model for tests that need it"""
    X_train, y_train, _, _ = sample_data
    model = RandomForestModel()
    model.fit(X_train, y_train)
    return model


class TestRandomForestInitialization:
    def test_pipeline_created(self):
        model = RandomForestModel()
        assert model.pipeline is not None
        assert 'tfidf' in model.pipeline.named_steps
        assert 'classifier' in model.pipeline.named_steps

    def test_default_config(self):
        model = RandomForestModel()
        assert model.config['n_estimators'] == 100
        assert model.config['max_depth'] is None
        assert model.config['n_jobs'] == -1

    def test_custom_config_overrides_defaults(self):
        config = {
            'n_estimators': 200,
            'max_depth': 10,
        }
        model = RandomForestModel(config)

        assert model.config['n_estimators'] == 200
        assert model.config['max_depth'] == 10
        assert model.config['min_samples_split'] == 2  # default preserved

    def test_tfidf_config_override(self):
        config = {'tfidf': {'ngram_range': (1, 3), 'min_df': 5}}
        model = RandomForestModel(config)

        assert model.config['tfidf']['ngram_range'] == (1, 3)
        assert model.config['tfidf']['min_df'] == 5
        assert model.config['tfidf']['max_df'] == 0.95  # default preserved


class TestRandomForestTraining:
    def test_fit_returns_self(self, sample_data):
        X_train, y_train, _, _ = sample_data
        model = RandomForestModel()
        result = model.fit(X_train, y_train)
        assert result is model

    def test_fit_learns_classes(self, sample_data):
        X_train, y_train, _, _ = sample_data
        model = RandomForestModel()
        model.fit(X_train, y_train)
        assert hasattr(model.pipeline, 'classes_')
        assert len(model.pipeline.classes_) == 5


class TestRandomForestPrediction:
    def test_predict_returns_array(self, trained_model, sample_data):
        _, _, X_test, _ = sample_data
        predictions = trained_model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)

    def test_predict_proba_shape(self, trained_model, sample_data):
        _, _, X_test, _ = sample_data
        proba = trained_model.predict_proba(X_test)

        assert proba.shape == (len(X_test), 5)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestRandomForestEvaluation:
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
            'confusion_matrix_labels',
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


class TestRandomForestFeatureImportance:
    def test_feature_importance_structure(self, trained_model):
        importance = trained_model.get_feature_importance(top_n=10)

        assert importance['type'] == 'global'
        assert isinstance(importance['data'], dict)
        assert len(importance['data']) <= 10

    def test_feature_importance_values_are_floats(self, trained_model):
        importance = trained_model.get_feature_importance(top_n=5)

        for value in importance['data'].values():
            assert isinstance(value, float)
            assert value >= 0.0
