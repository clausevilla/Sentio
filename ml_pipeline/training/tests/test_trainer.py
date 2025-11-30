# Author: Marcus Berggren
import numpy as np
import pandas as pd
import pytest
import torch

from ml_pipeline.storage.handler import StorageHandler
from ml_pipeline.training.trainer import ModelTrainer

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


@pytest.fixture
def storage_handler(tmp_path):
    return StorageHandler(model_dir=str(tmp_path / 'models'))


@pytest.fixture
def trainer(storage_handler):
    return ModelTrainer(storage_handler)


@pytest.fixture
def sample_data():
    X_train = pd.Series(
        [
            'i feel happy and good',
            'i am very depressed and sad',
            'i want to help others',
            'i feel anxious and worried',
            'i am stressed about work',
        ]
        * 20
    )
    y_train = pd.Series([0, 1, 2, 3, 4] * 20)

    X_test = pd.Series(['feeling good today', 'very sad and depressed'])
    y_test = pd.Series([0, 1])

    return X_train, y_train, X_test, y_test


class TestModelTrainerInitialization:
    def test_device_set(self, trainer):
        assert trainer.device in [torch.device('cpu'), torch.device('cuda')]

    def test_evaluator_initialized(self, trainer):
        assert trainer.evaluator is not None

    def test_sklearn_models_registered(self, trainer):
        assert 'logistic_regression' in trainer.SKLEARN_MODELS
        assert 'random_forest' in trainer.SKLEARN_MODELS

    def test_pytorch_models_registered(self, trainer):
        assert 'lstm' in trainer.PYTORCH_MODELS
        assert 'transformer' in trainer.PYTORCH_MODELS


class TestModelTrainerValidation:
    def test_rejects_non_tuple_data(self, trainer):
        with pytest.raises(TypeError, match='must be a tuple'):
            trainer.train('logistic_regression', ['not', 'a', 'tuple'], {}, 'job1')

    def test_rejects_wrong_tuple_length(self, trainer):
        with pytest.raises(ValueError, match='tuple of length'):
            trainer.train('logistic_regression', (1, 2, 3), {}, 'job1')

    def test_rejects_unknown_model(self, trainer, sample_data):
        with pytest.raises(ValueError, match='Unknown model'):
            trainer.train('unknown_model', sample_data, {}, 'job1')


class TestSklearnTraining:
    def test_logistic_regression_returns_success(self, trainer, sample_data):
        result = trainer.train('logistic_regression', sample_data, {}, 'test_lr')

        assert result['status'] == 'success'
        assert result['model_type'] == 'logistic_regression'
        assert result['job_id'] == 'test_lr'

    def test_logistic_regression_returns_metrics(self, trainer, sample_data):
        result = trainer.train('logistic_regression', sample_data, {}, 'test_lr')

        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 'precision' in result['metrics']
        assert 'recall' in result['metrics']
        assert 'f1_score' in result['metrics']

    def test_logistic_regression_saves_model(self, trainer, sample_data):
        result = trainer.train('logistic_regression', sample_data, {}, 'test_lr')

        assert 'model_path' in result
        assert 'logistic_regression_test_lr.pkl' in result['model_path']

    def test_random_forest_returns_success(self, trainer, sample_data):
        result = trainer.train('random_forest', sample_data, {}, 'test_rf')

        assert result['status'] == 'success'
        assert result['model_type'] == 'random_forest'

    def test_config_passed_to_model(self, trainer, sample_data):
        config = {'max_iter': 50, 'C': 0.5}
        result = trainer.train('logistic_regression', sample_data, config, 'test_cfg')

        assert result['status'] == 'success'

    def test_metrics_in_valid_range(self, trainer, sample_data):
        result = trainer.train('logistic_regression', sample_data, {}, 'test_lr')
        metrics = result['metrics']

        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0


class TestNeuralTraining:
    @pytest.fixture
    def minimal_neural_config(self):
        return {
            'epochs': 1,
            'batch_size': 16,
            'max_seq_len': 32,
            'vocab_size': 1000,
            'd_model': 32,
            'nhead': 2,
            'num_layers': 1,
            'dim_feedforward': 64,
            'embed_dim': 32,
            'hidden_dim': 32,
        }

    def test_lstm_returns_success(self, trainer, sample_data, minimal_neural_config):
        result = trainer.train('lstm', sample_data, minimal_neural_config, 'test_lstm')

        assert result['status'] == 'success'
        assert result['model_type'] == 'lstm'

    def test_lstm_returns_metrics(self, trainer, sample_data, minimal_neural_config):
        result = trainer.train('lstm', sample_data, minimal_neural_config, 'test_lstm')

        assert 'metrics' in result
        assert 'accuracy' in result['metrics']

    def test_lstm_saves_model(self, trainer, sample_data, minimal_neural_config):
        result = trainer.train('lstm', sample_data, minimal_neural_config, 'test_lstm')

        assert 'model_path' in result
        assert 'lstm_test_lstm.pt' in result['model_path']

    def test_transformer_returns_success(
        self, trainer, sample_data, minimal_neural_config
    ):
        result = trainer.train(
            'transformer', sample_data, minimal_neural_config, 'test_tf'
        )

        assert result['status'] == 'success'
        assert result['model_type'] == 'transformer'

    def test_handles_numpy_arrays(self, trainer, minimal_neural_config):
        X_train = np.array(['text one', 'text two', 'text three'] * 20)
        y_train = np.array([0, 1, 2] * 20)
        X_test = np.array(['test text'])
        y_test = np.array([0])

        data = (X_train, y_train, X_test, y_test)
        result = trainer.train('lstm', data, minimal_neural_config, 'test_np')

        assert result['status'] == 'success'


class TestIncrementalTraining:
    @pytest.fixture
    def minimal_neural_config(self):
        return {
            'epochs': 1,
            'batch_size': 16,
            'max_seq_len': 32,
            'vocab_size': 1000,
            'embed_dim': 32,
            'hidden_dim': 32,
            'num_layers': 1,
        }

    def test_incremental_training_loads_base_model(
        self, trainer, sample_data, minimal_neural_config
    ):
        initial_result = trainer.train(
            'lstm', sample_data, minimal_neural_config, 'base'
        )

        incremental_config = {
            **minimal_neural_config,
            'training_mode': 'incremental',
            'base_model_path': initial_result['model_path'],
        }

        result = trainer.train('lstm', sample_data, incremental_config, 'incremental')

        assert result['status'] == 'success'

    def test_incremental_with_vocab_expansion(
        self, trainer, sample_data, minimal_neural_config
    ):
        initial_result = trainer.train(
            'lstm', sample_data, minimal_neural_config, 'base'
        )

        X_train, y_train, X_test, y_test = sample_data
        X_train_new = pd.concat(
            [X_train, pd.Series(['completely novel vocabulary here'] * 20)]
        )
        y_train_new = pd.concat([y_train, pd.Series([0] * 20)])

        incremental_config = {
            **minimal_neural_config,
            'training_mode': 'incremental',
            'base_model_path': initial_result['model_path'],
            'expand_vocab': True,
        }

        new_data = (X_train_new, y_train_new, X_test, y_test)
        result = trainer.train('lstm', new_data, incremental_config, 'expanded')

        assert result['status'] == 'success'


class TestModelPersistence:
    def test_sklearn_model_can_be_loaded(self, trainer, sample_data, storage_handler):
        result = trainer.train('logistic_regression', sample_data, {}, 'persist_test')

        loaded = storage_handler.load_sklearn_model(result['model_path'])

        assert loaded is not None
        assert hasattr(loaded, 'predict')

    def test_neural_model_can_be_loaded(
        self, trainer, sample_data, storage_handler, tmp_path
    ):
        config = {
            'epochs': 1,
            'batch_size': 16,
            'max_seq_len': 32,
            'embed_dim': 32,
            'hidden_dim': 32,
            'num_layers': 1,
        }
        result = trainer.train('lstm', sample_data, config, 'persist_test')

        checkpoint = storage_handler.load_neural_model(result['model_path'])

        assert 'model_state_dict' in checkpoint
        assert 'tokenizer' in checkpoint
        assert 'label_encoder' in checkpoint
        assert 'config' in checkpoint
