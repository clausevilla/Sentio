# Author: Marcus Berggren
import pytest
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from ml_pipeline.models.neural.lstm import LSTMModel
from ml_pipeline.models.neural.transformer import TransformerModel
from ml_pipeline.storage.handler import StorageHandler
from ml_pipeline.training.data import WordTokenizer

from ..predictor import Predictor

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


@pytest.fixture
def storage_handler(tmp_path):
    return StorageHandler(model_dir=str(tmp_path / 'models'))


@pytest.fixture
def predictor(storage_handler):
    return Predictor(storage_handler)


@pytest.fixture
def fitted_sklearn_pipeline():
    pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression()),
        ]
    )
    texts = ['i feel happy', 'i am sad', 'feeling stressed'] * 10
    labels = ['positive', 'negative', 'stress'] * 10
    pipeline.fit(texts, labels)
    return pipeline


@pytest.fixture
def saved_sklearn_model(storage_handler, fitted_sklearn_pipeline):
    path = storage_handler.save_sklearn_model(
        fitted_sklearn_pipeline, 'test_model.joblib'
    )
    return path


@pytest.fixture
def mock_tokenizer():
    tokenizer = WordTokenizer(vocab_size=1000)
    tokenizer.fit(['i feel happy', 'i am sad', 'feeling stressed'] * 10)
    return tokenizer


@pytest.fixture
def mock_label_encoder():
    le = LabelEncoder()
    le.fit(['positive', 'negative', 'stress'])
    return le


@pytest.fixture
def saved_transformer_model(storage_handler, mock_tokenizer, mock_label_encoder):
    config = {
        'max_seq_len': 32,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'dropout': 0.1,
    }

    wrapper = TransformerModel(config)
    wrapper.build_model(
        vocab_size=mock_tokenizer.vocab_size_actual(),
        num_classes=len(mock_label_encoder.classes_),
    )

    path = storage_handler.save_neural_model(
        model=wrapper.model,
        tokenizer=mock_tokenizer,
        label_encoder=mock_label_encoder,
        config=config,
        filename='transformer_test.pt',
        model_type='transformer',
    )
    return path


@pytest.fixture
def saved_lstm_model(storage_handler, mock_tokenizer, mock_label_encoder):
    config = {
        'max_seq_len': 32,
        'embed_dim': 32,
        'hidden_dim': 32,
        'num_layers': 1,
        'dropout': 0.0,
    }

    wrapper = LSTMModel(config)
    wrapper.build_model(
        vocab_size=mock_tokenizer.vocab_size_actual(),
        num_classes=len(mock_label_encoder.classes_),
    )

    path = storage_handler.save_neural_model(
        model=wrapper.model,
        tokenizer=mock_tokenizer,
        label_encoder=mock_label_encoder,
        config=config,
        filename='lstm_test.pt',
        model_type='lstm',
    )
    return path


class TestPredictorInitialization:
    def test_device_set(self, predictor):
        assert predictor.device in [torch.device('cpu'), torch.device('cuda')]

    def test_model_initially_none(self, predictor):
        assert predictor.model is None

    def test_tokenizer_initially_none(self, predictor):
        assert predictor.tokenizer is None

    def test_label_encoder_initially_none(self, predictor):
        assert predictor.label_encoder is None


class TestPredictorLoad:
    def test_load_sklearn_model(self, predictor, saved_sklearn_model):
        predictor.load(saved_sklearn_model)

        assert predictor.model is not None
        assert predictor.model_type == 'sklearn'

    def test_load_transformer_model(self, predictor, saved_transformer_model):
        predictor.load(saved_transformer_model)

        assert predictor.model is not None
        assert predictor.model_type == 'neural'
        assert predictor.tokenizer is not None
        assert predictor.label_encoder is not None

    def test_load_lstm_model(self, predictor, saved_lstm_model):
        predictor.load(saved_lstm_model)

        assert predictor.model is not None
        assert predictor.model_type == 'neural'

    def test_load_unknown_format_raises(self, predictor):
        with pytest.raises(ValueError, match='Unknown model format'):
            predictor.load('model.unknown')

    def test_neural_model_in_eval_mode(self, predictor, saved_transformer_model):
        predictor.load(saved_transformer_model)

        assert not predictor.model.training


class TestPredictorPredictSklearn:
    @pytest.fixture
    def loaded_predictor(self, predictor, saved_sklearn_model):
        predictor.load(saved_sklearn_model)
        return predictor

    def test_returns_dict(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result, dict)

    def test_contains_required_keys(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert 'label' in result
        assert 'confidence' in result
        assert 'probabilities' in result

    def test_label_is_string(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result['label'], str)

    def test_label_is_valid_class(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert result['label'] in ['positive', 'negative', 'stress']

    def test_confidence_is_float(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result['confidence'], float)

    def test_confidence_in_valid_range(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert 0.0 <= result['confidence'] <= 1.0

    def test_probabilities_is_dict(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result['probabilities'], dict)

    def test_probabilities_sum_to_one(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_contain_all_classes(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert set(result['probabilities'].keys()) == {'positive', 'negative', 'stress'}

    def test_confidence_matches_predicted_class_probability(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert result['confidence'] == result['probabilities'][result['label']]


class TestPredictorPredictNeural:
    @pytest.fixture
    def loaded_predictor(self, predictor, saved_transformer_model):
        predictor.load(saved_transformer_model)
        return predictor

    def test_returns_dict(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result, dict)

    def test_contains_required_keys(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert 'label' in result
        assert 'confidence' in result
        assert 'probabilities' in result

    def test_label_is_string(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result['label'], str)

    def test_label_is_valid_class(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert result['label'] in ['positive', 'negative', 'stress']

    def test_confidence_is_float(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result['confidence'], float)

    def test_confidence_in_valid_range(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert 0.0 <= result['confidence'] <= 1.0

    def test_probabilities_sum_to_one(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_contain_all_classes(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert set(result['probabilities'].keys()) == {'positive', 'negative', 'stress'}


class TestPredictorPredictLSTM:
    @pytest.fixture
    def loaded_predictor(self, predictor, saved_lstm_model):
        predictor.load(saved_lstm_model)
        return predictor

    def test_returns_dict(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert isinstance(result, dict)

    def test_contains_required_keys(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert 'label' in result
        assert 'confidence' in result
        assert 'probabilities' in result

    def test_confidence_in_valid_range(self, loaded_predictor):
        result = loaded_predictor.predict('i feel happy today')

        assert 0.0 <= result['confidence'] <= 1.0


class TestPredictorNoModelLoaded:
    def test_predict_raises_without_load(self, predictor):
        with pytest.raises(RuntimeError, match='No model loaded'):
            predictor.predict('some text')

    def test_get_class_names_raises_without_load(self, predictor):
        with pytest.raises(RuntimeError, match='No model loaded'):
            predictor.get_class_names()


class TestPredictorGetClassNames:
    def test_sklearn_class_names(self, predictor, saved_sklearn_model):
        predictor.load(saved_sklearn_model)

        class_names = predictor.get_class_names()

        assert isinstance(class_names, list)
        assert set(class_names) == {'positive', 'negative', 'stress'}

    def test_neural_class_names(self, predictor, saved_transformer_model):
        predictor.load(saved_transformer_model)

        class_names = predictor.get_class_names()

        assert isinstance(class_names, list)
        assert set(class_names) == {'positive', 'negative', 'stress'}
