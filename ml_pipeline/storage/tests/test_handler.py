# Author: Marcus Berggren
import pytest
import torch
import torch.nn as nn

from ..handler import StorageHandler

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


@pytest.fixture
def handler(tmp_path):
    return StorageHandler(model_dir=str(tmp_path / 'models'))


@pytest.fixture
def fitted_sklearn_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression()),
        ]
    )
    pipeline.fit(['hello world', 'test text', 'another sample'], [0, 1, 0])
    return pipeline


@pytest.fixture
def mock_tokenizer():
    from ml_pipeline.training.data import WordTokenizer

    tokenizer = WordTokenizer(vocab_size=100)
    tokenizer.fit(['hello world', 'test text'])
    return tokenizer


@pytest.fixture
def mock_label_encoder():
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(['class_a', 'class_b', 'class_c'])
    return le


class TestStorageHandlerInitialization:
    def test_creates_model_directory(self, tmp_path):
        model_dir = tmp_path / 'new_models'
        StorageHandler(model_dir=str(model_dir))

        assert model_dir.exists()

    def test_creates_nested_directories(self, tmp_path):
        model_dir = tmp_path / 'deep' / 'nested' / 'models'
        StorageHandler(model_dir=str(model_dir))

        assert model_dir.exists()

    def test_default_no_gcs_client(self, handler):
        assert handler.gcs_client is None

    def test_gcs_client_none_without_credentials(self, tmp_path):
        handler = StorageHandler(
            model_dir=str(tmp_path), gcs_bucket='nonexistent-bucket'
        )

        assert handler.gcs_client is None


class TestSklearnModelSave:
    def test_returns_path_string(self, handler, fitted_sklearn_pipeline):
        path = handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.joblib')

        assert isinstance(path, str)
        assert 'model.joblib' in path

    def test_creates_file(self, handler, fitted_sklearn_pipeline):
        handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.joblib')

        assert (handler.model_dir / 'model.joblib').exists()

    def test_file_is_valid_joblib(self, handler, fitted_sklearn_pipeline):
        import joblib

        path = handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.joblib')
        loaded = joblib.load(path)

        assert hasattr(loaded, 'predict')

    def test_converts_pkl_extension_to_joblib(self, handler, fitted_sklearn_pipeline):
        path = handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.pkl')

        assert path.endswith('.joblib')
        assert (handler.model_dir / 'model.joblib').exists()


class TestSklearnModelLoad:
    def test_returns_pipeline(self, handler, fitted_sklearn_pipeline):
        path = handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.joblib')
        loaded = handler.load_sklearn_model(path)

        assert hasattr(loaded, 'named_steps')

    def test_preserves_pipeline_structure(self, handler, fitted_sklearn_pipeline):
        path = handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.joblib')
        loaded = handler.load_sklearn_model(path)

        assert 'tfidf' in loaded.named_steps
        assert 'classifier' in loaded.named_steps

    def test_preserves_fitted_state(self, handler, fitted_sklearn_pipeline):
        path = handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.joblib')
        loaded = handler.load_sklearn_model(path)

        predictions = loaded.predict(['hello world', 'test'])

        assert len(predictions) == 2

    def test_predictions_match_original(self, handler, fitted_sklearn_pipeline):
        test_input = ['hello world', 'new text']
        original_predictions = fitted_sklearn_pipeline.predict(test_input)

        path = handler.save_sklearn_model(fitted_sklearn_pipeline, 'model.joblib')
        loaded = handler.load_sklearn_model(path)
        loaded_predictions = loaded.predict(test_input)

        assert list(original_predictions) == list(loaded_predictions)


class TestNeuralModelSave:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))

    def test_returns_path_string(
        self, handler, simple_model, mock_tokenizer, mock_label_encoder
    ):
        path = handler.save_neural_model(
            model=simple_model,
            tokenizer=mock_tokenizer,
            label_encoder=mock_label_encoder,
            config={'test': True},
            filename='model.pt',
            model_type='transformer',
        )

        assert isinstance(path, str)
        assert 'model.pt' in path

    def test_creates_file(
        self, handler, simple_model, mock_tokenizer, mock_label_encoder
    ):
        handler.save_neural_model(
            model=simple_model,
            tokenizer=mock_tokenizer,
            label_encoder=mock_label_encoder,
            config={},
            filename='model.pt',
            model_type='transformer',
        )

        assert (handler.model_dir / 'model.pt').exists()

    def test_file_is_valid_torch_checkpoint(
        self, handler, simple_model, mock_tokenizer, mock_label_encoder
    ):
        path = handler.save_neural_model(
            model=simple_model,
            tokenizer=mock_tokenizer,
            label_encoder=mock_label_encoder,
            config={},
            filename='model.pt',
            model_type='transformer',
        )

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        assert 'model_state_dict' in checkpoint

    def test_saves_model_type(
        self, handler, simple_model, mock_tokenizer, mock_label_encoder
    ):
        path = handler.save_neural_model(
            model=simple_model,
            tokenizer=mock_tokenizer,
            label_encoder=mock_label_encoder,
            config={},
            filename='model.pt',
            model_type='lstm',
        )

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        assert checkpoint['model_type'] == 'lstm'


class TestNeuralModelLoad:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))

    @pytest.fixture
    def saved_model_path(
        self, handler, simple_model, mock_tokenizer, mock_label_encoder
    ):
        return handler.save_neural_model(
            model=simple_model,
            tokenizer=mock_tokenizer,
            label_encoder=mock_label_encoder,
            config={'d_model': 128, 'num_layers': 2},
            filename='model.pt',
            model_type='transformer',
        )

    def test_returns_dict(self, handler, saved_model_path):
        checkpoint = handler.load_neural_model(saved_model_path)

        assert isinstance(checkpoint, dict)

    def test_contains_required_keys(self, handler, saved_model_path):
        checkpoint = handler.load_neural_model(saved_model_path)

        assert 'model_state_dict' in checkpoint
        assert 'tokenizer' in checkpoint
        assert 'label_encoder' in checkpoint
        assert 'config' in checkpoint
        assert 'model_type' in checkpoint

    def test_preserves_model_weights(
        self, handler, simple_model, mock_tokenizer, mock_label_encoder
    ):
        original_weight = simple_model[0].weight.clone()

        path = handler.save_neural_model(
            model=simple_model,
            tokenizer=mock_tokenizer,
            label_encoder=mock_label_encoder,
            config={},
            filename='model.pt',
            model_type='transformer',
        )

        checkpoint = handler.load_neural_model(path)
        new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
        new_model.load_state_dict(checkpoint['model_state_dict'])

        assert torch.allclose(new_model[0].weight, original_weight)

    def test_preserves_tokenizer(self, handler, saved_model_path):
        checkpoint = handler.load_neural_model(saved_model_path)
        tokenizer = checkpoint['tokenizer']

        assert hasattr(tokenizer, 'word2idx')
        assert '<PAD>' in tokenizer.word2idx

    def test_preserves_label_encoder(self, handler, saved_model_path):
        checkpoint = handler.load_neural_model(saved_model_path)
        label_encoder = checkpoint['label_encoder']

        assert hasattr(label_encoder, 'classes_')
        assert len(label_encoder.classes_) == 3

    def test_preserves_config(self, handler, saved_model_path):
        checkpoint = handler.load_neural_model(saved_model_path)

        assert checkpoint['config']['d_model'] == 128
        assert checkpoint['config']['num_layers'] == 2

    def test_preserves_model_type(self, handler, saved_model_path):
        checkpoint = handler.load_neural_model(saved_model_path)

        assert checkpoint['model_type'] == 'transformer'


class TestListModels:
    def test_empty_directory(self, handler):
        models = handler.list_models()

        assert models == []

    def test_returns_model_files(self, handler):
        (handler.model_dir / 'model1.pkl').touch()
        (handler.model_dir / 'model2.pt').touch()

        models = handler.list_models()

        assert len(models) == 2
        assert 'model1.pkl' in models
        assert 'model2.pt' in models

    def test_excludes_directories(self, handler):
        (handler.model_dir / 'model.pkl').touch()
        (handler.model_dir / 'subdir').mkdir()

        models = handler.list_models()

        assert models == ['model.pkl']

    def test_returns_all_file_types(self, handler):
        (handler.model_dir / 'sklearn.pkl').touch()
        (handler.model_dir / 'neural.pt').touch()
        (handler.model_dir / 'config.json').touch()

        models = handler.list_models()

        assert len(models) == 3
