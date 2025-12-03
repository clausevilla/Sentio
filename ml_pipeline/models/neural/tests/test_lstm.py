# Author: Marcus Berggren
import pytest
import torch

from ml_pipeline.models.neural.lstm import LSTMClassifier, LSTMModel

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


class TestLSTMClassifier:
    @pytest.fixture
    def small_model(self):
        return LSTMClassifier(
            vocab_size=1000,
            num_classes=4,
            embed_dim=32,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
        )

    def test_forward_output_shape(self, small_model):
        batch_size = 2
        seq_len = 10
        x = torch.randint(1, 1000, (batch_size, seq_len))

        output = small_model(x)

        assert output.shape == (batch_size, 4)

    def test_different_sequence_lengths(self, small_model):
        short = torch.randint(1, 1000, (2, 5))
        long = torch.randint(1, 1000, (2, 50))

        short_out = small_model(short)
        long_out = small_model(long)

        assert short_out.shape == (2, 4)
        assert long_out.shape == (2, 4)

    def test_different_inputs_produce_different_outputs(self, small_model):
        x1 = torch.tensor([[1, 2, 3, 4, 5]])
        x2 = torch.tensor([[5, 4, 3, 2, 1]])

        out1 = small_model(x1)
        out2 = small_model(x2)

        assert not torch.allclose(out1, out2)

    def test_model_is_differentiable(self, small_model):
        x = torch.randint(1, 1000, (2, 10))

        output = small_model(x)
        loss = output.sum()
        loss.backward()

        for param in small_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_bidirectional_doubles_hidden(self):
        model = LSTMClassifier(
            vocab_size=1000,
            num_classes=4,
            embed_dim=32,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
        )

        assert model.classifier[0].in_features == 64

    def test_single_layer_no_dropout(self):
        model = LSTMClassifier(
            vocab_size=1000,
            num_classes=4,
            embed_dim=32,
            hidden_dim=32,
            num_layers=1,
            dropout=0.5,
        )

        assert model.lstm.dropout == 0.0

    def test_multi_layer_has_dropout(self):
        model = LSTMClassifier(
            vocab_size=1000,
            num_classes=4,
            embed_dim=32,
            hidden_dim=32,
            num_layers=2,
            dropout=0.5,
        )

        assert model.lstm.dropout == 0.5


class TestLSTMModel:
    def test_default_config(self):
        model = LSTMModel()

        assert model.config['embed_dim'] == 64
        assert model.config['hidden_dim'] == 64
        assert model.config['num_layers'] == 2

    def test_custom_config_overrides_defaults(self):
        model = LSTMModel({'embed_dim': 128, 'epochs': 20})

        assert model.config['embed_dim'] == 128
        assert model.config['epochs'] == 20
        assert model.config['hidden_dim'] == 64

    def test_model_is_none_before_build(self):
        model = LSTMModel()

        assert model.model is None

    def test_build_model_creates_classifier(self):
        model = LSTMModel()
        model.build_model(vocab_size=1000, num_classes=4)

        assert model.model is not None
        assert isinstance(model.model, LSTMClassifier)

    def test_get_num_parameters_zero_before_build(self):
        model = LSTMModel()

        assert model.get_num_parameters() == 0

    def test_get_num_parameters_nonzero_after_build(self):
        model = LSTMModel()
        model.build_model(vocab_size=1000, num_classes=4)

        assert model.get_num_parameters() > 0

    def test_get_config_returns_copy(self):
        model = LSTMModel()
        config = model.get_config()
        config['embed_dim'] = 999

        assert model.config['embed_dim'] == 64

    def test_device_is_set(self):
        model = LSTMModel()

        assert model.device in [torch.device('cpu'), torch.device('cuda')]

    def test_model_moved_to_device_after_build(self):
        model = LSTMModel()
        model.build_model(vocab_size=1000, num_classes=4)

        param_device = next(model.model.parameters()).device
        assert param_device == model.device

    def test_build_uses_config_values(self):
        model = LSTMModel({'embed_dim': 128, 'hidden_dim': 128, 'num_layers': 3})
        model.build_model(vocab_size=500, num_classes=7)

        assert model.model.embedding.embedding_dim == 128
        assert model.model.lstm.hidden_size == 128
        assert model.model.lstm.num_layers == 3
        assert model.model.classifier[-1].out_features == 7
