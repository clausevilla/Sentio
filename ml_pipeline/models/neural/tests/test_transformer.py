# Author: Marcus Berggren
import pytest
import torch

from ml_pipeline.models.neural.transformer import (
    PositionalEncoding,
    TransformerClassifier,
    TransformerModel,
)

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


class TestPositionalEncoding:
    def test_output_shape_matches_input(self):
        d_model = 64
        max_seq_len = 100
        pe = PositionalEncoding(d_model, max_seq_len)

        x = torch.randn(2, 50, d_model)
        output = pe(x)

        assert output.shape == x.shape

    def test_adds_positional_information(self):
        d_model = 64
        max_seq_len = 100
        pe = PositionalEncoding(d_model, max_seq_len, dropout=0.0)

        x = torch.zeros(1, 10, d_model)
        output = pe(x)

        assert not torch.allclose(output, x)

    def test_different_positions_have_different_encodings(self):
        d_model = 64
        max_seq_len = 100
        pe = PositionalEncoding(d_model, max_seq_len, dropout=0.0)

        x = torch.zeros(1, 10, d_model)
        output = pe(x)

        assert not torch.allclose(output[0, 0], output[0, 1])

    def test_handles_variable_sequence_lengths(self):
        d_model = 64
        max_seq_len = 100
        pe = PositionalEncoding(d_model, max_seq_len)

        short = torch.randn(2, 10, d_model)
        long = torch.randn(2, 50, d_model)

        short_out = pe(short)
        long_out = pe(long)

        assert short_out.shape == (2, 10, d_model)
        assert long_out.shape == (2, 50, d_model)

    def test_pe_buffer_is_registered(self):
        pe = PositionalEncoding(64, 100)

        assert 'pe' in dict(pe.named_buffers())

    def test_pe_buffer_shape(self):
        d_model = 64
        max_seq_len = 100
        pe = PositionalEncoding(d_model, max_seq_len)

        assert pe.pe.shape == (1, max_seq_len, d_model)


class TestTransformerClassifier:
    @pytest.fixture
    def small_model(self):
        return TransformerClassifier(
            vocab_size=1000,
            num_classes=4,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            max_seq_len=50,
        )

    def test_forward_output_shape(self, small_model):
        batch_size = 2
        seq_len = 10
        x = torch.randint(1, 1000, (batch_size, seq_len))

        output = small_model(x)

        assert output.shape == (batch_size, 4)

    def test_predict_proba_output_shape(self, small_model):
        batch_size = 2
        seq_len = 10
        x = torch.randint(1, 1000, (batch_size, seq_len))

        probs = small_model.predict_proba(x)

        assert probs.shape == (batch_size, 4)

    def test_predict_proba_sums_to_one(self, small_model):
        x = torch.randint(1, 1000, (2, 10))

        probs = small_model.predict_proba(x)

        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_predict_proba_values_between_zero_and_one(self, small_model):
        x = torch.randint(1, 1000, (2, 10))

        probs = small_model.predict_proba(x)

        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_padding_mask_creation(self, small_model):
        x = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])

        mask = small_model._create_padding_mask(x)

        expected = torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, True, True, True],
            ]
        )
        assert torch.equal(mask, expected)

    def test_mean_pool_ignores_padding(self, small_model):
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            ]
        )
        padding_mask = torch.tensor([[False, False, True]])

        pooled = small_model._mean_pool(hidden, padding_mask)

        expected = torch.tensor([[2.0, 3.0]])
        assert torch.allclose(pooled, expected)

    def test_handles_all_padding(self, small_model):
        hidden = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        padding_mask = torch.tensor([[True, True]])

        pooled = small_model._mean_pool(hidden, padding_mask)

        assert pooled.shape == (1, 2)
        assert not torch.isnan(pooled).any()

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


class TestTransformerModel:
    def test_default_config(self):
        model = TransformerModel()

        assert model.config['d_model'] == 128
        assert model.config['nhead'] == 4
        assert model.config['num_layers'] == 2

    def test_custom_config_overrides_defaults(self):
        model = TransformerModel({'d_model': 256, 'epochs': 20})

        assert model.config['d_model'] == 256
        assert model.config['epochs'] == 20
        assert model.config['nhead'] == 4

    def test_model_is_none_before_build(self):
        model = TransformerModel()

        assert model.model is None

    def test_build_model_creates_classifier(self):
        model = TransformerModel({'d_model': 32, 'nhead': 4})
        model.build_model(vocab_size=1000, num_classes=4)

        assert model.model is not None
        assert isinstance(model.model, TransformerClassifier)

    def test_get_num_parameters_zero_before_build(self):
        model = TransformerModel()

        assert model.get_num_parameters() == 0

    def test_get_num_parameters_nonzero_after_build(self):
        model = TransformerModel({'d_model': 32, 'nhead': 4})
        model.build_model(vocab_size=1000, num_classes=4)

        assert model.get_num_parameters() > 0

    def test_get_config_returns_copy(self):
        model = TransformerModel()
        config = model.get_config()
        config['d_model'] = 999

        assert model.config['d_model'] == 128

    def test_device_is_set(self):
        model = TransformerModel()

        assert model.device in [torch.device('cpu'), torch.device('cuda')]

    def test_model_moved_to_device_after_build(self):
        model = TransformerModel({'d_model': 32, 'nhead': 4})
        model.build_model(vocab_size=1000, num_classes=4)

        param_device = next(model.model.parameters()).device
        assert param_device == model.device
