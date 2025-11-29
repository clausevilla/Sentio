# Author: Marcus Berggren
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

"""
Encoder-only transformer for text classification.

Adapted from "Attention Is All You Need" (Vaswani et al., 2017) and Medium links.
Decoder omitted since classification requires encoding input, not generating output.

Classes:
    PositionalEncoding: Adds position information to token embeddings
    TransformerClassifier: The neural network (nn.Module)
    TransformerModel: Wrapper with sklearn-like interface for trainer.py to work

References:
    https://medium.com/@sayedebad.777/mastering-transformer-detailed-insights-into-each-block-and-their-math-4221c6ee0076
    https://medium.com/@sayedebad.777/building-a-transformer-from-scratch-a-step-by-step-guide-a3df0aeb7c9a
    https://arxiv.org/pdf/1706.03762
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)

        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1)  # (seq, 1)

        # Create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000 ** (2i / d_model))

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq, d_model)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Encoder-only transformer for text classification.

    This is not a full encoder-decoder transformer (as per paper and articles).
    For classification there's just need for the encoder to build representations, then pool and classify.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
    ):
        """
        Args:
            vocab_size: Number of tokens
            num_classes: Number of output classes
            d_model: Embedding dimension
            nhead: Number of attention heads. Must divide d_model evenly.
            num_layers: Number of encoder layers stacked. More = deeper model.
            dim_feedforward: Hidden size in feed-forward sublayer.
            dropout: Dropout probability for regularization
            max_seq_len: Maximum tokens per input sequence
            pad_idx: Token ID used for padding (ignored in attention)
        """
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding layer: (batch, seq_len) -> (batch, seq_len, d_model)
        # padding_idx ensures padding tokens get zero vectors
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # PyTorch's built-in transformer encoder layer
        # Each layer contains:
        #   1. Multi-head self-attention
        #   2. Add & normalize (residual connection + layer norm)
        #   3. Feed-forward network (two linear layers with ReLU)
        #   4. Add & normalize
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, features) not (seq, batch, features)
        )

        # Stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head: takes pooled representation, outputs class logits
        # Two-layer MLP with ReLU activation and dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # Loop through all parameters in the model
        # Only matrices (weights), skip vectors (biases) and reinitialize iwht Xavies
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x == self.pad_idx

    def _mean_pool(
        self, hidden_states: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # Invert mask: True for real tokens, False for padding
        # Unsqueeze adds dimension for broadcasting with hidden_states
        mask = (~padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)

        # Zero out padding positions and sum
        masked_hidden = hidden_states * mask  # (batch, seq_len, d_model)
        summed = masked_hidden.sum(dim=1)  # (batch, d_model)

        # Divide by number of real tokens (clamp prevents division by zero)
        lengths = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)

        return summed / lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sets boolen values
        padding_mask = self._create_padding_mask(x)

        # Embedding + scaling
        # Scaling by sqrt(d_model) keeps variance stable when combined with
        # positional encoding (which has values in [-1, 1])
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Self-attention
        # src_key_padding_mask tells attention which positions to ignore
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Aggregate sequence into single vector
        pooled = self._mean_pool(x, padding_mask)

        # Classifying
        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class TransformerModel:
    DEFAULT_CONFIG: Dict[str, Any] = {
        # Architecture (for model)
        'd_model': 128,  # Embedding dimension
        'nhead': 4,  # Attention heads
        'num_layers': 2,  # Encoder layers
        'dim_feedforward': 256,  # Feed-forward hidden size
        'dropout': 0.1,  # Dropout probability
        'max_seq_len': 512,  # Maximum input length
        'vocab_size': 50000,  # Vocabulary size for tokenizer
        # Training (for trainer)
        'learning_rate': 1e-4,  # 0.0001
        'batch_size': 32,
        'epochs': 10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Override default configuration values
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model: Optional[TransformerClassifier] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self, vocab_size: int, num_classes: int) -> None:
        self.model = TransformerClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            max_seq_len=self.config['max_seq_len'],
        ).to(self.device)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration (for saving with model)."""
        return self.config.copy()
