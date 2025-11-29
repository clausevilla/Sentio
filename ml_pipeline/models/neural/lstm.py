# Author: Marcus Berggren
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(hidden)


class LSTMModel:
    """
    Adapted RNN LSTM (Long Short-Term Memory) from toymodel, using Pytorch instead of Tensorflow.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        'embed_dim': 64,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'max_seq_len': 512,
        'vocab_size': 50000,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs': 10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model: Optional[LSTMClassifier] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self, vocab_size: int, num_classes: int) -> None:
        self.model = LSTMClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=self.config['embed_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
        ).to(self.device)

    def get_num_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
