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
    """
    Injects positional information into token embeddings.

    Transformers process all tokens in parallel (unlike RNNs) but with no info of order.
    Positional encoding adds that information for each position using sine and cosine waves
    at different frequencies. Sinusoidal because it allows model to learn relative positions.

    The formula from "Attention Is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length the model will encounter
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # Vector of shape (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Vector of shape (d_model / 2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even indices get sine, odd indices get cosine
        # Binary:      position 0 = 000, position 1 = 001, position 2 = 010
        # Sinusoidal:  position 0 = [0.0, 1.0, ...], position 1 = [0.84, 0.54, ...], etc
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_seq_len, d_model) -> (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but should be saved with model
        # and moved to correct device automatically)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedded tokens of shape (batch_size, seq_len, d_model)

        Returns:
            Position-encoded embeddings of same shape
        """
        # Slice positional encoding to match actual sequence length
        # This allows variable-length sequences up to max_seq_len
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Encoder-only transformer for text classification.

    Architecture:
        1. Token embedding layer (converts token IDs to vectors)
        2. Positional encoding (adds position information)
        3. Transformer encoder stack (self-attention + feed-forward layers)
        4. Mean pooling (aggregates sequence into single vector)
        5. Classification head (MLP that outputs class logits)

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
        """
        Initialize parameters. Applies to parameters that are matrices and
        leaves biases (since only 1 dimension).

        Example:
            Before:
            layer1.weights = [[-2.3, 5.1, -0.001], [99.2, -0.05, 3.2]] <- wild range

            AfteR:
            layer1.weights = [[0.12, -0.34, 0.21], [-0.18, 0.29, -0.15]] <- more centered

        """
        # Loop through all parameters in the model
        # Only matrices (weights), skip vectors (biases) and reinitialize iwht Xavies
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create mask that identifies padding tokens.

        The transformer's attention mechanism will ignore positions marked True,
        preventing the model from attending to padding tokens.

        Example:
            'I feel anxious'                -> [42, 87, 203, 0, 0, 0, 0]
            'Depression is overwhelming me' -> [91, 33, 445, 12, 0, 0, 0]
            'Help'                          -> [88, 0, 0, 0, 0, 0, 0]

        Args:
            x: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Boolean mask of shape (batch_size, seq_len) where True = padding
        """
        return x == self.pad_idx

    def _mean_pool(
        self, hidden_states: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate token representations into a single sequence representation.

        Transformer encoder outputs one vector per token, but classification
        needs one vector per sequence.

        Example:
            hidden_states for 'I feel anxious <PAD> <PAD>':
                'I'       -> [0.2, 0.5, 0.1, 0.8]
                'feel'    -> [0.8, 0.3, 0.4, 0.2]
                'anxious' -> [0.6, 0.9, 0.2, 0.7]
                <PAD>     -> [0.1, 0.1, 0.1, 0.1]  <- ignore
                <PAD>     -> [0.3, 0.2, 0.0, 0.4]  <- ignore

            padding_mask: [False, False, False, True, True]

        Computation:
            sum  = [0.2+0.8+0.6, 0.5+0.3+0.9, 0.1+0.4+0.2, 0.8+0.2+0.7] -> [1.6, 1.7, 0.7, 1.7]
            count = 3 (real tokens only)
            result = [0.53, 0.57, 0.23, 0.57] <- sum / count

            This single vector represents the entire sentence for classification.

        Args:
            hidden_states: Encoder output of shape (batch, seq_len, d_model)
            padding_mask: Boolean mask where True = padding token

        Returns:
            Pooled representation of shape (batch, d_model)
        """
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
        """
        Forward pass: tokens -> class logits.

        Very simplifies example with batch=1, seq_len=4, d_model=4, num_classes=4:

            Input token IDs:
                [[42, 87, 203, 0]]  # 'I feel anxious <PAD>'

            1. Padding mask (True = ignore this token):
                [[False, False, False, True]]

            2. Embedding lookup (each token ID maps to a learned vector):
                42  -> [[[0.12, 0.45, 0.23, 0.67] # random at init and later learned
                87  -> [0.89, 0.34, 0.56, 0.12]
                203 -> [0.45, 0.78, 0.23, 0.91]
                0   -> [0.00, 0.00, 0.00, 0.00]]] # padding always zeros

               Scale by sqrt(d_model) = sqrt(4) = 2:
                [[[0.24, 0.90, 0.46, 1.34],
                  [1.78, 0.68, 1.12, 0.24],
                  [0.90, 1.56, 0.46, 1.82],
                  [0.00, 0.00, 0.00, 0.00]]]

            3. Add positional encoding (sinusoidal patterns):
                position 0: + [0.00, 1.00, 0.00, 1.00]
                position 1: + [0.84, 0.54, 0.01, 0.99]
                position 2: + [0.91, -0.42, 0.02, 0.99]
                position 3: + [0.14, -0.99, 0.03, 0.99]

                Result:
                [[[0.24, 1.90, 0.46, 2.34],   # 'I' at position 0
                  [2.62, 1.22, 1.13, 1.23],   # 'feel' at position 1
                  [1.81, 1.14, 0.48, 2.81],   # 'anxious' at position 2
                  [0.14, -0.99, 0.03, 0.99]]] # <PAD> at position 3

            4. Encoder self-attention (each token attends to all others):
                'I' computes: 'How relevant is 'feel' to me? 'anxious' to me?'
                'feel' computes: 'How relevant is 'I' to me? 'anxious' to me?'
                'anxious' computes: 'How relevant is 'I' to me? 'feel' to me?'

                Attention weights for 'anxious' might be:
                    I: 0.15, feel: 0.60, anxious: 0.25
                    (high weight on 'feel' because 'feel anxious' is meaningful)

                Output: each token is now a weighted mix of text
                  [1.23, 0.98, 0.89, 1.45],   # 'anxious' + context (absorbed 'feel')
                  [0.02, 0.01, 0.01, 0.02]]]  # <PAD> (ignored via mask)

            5. Mean pool (avrage non-padding tokens):
                sum     = [0.45+0.67+1.23, 0.82+1.05+0.98, 0.34+0.78+0.89, 0.91+1.12+1.45]
                        = [2.35, 2.85, 2.01, 3.48]
                count   = 3
                pooled  = [0.78, 0.95, 0.67, 1.16]

            6. Classifier (linear -> ReLU -> dropout -> linear):
                [0.78, 0.95, 0.67, 1.16]
                    Linear(4, 4)
                [0.45, 0.89, 0.23, 0.67]
                    ReLU
                [0.45, 0.89, 0.23, 0.67]
                    Linear(4, 4)
                [0.12, 0.25, 0.08, 0.55]  # raw logits

                Classes: [Normal, Depression, Suicidal, Stress]
                Highest logit: Stress (0.55) -> prediction: Stress

        Args:
            x: Token IDs of shape (batch_size, seq_len)

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # 1. Sets boolen values
        padding_mask = self._create_padding_mask(x)

        # 2. Embedding + scaling
        # Scaling by sqrt(d_model) keeps variance stable when combined with
        # positional encoding (which has values in [-1, 1])
        x = self.embedding(x) * math.sqrt(self.d_model)

        # 3. Positional encoding
        x = self.pos_encoder(x)

        # 4. Self-attention
        # src_key_padding_mask tells attention which positions to ignore
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # 5. Aggregate sequence into single vector
        pooled = self._mean_pool(x, padding_mask)

        # 6. Classifying
        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (softmax over logits).

        Args:
            x: Token IDs of shape (batch_size, seq_len)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class TransformerModel:
    """
    Wrapper class that provides sklearn-like interface for the transformer.

    Matches earlier pattern created for LogisticRegressionModel, with aim to have
    trainer easily utilize different models.

    Responsibilities:
        - Configuration management (defaults + user overrides)
        - Model instantiation
        - Device management (CPU/GPU)
        - Serialization helpers

    The actual training loop lives in ModelTrainer, not here. This separation
    aims to have the model focused on architecture.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        # Architecture (for model)
        'd_model': 128,  #          Embedding dimension
        'nhead': 4,  #              Attention heads
        'num_layers': 2,  #         Encoder layers
        'dim_feedforward': 256,  #  Feed-forward hidden size
        'dropout': 0.1,  #          Dropout probability
        'max_seq_len': 512,  #      Maximum input length
        'vocab_size': 50000,  #     Vocabulary size for tokenizer
        # Training (for trainer)
        'learning_rate': 1e-4,  #   0.0001
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
        """
        Instantiate the transformer with given vocabulary and class count.

        Called by trainer after tokenizer is fitted (so vocab_size is known)
        and labels are encoded (so num_classes is known).

        Args:
            vocab_size: Actual vocabulary size from fitted tokenizer
            num_classes: Number of unique labels
        """
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
