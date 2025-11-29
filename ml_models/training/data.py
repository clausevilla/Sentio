# Author: Marcus Berggren
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class WordTokenizer:
    """
    Word-level tokenizer that builds vocabulary from training data.

    Expects preprocessed (lowercased) text.
    """

    def __init__(self, vocab_size: int = 50000):
        """
        Args:
            vocab_size: Maximum vocabulary size including <PAD> and <UNK>
        """
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}

    def fit(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.

        Counts word frequencies and keeps the most common words
        up to vocab_size. Words are lowercased.

        Args:
            texts: List of training texts
        """
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency descending, take top (vocab_size - 2) words
        # -2 because <PAD> and <UNK> are already in vocabulary
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[: self.vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx

    def encode(self, text: str, max_len: int) -> List[int]:
        """
        Convert text to list of token IDs.

        Args:
            text: Input text to encode
            max_len: Fixed output length

        Returns:
            List of integer token IDs with length max_len
        """
        tokens = [self.word2idx.get(w, 1) for w in text.split()]

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))

        return tokens

    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size (may be less than max if corpus is small)."""
        return len(self.word2idx)


class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification.

    Reference:
        https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: WordTokenizer,
        max_len: int,
    ):
        """
        Args:
            texts: List of raw text strings
            labels: Encoded integer labels
            tokenizer: Fitted WordTokenizer instance
            max_len: Maximum sequence length for padding/truncation
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Get single sample as dict of tensors.

        Args:
            idx: Sample index

        Returns:
            Dict with 'input_ids' and 'label' tensors
        """
        tokens = self.tokenizer.encode(self.texts[idx], max_len=self.max_len)
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
