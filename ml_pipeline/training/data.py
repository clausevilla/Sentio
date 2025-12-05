# Author: Marcus Berggren
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class WordTokenizer:
    """
    Word-level tokenizer that builds vocabulary from training data.

    Expects preprocessed (lowercased) text.

    Splits text on whitespace, maps words to integer IDs for neural network input.
    Unknown words map to <UNK> (index 1), padding is <PAD> (index 0).

    Rather than utilizing existing tokenizer from BERT, we utilized idea from Pytorch link below,
    but since usecase is only classification and not decoding it's simplified and a limit of
    vocabulary size set. The current dataset does not go over 50k unique tokens.

    Reference:
        https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    1. Count word frequencies
    2. Keep top N words
    3. Map word to integer
    4. Handle unknowns with <UNK>
    5. Handle padding with <PAD>

    Example:
        tokenizer = WordTokenizer(vocab_size=10000)
        tokenizer.fit(["i feel anxious", "i am happy"])

        tokenizer.encode("i feel great", max_len=5)
            [2, 3, 1, 0, 0]
            i feel <UNK> <PAD> <PAD>
            'great' not in vocab -> <UNK> (1)
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
        up to vocab_size.

        Args:
            texts: List of training texts (expected to be preprocessed/lowercased)
        """
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency descending, take top (vocab_size - 2) words
        # -2 because <PAD> and <UNK> are already in vocabulary
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[: self.vocab_size - 2]:
            self.word2idx[word] = len(self.word2idx)

    def expand_vocab(self, texts: List[str]) -> int:
        """
        Add new words from texts to existing vocabulary.

        Used for incremental training when new data may contain words
        not seen during initial training. Preserves existing word -> index
        mappings and only adds new words up to vocab_size limit.

        Args:
            texts: List of texts that may contain new words

        Returns:
            Number of new words added to vocabulary
        """
        word_counts = {}
        for text in texts:
            for word in text.split():
                if word not in self.word2idx:
                    word_counts[word] = word_counts.get(word, 0) + 1

        remaining_slots = self.vocab_size - len(self.word2idx)
        if remaining_slots <= 0:
            return 0

        sorted_new = sorted(word_counts.items(), key=lambda x: -x[1])
        new_words = 0
        for word, _ in sorted_new[:remaining_slots]:
            self.word2idx[word] = len(self.word2idx)
            new_words += 1

        return new_words

    def encode(self, text: str, max_len: int) -> List[int]:
        """
        Convert text to list of token IDs.

        Words not in vocabulary become <UNK> (1).
        Sequences longer than max_len are truncated.
        Sequences shorter than max_len are padded with <PAD> (0).

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

    Tokenizes texts on-the-fly and returns tensors ready for model input.

    Reference:
        https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    Example:
        dataset = TextDataset(texts, labels, tokenizer, max_len=256)
        loader = DataLoader(dataset, batch_size=32)

        for batch in loader:
            input_ids = batch['input_ids']  # (32, 256)
            labels = batch['label']         # (32,)
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
