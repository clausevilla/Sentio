# Author: Marcus Berggren
import pytest
import torch

from ..data import TextDataset, WordTokenizer

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


class TestWordTokenizerInitialization:
    def test_default_vocab_size(self):
        tokenizer = WordTokenizer()
        assert tokenizer.vocab_size == 50000

    def test_custom_vocab_size(self):
        tokenizer = WordTokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000

    def test_special_tokens_initialized(self):
        tokenizer = WordTokenizer()
        assert tokenizer.word2idx['<PAD>'] == 0
        assert tokenizer.word2idx['<UNK>'] == 1


class TestWordTokenizerFit:
    def test_fit_builds_vocabulary(self, sample_texts):
        tokenizer = WordTokenizer()
        tokenizer.fit(sample_texts)

        assert tokenizer.vocab_size_actual() > 2
        assert 'feel' in tokenizer.word2idx
        assert 'happy' in tokenizer.word2idx

    def test_fit_respects_vocab_limit(self, sample_texts):
        tokenizer = WordTokenizer(vocab_size=5)
        tokenizer.fit(sample_texts)

        assert tokenizer.vocab_size_actual() <= 5

    def test_fit_prioritizes_frequent_words(self):
        texts = ['common common common', 'rare', 'common common']
        tokenizer = WordTokenizer(vocab_size=4)
        tokenizer.fit(texts)

        assert 'common' in tokenizer.word2idx
        assert 'rare' not in tokenizer.word2idx


class TestWordTokenizerEncode:
    @pytest.fixture
    def fitted_tokenizer(self, sample_texts):
        tokenizer = WordTokenizer()
        tokenizer.fit(sample_texts)
        return tokenizer

    def test_encode_known_words(self, fitted_tokenizer):
        tokens = fitted_tokenizer.encode('i feel happy', max_len=5)

        assert len(tokens) == 5
        assert tokens[0] == fitted_tokenizer.word2idx['i']
        assert tokens[1] == fitted_tokenizer.word2idx['feel']
        assert tokens[2] == fitted_tokenizer.word2idx['happy']

    def test_encode_unknown_words_become_unk(self, fitted_tokenizer):
        tokens = fitted_tokenizer.encode('i feel xyz123', max_len=5)

        assert tokens[2] == 1  # <UNK>

    def test_encode_pads_short_sequences(self, fitted_tokenizer):
        tokens = fitted_tokenizer.encode('i feel', max_len=5)

        assert len(tokens) == 5
        assert tokens[2:] == [0, 0, 0]  # <PAD>

    def test_encode_truncates_long_sequences(self, fitted_tokenizer):
        tokens = fitted_tokenizer.encode('i feel happy and good today', max_len=3)

        assert len(tokens) == 3

    def test_encode_returns_list_of_ints(self, fitted_tokenizer):
        tokens = fitted_tokenizer.encode('i feel happy', max_len=5)

        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)


class TestWordTokenizerExpandVocab:
    def test_expand_adds_new_words(self, sample_texts):
        tokenizer = WordTokenizer(vocab_size=100)
        tokenizer.fit(sample_texts)
        initial_size = tokenizer.vocab_size_actual()

        new_texts = ['completely new words here']
        added = tokenizer.expand_vocab(new_texts)

        assert added > 0
        assert tokenizer.vocab_size_actual() == initial_size + added

    def test_expand_preserves_existing_mappings(self, sample_texts):
        tokenizer = WordTokenizer()
        tokenizer.fit(sample_texts)
        original_idx = tokenizer.word2idx['feel']

        tokenizer.expand_vocab(['new words'])

        assert tokenizer.word2idx['feel'] == original_idx

    def test_expand_respects_vocab_limit(self, sample_texts):
        tokenizer = WordTokenizer(vocab_size=10)
        tokenizer.fit(sample_texts)

        new_texts = ['a b c d e f g h i j k l m n o p']
        tokenizer.expand_vocab(new_texts)

        assert tokenizer.vocab_size_actual() <= 10

    def test_expand_returns_zero_when_full(self, sample_texts):
        tokenizer = WordTokenizer(vocab_size=5)
        tokenizer.fit(sample_texts)

        added = tokenizer.expand_vocab(['new words here'])

        assert added == 0


class TestTextDataset:
    @pytest.fixture
    def tokenizer(self, sample_texts):
        tokenizer = WordTokenizer()
        tokenizer.fit(sample_texts)
        return tokenizer

    @pytest.fixture
    def dataset(self, sample_texts, sample_labels, tokenizer):
        return TextDataset(sample_texts, sample_labels, tokenizer, max_len=10)

    def test_len(self, dataset, sample_texts):
        assert len(dataset) == len(sample_texts)

    def test_getitem_returns_dict(self, dataset):
        item = dataset[0]

        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'label' in item

    def test_getitem_input_ids_shape(self, dataset):
        item = dataset[0]

        assert item['input_ids'].shape == (10,)

    def test_getitem_input_ids_dtype(self, dataset):
        item = dataset[0]

        assert item['input_ids'].dtype == torch.long

    def test_getitem_label_dtype(self, dataset):
        item = dataset[0]

        assert item['label'].dtype == torch.long

    def test_getitem_label_value(self, dataset, sample_labels):
        for i, label in enumerate(sample_labels):
            item = dataset[i]
            assert item['label'].item() == label

    def test_works_with_dataloader(self, dataset):
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))

        assert batch['input_ids'].shape == (2, 10)
        assert batch['label'].shape == (2,)
