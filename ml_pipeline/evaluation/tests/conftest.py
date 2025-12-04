import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def label_encoder():
    le = LabelEncoder()
    le.fit(['class_a', 'class_b', 'class_c'])
    return le


@pytest.fixture
def binary_label_encoder():
    le = LabelEncoder()
    le.fit(['negative', 'positive'])
    return le


@pytest.fixture
def simple_model(device):
    """Model that outputs predictable logits based on input"""
    model = nn.Linear(10, 3)
    model.to(device)
    return model


@pytest.fixture
def mock_dataloader():
    """DataLoader with known inputs and labels"""
    input_ids = torch.randint(0, 100, (20, 16))
    labels = torch.tensor([0, 1, 2] * 6 + [0, 1])

    class SimpleDataset:
        def __init__(self, input_ids, labels):
            self.input_ids = input_ids
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {'input_ids': self.input_ids[idx], 'label': self.labels[idx]}

    dataset = SimpleDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=4)
