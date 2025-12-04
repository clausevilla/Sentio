import numpy as np
import pytest


@pytest.fixture
def sample_texts():
    return [
        'i feel happy and good',
        'i am very depressed and sad',
        'i want to help others',
        'i feel anxious and worried',
        'i am stressed about work',
    ]


@pytest.fixture
def sample_labels():
    return np.array([0, 1, 2, 3, 4])
