# Author: Marcus Berggren
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Small dataset for testing"""
    X_train = pd.Series(
        [
            'I feel happy and good',
            'I am very depressed and sad',
            'I want to kill myself',
            'I feel anxious and worried',
            'I am stressed about work',
        ]
        * 10
    )

    y_train = pd.Series([0, 1, 2, 3, 4] * 10)

    X_test = pd.Series(['feeling good today', 'very sad and depressed'])
    y_test = pd.Series([0, 1])

    return X_train, y_train, X_test, y_test
