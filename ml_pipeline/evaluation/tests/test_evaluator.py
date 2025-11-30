# Author: Marcus Berggren
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluator import ModelEvaluator

"""
Disclaimer: All tests in this file has been generated through use of an LLM.
"""


class TestModelEvaluatorInitialization:
    def test_default_device(self):
        evaluator = ModelEvaluator()

        assert evaluator.device in [torch.device('cpu'), torch.device('cuda')]

    def test_custom_device(self):
        evaluator = ModelEvaluator(device=torch.device('cpu'))

        assert evaluator.device == torch.device('cpu')


class TestQuickAccuracy:
    @pytest.fixture
    def evaluator(self, device):
        return ModelEvaluator(device=device)

    @pytest.fixture
    def matching_dataloader(self):
        """DataLoader with known labels: 0,1,2,0,1,2,..."""

        class SimpleDataset:
            def __len__(self):
                return 12

            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, 100, (16,)),
                    'label': torch.tensor(idx % 3),
                }

        return DataLoader(SimpleDataset(), batch_size=4, shuffle=False)

    @pytest.fixture
    def perfect_model(self, device):
        """Model that predicts class = idx % 3 for each sample"""

        class PerfectClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_idx = 0

            def forward(self, x):
                batch_size = x.shape[0]
                logits = torch.full((batch_size, 3), -10.0)
                for i in range(batch_size):
                    class_idx = self.sample_idx % 3
                    logits[i, class_idx] = 10.0
                    self.sample_idx += 1
                return logits

        return PerfectClassifier().to(device)

    @pytest.fixture
    def wrong_model(self, device):
        """Model that always predicts wrong class"""

        class WrongClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.sample_idx = 0

            def forward(self, x):
                batch_size = x.shape[0]
                logits = torch.full((batch_size, 3), -10.0)
                for i in range(batch_size):
                    correct_class = self.sample_idx % 3
                    wrong_class = (correct_class + 1) % 3
                    logits[i, wrong_class] = 10.0
                    self.sample_idx += 1
                return logits

        return WrongClassifier().to(device)

    @pytest.fixture
    def embedding_model(self, device):
        """Model that handles token IDs properly"""

        class EmbeddingClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 16)
                self.fc = nn.Linear(16, 3)

            def forward(self, x):
                x = self.embedding(x)
                x = x.mean(dim=1)
                return self.fc(x)

        return EmbeddingClassifier().to(device)

    def test_returns_float(self, evaluator, perfect_model, matching_dataloader):
        accuracy = evaluator.quick_accuracy(perfect_model, matching_dataloader)

        assert isinstance(accuracy, float)

    def test_perfect_accuracy(self, evaluator, perfect_model, matching_dataloader):
        accuracy = evaluator.quick_accuracy(perfect_model, matching_dataloader)

        assert accuracy == 1.0

    def test_zero_accuracy(self, evaluator, wrong_model, matching_dataloader):
        accuracy = evaluator.quick_accuracy(wrong_model, matching_dataloader)

        assert accuracy == 0.0

    def test_accuracy_in_valid_range(
        self, evaluator, embedding_model, matching_dataloader
    ):
        accuracy = evaluator.quick_accuracy(embedding_model, matching_dataloader)

        assert 0.0 <= accuracy <= 1.0

    def test_model_set_to_eval_mode(
        self, evaluator, embedding_model, matching_dataloader
    ):
        embedding_model.train()

        evaluator.quick_accuracy(embedding_model, matching_dataloader)

        assert not embedding_model.training


class TestEvaluateNeural:
    @pytest.fixture
    def evaluator(self, device):
        return ModelEvaluator(device=device)

    @pytest.fixture
    def dataloader_with_labels(self):
        """DataLoader with consistent labels"""

        class SimpleDataset:
            def __init__(self):
                self.labels = [0, 1, 2] * 6 + [0, 1]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, 100, (16,)),
                    'label': torch.tensor(self.labels[idx]),
                }

        return DataLoader(SimpleDataset(), batch_size=4, shuffle=False)

    @pytest.fixture
    def embedding_model(self, device):
        class EmbeddingClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 16)
                self.fc = nn.Linear(16, 3)

            def forward(self, x):
                x = self.embedding(x)
                x = x.mean(dim=1)
                return self.fc(x)

        return EmbeddingClassifier().to(device)

    def test_returns_dict(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        assert isinstance(result, dict)

    def test_contains_required_keys(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        required_keys = [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'auc',
            'confusion_matrix',
            'confusion_matrix_labels',
            'classification_report',
        ]
        for key in required_keys:
            assert key in result

    def test_accuracy_is_float(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        assert isinstance(result['accuracy'], float)

    def test_metrics_in_valid_range(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        assert 0.0 <= result['accuracy'] <= 1.0
        assert 0.0 <= result['precision'] <= 1.0
        assert 0.0 <= result['recall'] <= 1.0
        assert 0.0 <= result['f1_score'] <= 1.0

    def test_confusion_matrix_shape(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        cm = result['confusion_matrix']
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_confusion_matrix_labels_match_encoder(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        assert result['confusion_matrix_labels'] == ['class_a', 'class_b', 'class_c']

    def test_classification_report_is_dict(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        assert isinstance(result['classification_report'], dict)

    def test_classification_report_contains_classes(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])

        result = evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        report = result['classification_report']
        assert 'class_a' in report
        assert 'class_b' in report
        assert 'class_c' in report

    def test_model_set_to_eval_mode(
        self, evaluator, embedding_model, dataloader_with_labels, label_encoder
    ):
        y_true = np.array([0, 1, 2] * 6 + [0, 1])
        embedding_model.train()

        evaluator.evaluate_neural(
            embedding_model, dataloader_with_labels, label_encoder, y_true
        )

        assert not embedding_model.training


class TestComputeAUC:
    @pytest.fixture
    def evaluator(self, device):
        return ModelEvaluator(device=device)

    def test_multiclass_auc(self, evaluator, label_encoder):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_probs = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        auc = evaluator._compute_auc(y_true, y_probs, label_encoder)

        assert auc is not None
        assert 0.0 <= auc <= 1.0

    def test_binary_auc(self, evaluator, binary_label_encoder):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_probs = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.3, 0.7],
                [0.7, 0.3],
                [0.1, 0.9],
            ]
        )

        auc = evaluator._compute_auc(y_true, y_probs, binary_label_encoder)

        assert auc is not None
        assert 0.0 <= auc <= 1.0

    def test_perfect_auc(self, evaluator, binary_label_encoder):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_probs = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )

        auc = evaluator._compute_auc(y_true, y_probs, binary_label_encoder)

        assert auc == 1.0

    def test_handles_single_class_in_batch(self, evaluator, label_encoder):
        y_true = np.array([0, 0, 0])
        y_probs = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.9, 0.05, 0.05],
            ]
        )

        # Supress warning since behaviour is expected
        with pytest.warns(match='Only one class is present'):
            auc = evaluator._compute_auc(y_true, y_probs, label_encoder)

        assert auc is None or (isinstance(auc, float) and math.isnan(auc))


class TestLogFeatureImportance:
    @pytest.fixture
    def evaluator(self, device):
        return ModelEvaluator(device=device)

    @pytest.fixture
    def per_class_model(self):
        class MockModel:
            def get_feature_importance(self, top_n=20):
                return {
                    'type': 'per_class',
                    'data': {
                        0: {'word1': 0.5, 'word2': 0.3},
                        1: {'word3': 0.6, 'word4': 0.2},
                    },
                }

        return MockModel()

    @pytest.fixture
    def global_model(self):
        class MockModel:
            def get_feature_importance(self, top_n=20):
                return {
                    'type': 'global',
                    'data': {'feature1': 0.4, 'feature2': 0.3, 'feature3': 0.2},
                }

        return MockModel()

    @pytest.fixture
    def model_without_feature_importance(self):
        class MockModel:
            pass

        return MockModel()

    def test_handles_per_class_importance(self, evaluator, per_class_model):
        evaluator.log_feature_importance(per_class_model, 'test_job')

    def test_handles_global_importance(self, evaluator, global_model):
        evaluator.log_feature_importance(global_model, 'test_job')

    def test_handles_model_without_method(
        self, evaluator, model_without_feature_importance
    ):
        evaluator.log_feature_importance(model_without_feature_importance, 'test_job')
