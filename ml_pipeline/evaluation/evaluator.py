# Author: Marcus Berggren
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluation metrics for neural network models.

    Provides two evaluation modes:
        - quick_accuracy(): Fast accuracy check for training loop
        - evaluate_neural(): Full metrics for final evaluation

    Output format matches sklearn model evaluate() methods, allowing
    consistent metric handling regardless of model type.

    Reference:
        https://scikit-learn.org/stable/modules/model_evaluation.html
    """

    def __init__(self, device: torch.device = None):
        """
        Args:
            device: PyTorch device for inference. Defaults to CUDA if available.
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def evaluate_neural(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        label_encoder: LabelEncoder,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Full evaluation with all metrics for neural models.

        Computes accuracy, precision, recall, F1, confusion matrix,
        per-class classification report, and ROC-AUC score.

        Args:
            model: Trained PyTorch model
            data_loader: DataLoader for test/validation data
            label_encoder: Fitted sklearn LabelEncoder for class names
            y_true: Ground truth labels (encoded as integers)

        Returns:
            Dict containing:
                - accuracy: Overall accuracy (0-1)
                - precision: Weighted precision
                - recall: Weighted recall
                - f1_score: Weighted F1 score
                - auc: ROC-AUC score (weighted, one-vs-rest)
                - confusion_matrix: List of lists
                - confusion_matrix_labels: Class names
                - classification_report: Per-class metrics dict
        """
        model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                logits = model(input_ids)
                probs = torch.softmax(logits, dim=-1)

                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        class_names = list(label_encoder.classes_)
        all_labels = list(range(len(class_names)))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average='weighted', zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        auc = self._compute_auc(y_true, y_probs, label_encoder)

        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': class_names,
            'classification_report': classification_report(
                y_true,
                y_pred,
                labels=all_labels,
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            ),
        }

    def _compute_auc(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        label_encoder: LabelEncoder,
    ) -> Optional[float]:
        """
        Compute ROC-AUC score for multi-class classification.

        Uses one-vs-rest (OvR) strategy with weighted averaging.

        Args:
            y_true: Ground truth labels (encoded)
            y_probs: Predicted probabilities, shape (n_samples, n_classes)
            label_encoder: Fitted LabelEncoder

        Returns:
            Weighted AUC score, or None if computation fails
        """
        try:
            n_classes = len(label_encoder.classes_)
            if n_classes == 2:
                return float(roc_auc_score(y_true, y_probs[:, 1]))
            else:
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                return float(
                    roc_auc_score(
                        y_true_bin,
                        y_probs,
                        average='weighted',
                        multi_class='ovr',
                    )
                )
        except ValueError as e:
            logger.warning(f'Could not compute AUC: {e}')
            return None

    def quick_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        """
        Fast accuracy check for training loop.

        Skips expensive metrics (confusion matrix, per-class stats) since
        we only need a single number to track progress and checkpoint.

        Args:
            model: PyTorch model
            data_loader: Validation DataLoader

        Returns:
            Accuracy as float between 0 and 1
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = model(input_ids)
                predictions = logits.argmax(dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def log_feature_importance(self, model, job_id: str) -> None:
        """
        Log feature importance for sklearn models.

        Supports two formats:
            - 'per_class': Logistic regression coefficients per class
            - 'global': Random forest feature importances

        Args:
            model: Sklearn model with get_feature_importance() method
            job_id: Job identifier for logging
        """
        if not hasattr(model, 'get_feature_importance'):
            return

        result = model.get_feature_importance(top_n=20)

        if result['type'] == 'per_class':
            logger.info('Top features per class:', extra={'job_id': job_id})
            for class_id, features in result['data'].items():
                logger.debug(f'Class {class_id}:', extra={'job_id': job_id})
                for feat, coef in list(features.items())[:5]:
                    logger.debug(f'  {feat}: {coef:.4f}', extra={'job_id': job_id})
        elif result['type'] == 'global':
            logger.info('Top 10 features:', extra={'job_id': job_id})
            for feat, imp in list(result['data'].items())[:10]:
                logger.debug(f'  {feat}: {imp:.4f}', extra={'job_id': job_id})
