# Author: Marcus Berggren
import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from ml_models.evaluation.evaluator import ModelEvaluator
from ml_models.models import (
    LogisticRegressionModel,
    LSTMModel,
    RandomForestModel,
    TransformerModel,
)
from ml_models.storage.handler import StorageHandler
from ml_models.training import TextDataset, WordTokenizer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training for Sklearn and Pytorch models.

    Training loop follows PyTorch tutorial patterns:
    https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    """

    SKLEARN_MODELS = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestModel,
    }
    PYTORCH_MODELS = {
        'lstm': LSTMModel,
        'transformer': TransformerModel,
    }

    def __init__(self, storage_handler: StorageHandler):
        self.storage = storage_handler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(self.device)
        logger.info(f'Initialized ModelTrainer with device: {self.device}')

    def train(
        self, model_name: str, data: Tuple, config: Dict[str, Any], job_id: str
    ) -> Dict[str, Any]:
        """
        Train a model and return results with metrics.

        Args:
            model_name: Key from SKLEARN_MODELS or PYTORCH_MODELS
            data: Tuple of (X_train, y_train, X_test, y_test)
            config: Model-specific configuration overrides
            job_id: Unique identifier for logging and file naming

        Returns:
            Dict with status, job_id, model_path, metrics, model_type
        """
        logger.info(f'Starting training for {model_name}', extra={'job_id': job_id})

        if not isinstance(data, tuple):
            raise TypeError(f'data must be a tuple, got {type(data).__name__}')

        if len(data) != 4:
            raise ValueError(
                f'Expected (X_train, y_train, X_test, y_test), got tuple of length {len(data)}'
            )

        if model_name in self.SKLEARN_MODELS:
            return self._train_sklearn_model(model_name, data, config, job_id)
        elif model_name in self.PYTORCH_MODELS:
            return self._train_neural_model(model_name, data, config, job_id)
        else:
            raise ValueError(f'Unknown model: {model_name}')

    def _train_sklearn_model(
        self,
        model_name: str,
        data: Tuple,
        config: Dict[str, Any],
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Train sklearn-based models (logistic regression, random forest).

        These models handle their own TF-IDF vectorization internally
        via their pipeline, so we pass raw text directly.
        """
        X_train, y_train, X_test, y_test = data

        model_class = self.SKLEARN_MODELS[model_name]
        model = model_class(config if config else None)

        logger.info(f'Training on {len(X_train)} samples', extra={'job_id': job_id})
        model.fit(X_train, y_train)

        logger.info(f'Evaluating on {len(X_test)} samples', extra={'job_id': job_id})
        metrics = model.evaluate(X_test, y_test)

        self.evaluator.log_feature_importance(model, job_id)

        model_path = self.storage.save_sklearn_model(
            model.pipeline, f'{model_name}_{job_id}.pkl'
        )

        logger.info(
            f'Training complete. Accuracy: {metrics["accuracy"]:.4f}',
            extra={'job_id': job_id, 'model_path': model_path},
        )

        return {
            'status': 'success',
            'job_id': job_id,
            'model_path': model_path,
            'metrics': metrics,
            'model_type': model_name,
        }

    def _train_neural_model(
        self,
        model_name: str,
        data: Tuple,
        config: Dict[str, Any],
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Train PyTorch-based models (LSTM, Transformer).

        Pipeline:
            1. Fit tokenizer on training text
            2. Encode labels to integers
            3. Create DataLoaders for batching
            4. Build model with vocab size and class count
            5. Train with AdamW + OneCycleLR
            6. Checkpoint best model by validation accuracy
            7. Evaluate and save
        """
        X_train, y_train, X_test, y_test = data

        config = config or {}
        batch_size = config.get('batch_size', 32)
        epochs = config.get('epochs', 10)
        max_seq_len = config.get('max_seq_len', 512)
        learning_rate = config.get('learning_rate', 1e-4)

        if isinstance(X_train, np.ndarray):
            X_train = X_train.tolist()
            X_test = X_test.tolist()

        logger.info('Fitting tokenizer...', extra={'job_id': job_id})
        tokenizer = WordTokenizer(vocab_size=config.get('vocab_size', 50000))
        tokenizer.fit(X_train)

        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        logger.info(
            f'Vocab size: {tokenizer.vocab_size_actual()}, '
            f'Classes: {len(label_encoder.classes_)}',
            extra={'job_id': job_id},
        )

        train_dataset = TextDataset(X_train, y_train_encoded, tokenizer, max_seq_len)
        test_dataset = TextDataset(X_test, y_test_encoded, tokenizer, max_seq_len)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        model_class = self.PYTORCH_MODELS[model_name]
        model_wrapper = model_class(config)
        model_wrapper.build_model(
            vocab_size=tokenizer.vocab_size_actual(),
            num_classes=len(label_encoder.classes_),
        )
        model = model_wrapper.model

        logger.info(
            f'Model parameters: {model_wrapper.get_num_parameters():,}',
            extra={'job_id': job_id},
        )

        # Class weighting for imbalanced dataset
        # Inverse frequency: rare classes get higher weight
        # Normalization ensures weights sum to num_classes (stable loss scale)
        class_counts = np.bincount(y_train_encoded)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Have tried SGD (stochastic gradient descent), Adam and AdamW has given best results
        # AdamW: Adam with decoupled weight decay
        # Better regularization than L2 penalty in original Adam
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # OneCycleLR: Learning rate starts low, peaks at max_lr, then anneals
        # Often faster convergence than constant LR or step decay
        total_steps = len(train_loader) * epochs
        scheduler = OneCycleLR(
            optimizer, max_lr=learning_rate * 10, total_steps=total_steps
        )

        best_accuracy = 0.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids)
                loss = criterion(logits, labels)
                loss.backward()

                # Gradient clipping prevents exploding gradients
                # Common in transformers and RNNs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_accuracy = self.evaluator.quick_accuracy(model, test_loader)

            logger.info(
                f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, '
                f'Val Accuracy: {val_accuracy:.4f}',
                extra={'job_id': job_id},
            )

            # Checkpoint best model (by validation accuracy, not loss)
            # Prevents overfitting: we keep the most generalizable state
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best checkpoint for final evaluation and saving
        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = self.evaluator.evaluate_neural(
            model, test_loader, label_encoder, y_test_encoded
        )

        model_path = self.storage.save_neural_model(
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            config=config,
            filename=f'{model_name}_{job_id}.pt',
        )

        logger.info(
            f'Training complete. Accuracy: {metrics["accuracy"]:.4f}',
            extra={'job_id': job_id, 'model_path': model_path},
        )

        return {
            'status': 'success',
            'job_id': job_id,
            'model_path': model_path,
            'metrics': metrics,
            'model_type': model_name,
        }
