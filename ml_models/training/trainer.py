# Author: Marcus Berggren

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

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

        Follows standard PyTorch training loop pattern with separate
        train_loop and test_loop functions.
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

        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_accuracy = 0.0
        best_model_state = None

        for epoch in range(epochs):
            logger.info(f'\nEpoch {epoch + 1}/{epochs}')
            logger.info('-' * 40)

            train_loss = self._train_loop(train_loader, model, loss_fn, optimizer)
            test_loss, accuracy = self._test_loop(test_loader, model, loss_fn)

            logger.info(f'Train loss: {train_loss:.6f}')
            logger.info(f'Test loss:  {test_loss:.6f}, Accuracy: {accuracy * 100:.1f}%')

            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                logger.info('  -> New best model saved!')

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation
        metrics = self._compute_metrics(
            model, test_loader, label_encoder, y_test_encoded
        )

        # Save model
        model_path = self.storage.save_neural_model(
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            config=config,
            filename=f'{model_name}_{job_id}.pt',
        )

        logger.info('\nTraining complete!')
        logger.info(f'Best accuracy: {best_accuracy * 100:.1f}%')
        logger.info(f'Model saved to: {model_path}')

        return {
            'status': 'success',
            'job_id': job_id,
            'model_path': model_path,
            'metrics': metrics,
            'model_type': model_name,
        }

    def _train_loop(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Training loop for one epoch.

        Based on PyTorch tutorial:
        https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_loss = 0.0

        # Set the model to training mode
        # Important for batch normalization and dropout layers
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            pred = model(input_ids)
            loss = loss_fn(pred, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # logger.info progress every 100 batches
            if batch_idx % 100 == 0:
                loss_val = loss.item()
                current = batch_idx * len(input_ids)
                logger.info(f'  loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]')

        return total_loss / num_batches

    def _test_loop(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
    ) -> Tuple[float, float]:
        """
        Test/validation loop.

        Based on PyTorch tutorial:
        https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0.0
        correct = 0

        # Set the model to evaluation mode
        # Important for batch normalization and dropout layers
        model.eval()

        # Evaluating with torch.no_grad() ensures no gradients are computed
        # Reduces memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                pred = model(input_ids)
                test_loss += loss_fn(pred, labels).item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        test_loss /= num_batches
        accuracy = correct / size

        return test_loss, accuracy
