import logging
from typing import Any, Dict, Tuple

import torch

from ml_models.models.logistic_regression import LogisticRegressionModel
from ml_models.models.lstm import LSTMModel
from ml_models.models.random_forest import RandomForestModel
from ml_models.models.transformer import TransformerModel
from ml_models.storage.handler import StorageHandler

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training for all model types"""

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
        Args:
            model_name: Name of the model to train
            data: Tuple of (X_train, y_train, X_test, y_test)
            config: Model-specific configuration overrides
            job_id: Unique job identifier for logging
        """

        logger.info(f'Starting training for {model_name}', extra={'job_id': job_id})

        if not isinstance(data, tuple):
            logger.error(
                f'Invalid data type: {type(data).__name__}', extra={'job_id': job_id}
            )
            raise TypeError(f'data must be a tuple, got {type(data).__name__}')

        if len(data) != 4:
            logger.error(
                f'Invalid data tuple length: {len(data)}', extra={'job_id': job_id}
            )
            raise ValueError(
                f'Expected (X_train, y_train, X_test, y_test), got tuple of length {len(data)}'
            )

        if model_name in self.SKLEARN_MODELS:
            return self._train_sklearn_model(model_name, data, config, job_id)
        elif model_name in self.PYTORCH_MODELS:
            return self._train_neural_model(model_name, data, config, job_id)
        else:
            logger.error(f'Unknown model: {model_name}', extra={'job_id': job_id})
            raise ValueError(f'Unknown model: {model_name}')

    def _train_sklearn_model(
        self,
        model_name: str,
        data: Tuple,
        config: Dict[str, Any],
        job_id: str,
    ) -> Dict[str, Any]:
        """Train any sklearn-based model"""

        X_train, y_train, X_test, y_test = data

        model_class = self.SKLEARN_MODELS[model_name]
        model = model_class(config if config else None)

        logger.info(f'Training on {len(X_train)} samples', extra={'job_id': job_id})
        model.fit(X_train, y_train)

        logger.info(f'Evaluating on {len(X_test)} samples', extra={'job_id': job_id})
        metrics = model.evaluate(X_test, y_test)

        self._log_feature_importance(model, job_id)

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
    ):
        logger.info(
            'Neural model training not yet implemented', extra={'job_id': job_id}
        )
        raise NotImplementedError(
            f'Neural model training for {model_name} not yet implemented'
        )

    def _log_feature_importance(self, model, job_id: str):
        """Extract and log feature importance if available"""
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance(top_n=20)
            logger.info('Top 10 features:', extra={'job_id': job_id})
            for feat, imp in list(importance.items())[:10]:
                logger.debug(f'  {feat}: {imp:.4f}', extra={'job_id': job_id})
        elif hasattr(model, 'get_top_features'):
            top_features = model.get_top_features(top_n=10)
            logger.info('Top features per class:', extra={'job_id': job_id})
            for class_id, features in top_features.items():
                logger.debug(f'Class {class_id}:', extra={'job_id': job_id})
                for feat, coef in list(features.items())[:5]:
                    logger.debug(f'  {feat}: {coef:.4f}', extra={'job_id': job_id})
