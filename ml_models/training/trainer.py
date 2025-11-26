from typing import Any, Dict, Tuple

import torch

from ml_models.models.logistic_regression import LogisticRegressionModel
from ml_models.storage.handler import StorageHandler


class ModelTrainer:
    """Handles training for all model types"""

    SKLEARN_MODELS = {
        'logistic_regression': LogisticRegressionModel,
    }

    def __init__(self, storage_handler: StorageHandler):
        self.storage = storage_handler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model_name: str, data, config: Dict[str, Any], job_id: str):
        print(f'[{job_id}] Training {model_name}...')

        if model_name in self.SKLEARN_MODELS:
            return self._train_sklearn_model(model_name, data, config, job_id)

        else:
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

        print(f'[{job_id}] Training on {len(X_train)} samples...')
        model.fit(X_train, y_train)

        print(f'[{job_id}] Evaluating on {len(X_test)} samples...')
        metrics = model.evaluate(X_test, y_test)

        self._print_feature_importance(model, job_id)

        model_path = self.storage.save_sklearn_model(
            model.pipeline, f'{model_name}_{job_id}.pkl'
        )

        return {
            'status': 'success',
            'job_id': job_id,
            'model_path': model_path,
            'metrics': metrics,
            'model_type': model_name,
        }

    def _print_feature_importance(self, model, job_id: str):
        """Extract and print feature importance if available"""
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance(top_n=20)
            print(f'\n[{job_id}] Top features:')
            for feat, imp in list(importance.items())[:10]:
                print(f'  {feat}: {imp:.4f}')
        elif hasattr(model, 'get_top_features'):
            top_features = model.get_top_features(top_n=10)
            print(f'\n[{job_id}] Top features per class:')
            for class_id, features in top_features.items():
                print(f'\nClass {class_id}:')
                for feat, coef in list(features.items())[:5]:
                    print(f'  {feat}: {coef:.4f}')
