from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch

from ml_models.models.logistic_regression import LogisticRegressionModel
from ml_models.storage.handler import StorageHandler


class ModelTrainer:
    """Handles training for all model types"""

    def __init__(self, storage_handler: StorageHandler):
        self.storage = storage_handler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model_name: str, data, config: Dict[str, Any], job_id: str):
        print(f'[{job_id}] Training {model_name}...')

        if model_name in ['logistic_regression']:
            return self._train_sklearn_model(model_name, data, config, job_id)
        else:
            raise ValueError(f'Unknown model: {model_name}')

    def _train_sklearn_model(
        self,
        model_name: str,
        data: Union[pd.DataFrame, Tuple],
        config: Dict[str, Any],
        job_id: str,
    ) -> Dict[str, Any]:
        """Train any sklearn-based model"""

        X_train, y_train, X_test, y_test = self._parse_data(data)

        # Dispatch to correct model class
        if model_name == 'logistic_regression':
            model = LogisticRegressionModel(config)

        print(f'[{job_id}] Training on {len(X_train)} samples...')
        model.fit(X_train, y_train)

        print(f'[{job_id}] Evaluating on {len(X_test)} samples...')
        metrics = model.evaluate(X_test, y_test)

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
