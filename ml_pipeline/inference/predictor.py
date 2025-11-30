# Author: Marcus Berggren
import logging
from typing import Any, Dict, List

import torch

from ml_pipeline.models import (
    LSTMModel,
    TransformerModel,
)
from ml_pipeline.storage.handler import StorageHandler

logger = logging.getLogger(__name__)


class Predictor:
    """
    Handles inference for trained sklearn and PyTorch models.

    Loads a saved model and provides prediction method that returns
    predicted label and confidence scores.

    Example:
        predictor = Predictor(storage_handler)
        predictor.load('transformer_v1.pt')

        result = predictor.predict("I feel anxious and stressed")
        print(result['label'])         # 'Stress'
        print(result['confidence'])    # 0.87
        print(result['probabilities']) # {'Depression': 0.05, 'Normal': 0.03, ...}
    """

    PYTORCH_MODELS = {
        'lstm': LSTMModel,
        'transformer': TransformerModel,
    }

    def __init__(self, storage_handler: StorageHandler):
        """
        Args:
            storage_handler: StorageHandler instance for loading models
        """
        self.storage = storage_handler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None
        self.model_type = None

    def load(self, model_path: str) -> None:
        """
        Load a trained model for inference.

        Automatically detects model type from file extension:
            - .pt: PyTorch model (LSTM, Transformer)
            - .joblib: sklearn model (Logistic Regression, Random Forest)

        Args:
            model_path: Path to saved model file

        Raises:
            ValueError: If model file extension is not recognized
        """
        if model_path.endswith('.pt'):
            self._load_neural_model(model_path)
        elif model_path.endswith('.joblib'):
            self._load_sklearn_model(model_path)
        else:
            raise ValueError(
                f'Unknown model format: {model_path}. '
                f'Expected .pt (PyTorch) or .joblib (Sklearn)'
            )

        logger.info(f'Loaded model from {model_path}')

    def _load_neural_model(self, model_path: str) -> None:
        """Load PyTorch model with tokenizer and label encoder."""
        checkpoint = self.storage.load_neural_model(model_path)

        self.tokenizer = checkpoint['tokenizer']
        self.label_encoder = checkpoint['label_encoder']
        self.config = checkpoint['config']

        model_name = self.config.get('model_name', None)
        if model_name is None:
            if 'transformer' in model_path.lower():
                model_name = 'transformer'
            elif 'lstm' in model_path.lower():
                model_name = 'lstm'
            else:
                model_name = 'transformer'

        model_class = self.PYTORCH_MODELS[model_name]
        model_wrapper = model_class(self.config)
        model_wrapper.build_model(
            vocab_size=self.tokenizer.vocab_size_actual(),
            num_classes=len(self.label_encoder.classes_),
        )
        model_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        model_wrapper.model.to(self.device)
        model_wrapper.model.eval()

        self.model = model_wrapper.model
        self.model_type = 'neural'

    def _load_sklearn_model(self, model_path: str) -> None:
        """Load sklearn pipeline model."""
        self.model = self.storage.load_sklearn_model(model_path)
        self.model_type = 'sklearn'

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict mental health category for a single text.

        Args:
            text: Input text to classify

        Returns:
            Dict containing:
                - label: Predicted class name (str)
                - confidence: Probability of predicted class (float)
                - probabilities: Dict of all class probabilities

        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError('No model loaded. Call load() first.')

        if self.model_type == 'neural':
            return self._predict_neural(text)
        else:
            return self._predict_sklearn(text)

    def _predict_neural(self, text: str) -> Dict[str, Any]:
        """Prediction for neural model."""
        max_seq_len = self.config.get('max_seq_len', 256)
        tokens = self.tokenizer.encode(text, max_len=max_seq_len)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        predicted_idx = probs.argmax()
        class_names = list(self.label_encoder.classes_)

        return {
            'label': class_names[predicted_idx],
            'confidence': float(probs[predicted_idx]),
            'probabilities': {
                name: float(probs[i]) for i, name in enumerate(class_names)
            },
        }

    def _predict_sklearn(self, text: str) -> Dict[str, Any]:
        """Prediction for sklearn model."""
        probs = self.model.predict_proba([text])[0]
        predicted_idx = probs.argmax()
        class_names = list(self.model.classes_)

        return {
            'label': class_names[predicted_idx],
            'confidence': float(probs[predicted_idx]),
            'probabilities': {
                name: float(probs[i]) for i, name in enumerate(class_names)
            },
        }

    def get_class_names(self) -> List[str]:
        """
        Get list of class names the model can predict.

        Returns:
            List of class names

        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError('No model loaded. Call load() first.')

        if self.model_type == 'neural':
            return list(self.label_encoder.classes_)
        else:
            return list(self.model.classes_)
