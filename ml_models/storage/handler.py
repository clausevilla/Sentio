# Author: Marcus Berggren
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StorageHandler:
    """
    Handles saving and loading models.
    """

    def __init__(
        self,
        model_dir: str = './models',
        gcs_bucket: Optional[str] = None,
    ):
        """
        Args:
            model_dir: Local directory for saving models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_sklearn_model(
        self,
        pipeline,
        filename: str,
        to_gcs: bool = False,
    ) -> str:
        """
        Save sklearn pipeline.

        Args:
            pipeline: Trained sklearn pipeline
            filename: Name for the saved file
        """

        local_path = self.model_dir / filename

        with open(local_path, 'wb') as f:
            pickle.dump(pipeline, f)

        logger.info(f'Saved sklearn model to {local_path}')
        return str(local_path)

    def load_sklearn_model(self, path: str):
        """
        Load sklearn pipeline.

        Args:
            path: Local path
        """

        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_neural_model(
        self,
        model: nn.Module,
        tokenizer,
        label_encoder,
        config: Dict[str, Any],
        filename: str,
    ) -> str:
        """
        Save PyTorch model with tokenizer and label encoder.

        Saves everything needed for inference:
            - model_state_dict: learned weights
            - tokenizer: converts text to token IDs
            - label_encoder: converts predictions to label names
            - config: architecture config to rebuild model
        """
        save_dict = {
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'config': config,
        }

        local_path = self.model_dir / filename
        torch.save(save_dict, local_path)
        logger.info(f'Saved neural model to {local_path}')

        return str(local_path)

    def load_neural_model(self, path: str) -> Dict[str, Any]:
        """
        Load PyTorch model checkpoint.

        Args:
            path: Local path

        Returns:
            Dict with model_state_dict, tokenizer, label_encoder, config
        """

        return torch.load(path, map_location='cpu', weights_only=False)
