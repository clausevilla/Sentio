# Author: Marcus Berggren
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StorageHandler:
    """
    Handles saving and loading models.

    Supports local filesystem and Google Cloud Storage.

    File formats:
        - sklearn models: .joblib (compressed with joblib)
        - PyTorch models: .pt (torch.save)
    """

    def __init__(
        self,
        model_dir: str = './ml-models',
        gcs_bucket: Optional[str] = None,
        use_gcs: bool = False,
    ):
        """
        Args:
            model_dir: Local directory for saving models
            gcs_bucket: Optional GCS bucket name for cloud storage
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.gcs_bucket = gcs_bucket
        self.use_gcs = use_gcs
        self.gcs_client = None

        if gcs_bucket:
            self._init_gcs()

    def _init_gcs(self):
        """Initialize Google Cloud Storage client."""
        try:
            from google.cloud import storage

            self.gcs_client = storage.Client()
            logger.info(f'Connected to GCS bucket: {self.gcs_bucket}')
        except ImportError:
            logger.warning(
                'google-cloud-storage not installed. '
                'Run: pip install google-cloud-storage'
            )
        except Exception as e:
            logger.warning(f'Failed to connect to GCS: {e}')

    def save_sklearn_model(
        self,
        pipeline,
        filename: str,
        compress: int = 3,
    ) -> str:
        """
        Save sklearn pipeline using joblib with compression.

        Args:
            pipeline: Trained sklearn pipeline
            filename: Name for the saved file (should end with .joblib)
            compress: Compression level 0-9 (default 3, good balance)

        Returns:
            Path to saved model
        """
        if not filename.endswith('.joblib'):
            filename = filename.replace('.pkl', '.joblib')
            if not filename.endswith('.joblib'):
                filename += '.joblib'

        if self.use_gcs and self.gcs_client:
            return self._save_sklearn_to_gcs(pipeline, filename, compress)

        local_path = self.model_dir / filename
        joblib.dump(pipeline, local_path, compress=compress)

        logger.info(f'Saved sklearn model to {local_path}')
        return str(local_path)

    def _save_sklearn_to_gcs(self, pipeline, filename: str, compress: int) -> str:
        """Save sklearn pipeline directly to GCS."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = Path(f.name)

        joblib.dump(pipeline, temp_path, compress=compress)

        gcs_path = f'models/{filename}'
        self._upload_to_gcs(temp_path, gcs_path)
        temp_path.unlink()

        return f'gs://{self.gcs_bucket}/{gcs_path}'

    def load_sklearn_model(self, path: str):
        """
        Load sklearn pipeline.

        Args:
            path: Local path or GCS path (gs://bucket/path)
                  Supports both .joblib and .pkl files
        """
        if path.startswith('gs://'):
            return self._load_sklearn_from_gcs(path)

        return joblib.load(path)

    def _load_sklearn_from_gcs(self, gcs_path: str):
        """Load sklearn pipeline from GCS."""
        import tempfile

        gcs_path_clean = gcs_path.replace(f'gs://{self.gcs_bucket}/', '')

        suffix = '.joblib' if '.joblib' in gcs_path else '.pkl'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            temp_path = Path(f.name)

        self._download_from_gcs(gcs_path_clean, temp_path)
        pipeline = joblib.load(temp_path)
        temp_path.unlink()

        return pipeline

    def save_neural_model(
        self,
        model: nn.Module,
        tokenizer,
        label_encoder,
        config: Dict[str, Any],
        filename: str,
        model_type: str,
        metrics: Dict[str, Any] = None,
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
            'model_type': model_type,
            'metrics': metrics,
        }

        if self.use_gcs and self.gcs_client:
            return self._save_neural_to_gcs(save_dict, filename)

        local_path = self.model_dir / filename
        torch.save(save_dict, local_path)
        logger.info(f'Saved neural model to {local_path}')

        return str(local_path)

    def _save_neural_to_gcs(self, save_dict: Dict, filename: str) -> str:
        """Save neural model directly to GCS."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        torch.save(save_dict, temp_path)

        gcs_path = f'models/{filename}'
        self._upload_to_gcs(temp_path, gcs_path)
        temp_path.unlink()

        return f'gs://{self.gcs_bucket}/{gcs_path}'

    def load_neural_model(self, path: str) -> Dict[str, Any]:
        """
        Load PyTorch model checkpoint.

        Args:
            path: Local path or GCS path (gs://bucket/path)

        Returns:
            Dict with model_state_dict, tokenizer, label_encoder, config
        """
        if path.startswith('gs://'):
            return self._load_neural_from_gcs(path)

        return torch.load(path, map_location='cpu', weights_only=False)

    def _load_neural_from_gcs(self, gcs_path: str) -> Dict[str, Any]:
        """Load neural model from GCS."""
        import tempfile

        gcs_path_clean = gcs_path.replace(f'gs://{self.gcs_bucket}/', '')

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        self._download_from_gcs(gcs_path_clean, temp_path)
        checkpoint = torch.load(temp_path, map_location='cpu', weights_only=False)
        temp_path.unlink()

        return checkpoint

    def _upload_to_gcs(self, local_path: Path, gcs_path: str) -> None:
        """Upload a local file to GCS."""
        try:
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            logger.info(f'Uploaded to gs://{self.gcs_bucket}/{gcs_path}')
        except Exception as e:
            logger.error(f'Failed to upload to GCS: {e}')
            raise

    def _download_from_gcs(self, gcs_path: str, local_path: Path) -> None:
        """Download a file from GCS to local path."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(str(local_path))
            logger.info(f'Downloaded gs://{self.gcs_bucket}/{gcs_path} to {local_path}')
        except Exception as e:
            logger.error(f'Failed to download from GCS: {e}')
            raise

    def list_models(self, from_gcs: bool = False) -> List[str]:
        """List available models from local or GCS."""
        if from_gcs and self.gcs_client:
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            blobs = bucket.list_blobs(prefix='models/')
            return [blob.name.replace('models/', '') for blob in blobs]

        return [f.name for f in self.model_dir.iterdir() if f.is_file()]
