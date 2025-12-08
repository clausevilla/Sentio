# Authors: Julia McCall, Claudia Sevilla, Marcus Berggren
import csv
import logging
import threading
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from django.conf import settings
from django.utils import timezone

from apps.ml_admin.models import (
    DatasetRecord,
    DataUpload,
    ModelVersion,
    Parameter,
    TrainingJob,
)
from ml_pipeline.data_cleaning.cleaner import DataCleaningPipeline
from ml_pipeline.preprocessing.preprocessor import DataPreprocessingPipeline
from ml_pipeline.storage.handler import StorageHandler
from ml_pipeline.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)

MIN_WORDS_PREPROCESSED = 3

CATEGORY_MAP = {  # Define category mapping
    'normal': 0,
    'depression': 1,
    'suicidal': 2,
    'stress': 3,
}

PIPELINE_TO_MODEL_TYPE = {
    'full': 'traditional',
    'partial': 'rnn',
    'raw': 'transformer',
}


def trigger_full_pipeline_in_background(
    data_upload_id: int,
    dataset_type: str = 'unlabeled',  # !!! PLACEHOLDER
    pipeline_type: str = 'full',
):
    """
    Wrapper to run the pipeline in a separate thread.
    """
    thread = threading.Thread(
        target=run_full_pipeline, args=(data_upload_id, dataset_type, pipeline_type)
    )
    thread.daemon = True
    thread.start()


def run_full_pipeline(
    data_upload_id: int,
    dataset_type: str = 'train',
    pipeline_type: str = 'full',
):
    # !!! PLACEHOLDERS here
    """
    Function used for interacting through the UI.

    Automatically runs the following steps:
    Cleaning pipeline -> Preprocessing pipeline -> Save clean, preprocessed data to database
    """
    try:
        upload = DataUpload.objects.get(id=data_upload_id)
        if upload.status == 'processing':
            logger.warning(f'Upload {data_upload_id} is already processing.')
            return

        upload.status = 'processing'
        upload.save()
        logger.info(f'Started pipeline for {upload.file_name} with {pipeline_type}')

        # Run cleaning pipeline
        cleaner = DataCleaningPipeline()
        df, report = cleaner.clean_file(upload.file_path)

        # Determine the correct preprocessing branch to run based on the algorithm key
        model_type = PIPELINE_TO_MODEL_TYPE.get(
            pipeline_type, 'traditional'
        )  # Fallback

        logger.info(
            f"Pipeline type '{pipeline_type}' selected. Branching to '{model_type}' preprocessing."
        )

        # Run preprocessing pipeline, passing the cleaned data frame to the preprocessor
        preprocessor = DataPreprocessingPipeline()
        df, prep_report = preprocessor.preprocess_dataframe(df, model_type=model_type)
        report.update(prep_report)

        # Filter out preprocessed rows that have text shorter than 3 words
        df['word_count_proc'] = df['text_preprocessed'].str.split().str.len()
        initial_count = len(df)
        df = df[df['word_count_proc'] >= 3].copy()
        report['removed_post_preprocessing'] = initial_count - len(df)

        # Save clean, preprocessed data to database
        df['text'] = df['text_preprocessed']
        df.reset_index(drop=True, inplace=True)
        _save_dataset_records(df, upload, dataset_type)
        _finalize_upload(upload, len(df))

        logger.info(f'Full pipeline finished successfully. Saved {len(df)} rows.')

    except DataUpload.DoesNotExist:
        error = f'Upload ID {data_upload_id} not found'
        logger.error(error)
        return {'success': False, 'error': error}

    except Exception as e:
        logger.exception(f'Pipeline failed: {e}')
        try:
            upload = DataUpload.objects.get(id=data_upload_id)
            upload.status = 'failed'
            upload.save()
        except DataUpload.DoesNotExist:
            logger.warning(
                'DataUpload %s not found during error handling', data_upload_id
            )
        return {'success': False, 'error': str(e)}


def run_cleaning_only(data_upload_id: int):
    """
    Function used in unit testing. Cleaning pipeline only.

    Runs only the cleaning pipeline and saves the cleaned data.
    Allows us to check intermediate cleaning results in the database.
    """
    try:
        upload = _get_and_lock_upload(data_upload_id)
        if not upload:
            return

        logger.info('=== Running cleaning pipeline ===')
        cleaner = DataCleaningPipeline()
        df, report = cleaner.clean_file(upload.file_path)

        df['text'] = df['text']  # Raw cleaned text, not preprocessed
        df.reset_index(drop=True, inplace=True)
        _save_dataset_records(df, upload)
        _finalize_upload(upload, len(df))

    except DataUpload.DoesNotExist:
        error = f'Upload ID {data_upload_id} not found'
        logger.error(error)
        return {'success': False, 'error': error}

    except Exception as e:
        logger.exception('Cleaning pipeline failed')
        try:
            upload = DataUpload.objects.get(id=data_upload_id)
            upload.status = 'failed'
            upload.save()
        except DataUpload.DoesNotExist:
            logger.warning(
                'DataUpload %s not found during error handling', data_upload_id
            )
        return {'success': False, 'error': str(e)}


def run_preprocessing_only(data_upload_id: int):
    """
    Function used for unit testing. Preprocessing pipeline only.

    Runs only the cleaning pipeline and saves the cleaned data.
    Allows us to check intermediate cleaning results in the database.
    Assumes that the data accessed from the database is already cleaned.
    """
    try:
        upload = _get_and_lock_upload(data_upload_id)
        if not upload:
            return

        logger.info('Running preprocessing pipeline')

        # Load from DB
        records = DatasetRecord.objects.filter(data_upload=upload)
        if not records.exists():
            raise ValueError('No records found to preprocess.')

        df = pd.DataFrame.from_records(
            records.values(
                'id',
                'text',
                'label',
                'category_id',
                'normal',
                'depression',
                'suicidal',
                'stress',
            )
        )

        # Preprocess the text
        preprocessor = DataPreprocessingPipeline()
        df, report = preprocessor.preprocess_dataframe(df)

        # Filter rows with less than the minimum number of words after preprocessing
        df['word_count_proc'] = df['text_preprocessed'].str.split().str.len()
        initial_count = len(df)
        df = df[df['word_count_proc'] >= MIN_WORDS_PREPROCESSED].copy()

        report['removed_post_preprocessing'] = initial_count - len(df)

        df['text'] = df['text_preprocessed']
        df.reset_index(drop=True, inplace=True)

        # Save data records (updated text with preprocessed text)
        _save_dataset_records(df, upload)
        _finalize_upload(upload, len(df))

    except DataUpload.DoesNotExist:
        error = f'Upload ID {data_upload_id} not found'
        logger.error(error)
        return {'success': False, 'error': error}

    except Exception as e:
        logger.exception(f'Preprocessing pipeline failed: {e}')
        try:
            upload = DataUpload.objects.get(id=data_upload_id)
            upload.status = 'failed'
            upload.save()
        except DataUpload.DoesNotExist:
            logger.warning(
                'DataUpload %s not found during error handling', data_upload_id
            )
        return {'success': False, 'error': str(e)}


def _save_dataset_records(
    df: pd.DataFrame, upload: DataUpload, dataset_type: str = 'train'
):
    """
    Handles the bulk creation of records.
    """
    # Clear old records
    DatasetRecord.objects.filter(data_upload=upload).delete()

    records = []
    for _, row in df.iterrows():
        records.append(
            DatasetRecord(
                data_upload=upload,
                text=row['text'],
                label=row['label'],
                category_id=row['category_id'],
                normal=row['normal'],
                depression=row['depression'],
                suicidal=row['suicidal'],
                stress=row['stress'],
                dataset_type=dataset_type,
            )
        )

    # Bulk create records
    DatasetRecord.objects.bulk_create(records, batch_size=2000)
    logger.info(f'=== Saved {len(records):,} records to database ===')


def _save_training_parameters(model_version, config: dict):
    """
    Save training configuration as Parameter record linked to ModelVersion.
    """

    param_data = {'model_version': model_version}

    field_mapping = {
        'max_iter': 'max_iter',
        'C': 'regularization_strength',
        'solver': 'solver',
        'n_estimators': 'n_estimators',
        'max_depth': 'max_depth',
        'min_samples_split': 'min_samples_split',
        'min_samples_leaf': 'min_samples_leaf',
        'max_features': 'rf_max_features',
        'n_jobs': 'n_jobs',
        'num_layers': 'num_layers',
        'dropout': 'dropout',
        'max_seq_len': 'max_seq_length',
        'vocab_size': 'vocab_size',
        'learning_rate': 'learning_rate',
        'batch_size': 'batch_size',
        'epochs': 'epochs',
        'embed_dim': 'embed_dim',
        'hidden_dim': 'hidden_dim',
        'd_model': 'd_model',
        'nhead': 'n_head',
        'dim_feedforward': 'dim_feedforward',
    }

    for config_key, param_field in field_mapping.items():
        if config_key in config:
            param_data[param_field] = config[config_key]

    if 'tfidf' in config:
        tfidf = config['tfidf']
        if 'ngram_range' in tfidf:
            param_data['ngram_range_min'] = tfidf['ngram_range'][0]
            param_data['ngram_range_max'] = tfidf['ngram_range'][1]
        if 'min_df' in tfidf:
            param_data['min_df'] = tfidf['min_df']
        if 'max_df' in tfidf:
            param_data['max_df'] = tfidf['max_df']
        if 'max_features' in tfidf:
            param_data['tfidf_max_features'] = tfidf['max_features']

    Parameter.objects.create(**param_data)


def _finalize_upload(upload, count):
    upload.row_count = count
    upload.status = 'completed'
    upload.is_validated = True
    upload.save()


def _get_and_lock_upload(upload_id):
    upload = DataUpload.objects.get(id=upload_id)
    if upload.status == 'processing':
        logger.warning(f'Upload {upload_id} already processing.')
        return None
    upload.status = 'processing'
    upload.save()
    return upload


def import_csv_dataset(file_path, data_upload, dataset_type='train', batch_size=5000):
    """
    Imports a CSV file into DatasetRecord model using one-hot-encoding and category_id.
    :param file_path: path to CSV file
    :param data_upload: DataUpload instance
    """
    batch = []

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:  # Map known columns to model fields
            label = row.get('status', '').strip().lower()
            category_id = CATEGORY_MAP.get(label, None)

            if category_id is None:  # Handle personality_disorder category (not in map)
                continue  # Skip the record

            one_hot_categories = {
                'normal': 0,
                'depression': 0,
                'suicidal': 0,
                'anxiety': 0,
                'bipolar': 0,
                'stress': 0,
            }  # Initialise all columns to 0

            if category_id is not None:
                one_hot_categories[label] = 1

            record = DatasetRecord(
                text=row.get('statement', ''),
                label=row.get('status', ''),
                category_id=category_id,
                dataset_type=dataset_type,  # (train, test, or unlabeled)
                data_upload=data_upload,
                **one_hot_categories,  # unpack into model fields
            )
            batch.append(record)

            # Batch insertion for performance (insert in bulks)
            if len(batch) >= batch_size:
                DatasetRecord.objects.bulk_create(batch, ignore_conflicts=True)
                batch = []

        if batch:
            DatasetRecord.objects.bulk_create(batch, ignore_conflicts=True)


def get_training_data(
    dataset_type: Literal['train', 'increment'], upload_ids: List[int] = None
) -> Tuple:
    """
    Use one method for acquiring different types of data
    """

    if dataset_type not in ('train', 'increment'):
        raise ValueError(f'Invalid dataset_type: {dataset_type}')

    # Test records are always static but train data depens on full train or incremental
    test_records = DatasetRecord.objects.filter(dataset_type='test')
    train_records = DatasetRecord.objects.filter(dataset_type=dataset_type)

    if upload_ids:
        train_records = train_records.filter(data_upload_id__in=upload_ids)

    # In case performance gain needed with a larger dataset, could switch to Pandas
    X_train = [r.text for r in train_records]
    y_train = [r.label for r in train_records]
    X_test = [r.text for r in test_records]
    y_test = [r.label for r in test_records]

    return X_train, y_train, X_test, y_test


def _create_progress_callback(job_id: int):
    """Create a callback that updates job progress in the database."""

    def callback(epoch: int, total_epochs: int, loss: float, val_accuracy: float):
        job = TrainingJob.objects.get(id=job_id)

        if job.status == 'CANCELLED':
            raise InterruptedError('Job was cancelled')

        # Add info to progress_log, current_epoch and total_epochs
        log_line = f'Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}, Val Acc: {val_accuracy:.4f}'
        job.progress_log = (
            (job.progress_log + '\n' + log_line) if job.progress_log else log_line
        )
        job.current_epoch = epoch
        job.total_epochs = total_epochs
        job.save(update_fields=['progress_log', 'current_epoch', 'total_epochs'])

    return callback


def _run_training(
    job_id: int,
    model_name: str,
    config: Dict,
    is_incremental: bool,
    upload_ids: List[int],
):
    """
    Background training task
    """

    # Count existing versions of this model type
    existing_count = ModelVersion.objects.filter(model_type=model_name).count()
    version_name = f'{model_name}_v{existing_count + 1}'

    job = TrainingJob.objects.get(id=job_id)

    try:
        storage = StorageHandler(
            model_dir=settings.MODEL_DIR,
            gcs_bucket=getattr(settings, 'GCS_BUCKET', None),
        )
        trainer = ModelTrainer(storage)

        if is_incremental:
            X_train, y_train, X_test, y_test = get_training_data(
                'increment', upload_ids
            )
        else:
            X_train, y_train, X_test, y_test = get_training_data('train', upload_ids)

        if not X_train:
            raise ValueError('No training data found')
        if not X_test:
            raise ValueError('No test data found. Split your dataset first.')

        # Set initial progress for neural network models only
        if model_name in ('lstm', 'transformer'):
            job.progress_log = f'Initializing {model_name} training...'
            job.current_epoch = 0
            job.total_epochs = config.get('epochs', 10)
            job.save(update_fields=['progress_log', 'current_epoch', 'total_epochs'])

        result = trainer.train(
            model_name=model_name,
            data=(X_train, y_train, X_test, y_test),
            config=config,
            job_id=str(job_id),
            progress_callback=_create_progress_callback(job.id),
        )

        # Check if cancelled before creating model version
        job.refresh_from_db()
        if job.status == 'CANCELLED':
            logger.info(f'Job {job_id} was cancelled, skipping model creation')
            return

        model_version = ModelVersion.objects.create(
            model_type=model_name,
            version_name=version_name,
            model_file_path=result['model_path'],
            accuracy=result['metrics']['accuracy'],
            precision=result['metrics']['precision'],
            recall=result['metrics']['recall'],
            f1_score=result['metrics']['f1_score'],
            roc_plot_base64=result['metrics'].get('roc_plot_base64'),
            confusion_matrix_base64=result['metrics'].get('confusion_matrix_base64'),
            is_active=False,
            created_by_id=job.initiated_by_id,
        )

        # Save training parameters
        _save_training_parameters(model_version, config)

        job.status = 'COMPLETED'
        job.completed_at = timezone.now()
        job.resulting_model = model_version
        job.save()

    except InterruptedError:
        # Job was cancelled - don't overwrite the CANCELLED status
        job.refresh_from_db()
        if job.status != 'CANCELLED':
            job.status = 'CANCELLED'
            job.completed_at = timezone.now()
            job.save()

    except Exception as e:
        job.refresh_from_db()
        if job.status == 'CANCELLED':
            return  # Don't overwrite cancelled status
        job.status = 'FAILED'
        job.completed_at = timezone.now()
        # Append to possible training log instead of overwriting
        error_msg = f'\n\nERROR: {str(e)}'
        job.progress_log = (
            (job.progress_log + error_msg) if job.progress_log else str(e)
        )
        job.save()


def train_full(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    initiated_by: Optional[int] = None,
    upload_ids: List[int] = None,
) -> Dict[str, Any]:
    """
    Start full training in background thread.
    """
    job = TrainingJob.objects.create(
        model_type=model_name,
        status='RUNNING',
        initiated_by_id=initiated_by,
    )

    if upload_ids:
        job.data_uploads.set(upload_ids)

    thread = threading.Thread(
        target=_run_training,
        args=(job.id, model_name, config or {}, False, upload_ids),
        daemon=True,
    )
    thread.start()

    return {'status': 'started', 'job_id': job.id}


def train_incremental(
    model_name: str,
    base_model_path: str,
    config: Optional[Dict[str, Any]] = None,
    initiated_by: Optional[int] = None,
    upload_ids: List[int] = None,
) -> Dict[str, Any]:
    """
    Start incremental training in background thread.
    """
    incremental_config = {
        'training_mode': 'incremental',
        'base_model_path': base_model_path,
        'expand_vocab': True,
        **(config or {}),  # Unpacking optional dict
    }

    job = TrainingJob.objects.create(
        model_type=model_name,
        status='RUNNING',
        initiated_by_id=initiated_by,
    )

    if upload_ids:
        job.data_uploads.set(upload_ids)

    thread = threading.Thread(
        target=_run_training,
        args=(job.id, model_name, incremental_config, True, upload_ids),
        daemon=True,
    )
    thread.start()

    return {'status': 'started', 'job_id': job.id}


def get_job_status(job_id: int) -> Dict[str, Any]:
    """
    Check training job status.
    Used for querying status every N seconds from UI during a job run.
    """

    job = TrainingJob.objects.get(id=job_id)

    result = {
        'job_id': job.id,
        'status': job.status,
        'started_at': job.started_at,
        'completed_at': job.completed_at,
    }

    if job.status == 'COMPLETED' and job.resulting_model:
        result['model'] = {
            'id': job.resulting_model.id,
            'accuracy': job.resulting_model.accuracy,
        }
    elif job.status == 'FAILED':
        result['error'] = job.progress_log

    return result


def get_active_model() -> Optional[ModelVersion]:
    """Get currently active model."""
    return ModelVersion.objects.filter(is_active=True).first()


def activate_model(model_version_id: int) -> ModelVersion:
    """Set a model version as active."""
    ModelVersion.objects.update(is_active=False)
    model = ModelVersion.objects.get(id=model_version_id)
    model.is_active = True
    model.save()
    return model


def _load_model_metadata(model_file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a model file.

    Returns dict with model_type and metrics (if available).
    """
    metadata = {'model_type': None, 'metrics': {}}

    if model_file_path.endswith('.pt'):
        storage = StorageHandler(model_dir=settings.MODEL_DIR)
        checkpoint = storage.load_neural_model(model_file_path)
        metadata['model_type'] = checkpoint.get('model_type')
        metadata['metrics'] = checkpoint.get('metrics') or {}

    return metadata


def register_model(
    model_file_path: str, version_name: str = None, set_active: bool = False
) -> ModelVersion:
    """
    Register a trained model into the database.

    Reads model_type, metrics, and config directly from the model file.
    """
    metadata = _load_model_metadata(model_file_path)
    model_type = metadata['model_type']
    metrics = metadata['metrics']
    config = metadata.get('config', {})

    if model_type is None:
        raise ValueError(f'Could not determine model_type from {model_file_path}')

    if version_name is None:
        existing_count = ModelVersion.objects.filter(model_type=model_type).count()
        version_name = f'{model_type}_v{existing_count + 1}'

    if set_active:
        ModelVersion.objects.update(is_active=False)

    model_version = ModelVersion.objects.create(
        model_type=model_type,
        version_name=version_name,
        model_file_path=model_file_path,
        accuracy=metrics.get('accuracy'),
        precision=metrics.get('precision'),
        recall=metrics.get('recall'),
        f1_score=metrics.get('f1_score'),
        roc_plot_base64=metrics.get('roc_plot_base64'),
        confusion_matrix_base64=metrics.get('confusion_matrix_base64'),
        is_active=set_active,
    )

    # Save parameters from config
    if config:
        _save_training_parameters(model_version, config)

    logger.info(f'Registered model: {version_name}')

    return model_version
