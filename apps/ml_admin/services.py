import csv
import logging
import threading

import pandas as pd

from apps.ml_admin.models import DatasetRecord, DataUpload
from ml_pipeline.data_cleaning.cleaner import DataCleaningPipeline
from ml_pipeline.preprocessing.preprocessor import DataPreprocessingPipeline

logger = logging.getLogger(__name__)

MIN_WORDS_PREPROCESSED = 3

CATEGORY_MAP = {  # Define category mapping
    'normal': 0,
    'depression': 1,
    'suicidal': 2,
    'anxiety': 3,
    'bipolar': 4,
    'stress': 5,
}


def trigger_full_pipeline_in_background(data_upload_id: int):
    """
    Wrapper to run the pipeline in a separate thread.
    """
    thread = threading.Thread(target=run_full_pipeline, args=(data_upload_id,))
    thread.daemon = True
    thread.start()


def run_full_pipeline(data_upload_id: int):
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
        logger.info(f'Started pipeline for {upload.file_name}')

        # Run cleaning pipeline
        cleaner = DataCleaningPipeline()
        df, report = cleaner.clean_file(upload.file_path)

        # Run preprocessing pipeline, passing the cleaned data frame to the preprocessor
        preprocessor = DataPreprocessingPipeline()
        df, prep_report = preprocessor.preprocess_dataframe(df)
        report.update(prep_report)

        # Filter out preprocessed rows that have text shorter than 3 words
        df['word_count_proc'] = df['text_preprocessed'].str.split().str.len()
        initial_count = len(df)
        df = df[df['word_count_proc'] >= 3].copy()
        report['removed_post_preprocessing'] = initial_count - len(df)

        # Save clean, preprocessed data to database
        df['text'] = df['text_preprocessed']
        df.reset_index(drop=True, inplace=True)
        _save_dataset_records(df, upload)
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
        except:
            pass
        error = f'Pipeline failed: {str(e)}'
        logger.exception(error)
        return {'success': False, 'error': error}


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
        logger.exception(f'Cleaning pipeline failed: {e}')
        try:
            upload = DataUpload.objects.get(id=data_upload_id)
            upload.status = 'failed'
            upload.save()
        except:
            pass
        error = f'Cleaning pipeline failed: {str(e)}'
        logger.exception(error)
        return {'success': False, 'error': error}


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
        except:
            pass
        error = f'Preprocessing pipeline failed: {str(e)}'
        logger.exception(error)
        return {'success': False, 'error': error}


def _save_dataset_records(df: pd.DataFrame, upload: DataUpload):
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
                dataset_type='train',
            )
        )

    # Bulk create records
    DatasetRecord.objects.bulk_create(records, batch_size=2000)
    logger.info(f'=== Saved {len(records):,} records to database ===')


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
