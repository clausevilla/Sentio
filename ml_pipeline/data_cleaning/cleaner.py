import logging
from typing import Dict, Tuple

import pandas as pd

from apps.ml_admin.models import DatasetRecord, DataUpload
from ml_pipeline.preprocessing.preprocessor import preprocess_cleaned_data

logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    """
    Cleans raw CSV data for mental health text classification.

    Output columns: text, label, category_id, Normal, Depression, Suicidal, Stress
    """

    def __init__(self):
        self.min_word_count = 3
        self.min_text_length = 10
        self.max_text_length = 5000
        self.valid_labels = ['Normal', 'Depression', 'Suicidal', 'Stress']

        self.report = {
            'original_count': 0,
            'removed_missing_labels': 0,
            'removed_empty_text': 0,
            'removed_short_text': 0,
            'removed_invalid_labels': 0,
            'trimmed_long_text': 0,
            'removed_duplicates': 0,
            'final_count': 0,
        }

        self.encoding_fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€¦': '...',
            'Â': '',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã¢': 'â',
            'Ã´': 'ô',
            'Ã«': 'ë',
            'Ã¯': 'ï',
            'Ã¼': 'ü',
            'Ã±': 'ñ',
            'Ã§': 'ç',
            '\x00': '',
            '\ufeff': '',
            '\u2018': "'",
            '\u2019': "'",
            '\u201a': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u201e': '"',
            '\u2013': '-',
            '\u2014': '--',
            '\u2026': '...',
            '\u00a0': ' ',
        }

    def clean_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean data from a CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (cleaned_dataframe, report_dict)
        """
        logger.info('=== Starting Data Cleaning Pipeline ===')
        logger.info(f'File: {file_path}')

        try:
            # Load data
            df = pd.read_csv(file_path)
            self.report['original_count'] = len(df)
            logger.info(f'Loaded {len(df):,} rows')

            # Execute all cleaning steps
            df = self._remove_missing_labels(df)
            df = self._remove_empty_text(df)
            df = self._remove_short_text(df)
            df = self._trim_long_text(df)
            df = self._combine_anxiety_stress(df)
            df = self._remove_invalid_labels(df)
            df = self._fix_encoding(df)
            df = self._remove_duplicates(df)
            df = df.reset_index(drop=True)
            df = self._create_label_encodings(df)

            self.report['final_count'] = len(df)
            logger.info(f'=== Cleaning Complete: {len(df):,} rows ===')

            return df, self.report

        except Exception as e:
            logger.error(f'Error in cleaning pipeline: {str(e)}', exc_info=True)
            raise

    def _remove_missing_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df = df.dropna(subset=['label'])
        removed = initial - len(df)
        self.report['removed_missing_labels'] = removed
        logger.info(f'Removed {removed:,} rows with missing labels')
        return df

    def _remove_empty_text(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].astype(str)
        df = df[df['text'].str.strip() != '']
        removed = initial - len(df)
        self.report['removed_empty_text'] = removed
        logger.info(f'Removed {removed:,} rows with empty text')
        return df

    def _remove_short_text(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df['word_count'] = df['text'].str.split().str.len()
        df = df[
            (df['text'].str.len() >= self.min_text_length)
            & (df['word_count'] >= self.min_word_count)
        ]
        df = df.drop(columns=['word_count'])
        removed = initial - len(df)
        self.report['removed_short_text'] = removed

        logger.info(
            f'Removed {removed:,} rows with "text" column having < {self.min_text_length} characters or < {self.min_word_count} words'
        )
        return df

    def _trim_long_text(self, df: pd.DataFrame) -> pd.DataFrame:
        long_count = (df['text'].str.len() > self.max_text_length).sum()
        df['text'] = df['text'].str[: self.max_text_length]
        self.report['trimmed_long_text'] = long_count
        logger.info(f'Trimmed {long_count:,} text to {self.max_text_length} chars')
        return df

    def _combine_anxiety_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        before_anxiety = len(df[df['label'] == 'Anxiety'])
        before_stress = len(df[df['label'] == 'Stress'])
        df['label'] = df['label'].replace({'Anxiety': 'Stress', 'Stress': 'Stress'})
        after_stress = len(df[df['label'] == 'Stress'])
        logger.info(
            f'Combined Anxiety + Stress: {before_anxiety:,} + {before_stress:,} = {after_stress:,}'
        )
        return df

    def _remove_invalid_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        invalid = df[~df['label'].isin(self.valid_labels)]['label'].value_counts()
        if len(invalid) > 0:
            logger.info(f'Invalid labels found: {invalid.to_dict()}')

        initial = len(df)
        df = df[df['label'].isin(self.valid_labels)]
        removed = initial - len(df)
        self.report['removed_invalid_labels'] = removed
        logger.info(f'Removed {removed:,} rows with invalid labels')
        return df

    def _fix_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        issues_count = 0
        for bad_char, good_char in self.encoding_fixes.items():
            count = df['text'].str.contains(bad_char, regex=False).sum()
            if count > 0:
                issues_count += count
                df['text'] = df['text'].str.replace(bad_char, good_char, regex=False)

        # Normalize whitespace
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        logger.info(f'Fixed {issues_count:,} encoding issues and normalized whitespace')
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        removed = initial - len(df)
        self.report['removed_duplicates'] = removed
        logger.info(f'Removed {removed:,} duplicate text entries')
        return df

    def _create_label_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create label encodings for ML models.

        category_id (numeric 0-3): for transformer model
        normal, depression, suicidal, stress (one-hot encoding): for RNN
        """
        # Numeric encoding
        label_to_id = {
            label: idx for idx, label in enumerate(sorted(self.valid_labels))
        }
        df['category_id'] = df['label'].map(label_to_id)

        for label in self.valid_labels:
            label_lowercase = f'{label.replace("/", "_").replace("-", "_").lower()}'
            df[label_lowercase] = (df['label'] == label).astype(int)

        logger.info(f'Created encodings: {label_to_id}')
        return df


def run_cleaning_pipeline(data_upload_id: int) -> Dict:
    """
    Clean data and save to database.

    Args:
        data_upload_id: ID of DataUpload record,
        necessary for the data cleaning pipeline to get triggered

    Returns:
        Dict with 'success', 'row_count', or 'error'
    """
    try:
        # Get upload record from DataUpload and set status to processing
        upload = DataUpload.objects.get(id=data_upload_id)

        # If already processing, stop (to prevent double clicks)
        if upload.status == 'processing':
            logger.warning(f'Upload {data_upload_id} is already processing.')
            return {'success': False, 'error': 'Already processing'}

        upload.status = 'processing'
        upload.save()

        logger.info(f'Processing upload ID {data_upload_id}: {upload.file_name}')

        # Run cleaning pipeline
        cleaner = DataCleaningPipeline()
        df_cleaned, report = cleaner.clean_file(upload.file_path)

        # Save cleaned data to database
        _save_to_database(df_cleaned, upload)

        # Update upload metadata
        upload.row_count = len(df_cleaned)
        upload.is_validated = True
        upload.save()

        logger.info(f'=== Cleaning complete: {len(df_cleaned):,} records saved ===')

        # Automatically trigger preprocessing pipeline
        logger.info('>>> Automatically triggering Preprocessing Phase <<<')

        # The cleaning output (in DB) becomes the preprocessing input (from DB)
        preprocess_result = preprocess_cleaned_data(data_upload_id)

        if not preprocess_result.get('success'):
            # If preprocessing fails, still consider cleaning successful, but report the error.
            upload.status = 'failed'
            upload.save()
            return {
                'success': False,
                'error': f'Cleaning successful, but Preprocessing failed: {preprocess_result.get("error")}',
            }

        upload.status = 'completed'
        upload.save()
        return {
            'success': True,
            'row_count': len(df_cleaned),
            'report': report,
            'preprocessing_status': 'completed',
        }

    except DataUpload.DoesNotExist:
        error = f'Upload ID {data_upload_id} not found'
        logger.error(error)
        return {'success': False, 'error': error}

    except Exception as e:
        try:
            upload = DataUpload.objects.get(id=data_upload_id)
            upload.status = 'failed'
            upload.save()
        except:
            pass
        error = f'Pipeline failed: {str(e)}'
        logger.exception(error)
        return {'success': False, 'error': error}


def _save_to_database(df: pd.DataFrame, upload: DataUpload):
    """Save cleaned DataFrame to DatasetRecord model."""
    # Delete old records
    DatasetRecord.objects.filter(data_upload=upload).delete()

    # Bulk create new records
    records = []
    for _, row in df.iterrows():
        record = DatasetRecord(
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
        records.append(record)

    DatasetRecord.objects.bulk_create(records, batch_size=1000)
    logger.info(f'=== Saved {len(records):,} records to database ===')
