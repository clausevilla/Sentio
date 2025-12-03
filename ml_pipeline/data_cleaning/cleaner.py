# Author: Julia McCall

import logging
import os
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

MIN_WORD_COUNT = 3
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 5000
VALID_LABELS = ['Normal', 'Depression', 'Suicidal', 'Stress']


class DataCleaningPipeline:
    """
    Cleans raw CSV data for mental health text classification.

    Output columns: text, label, category_id, Normal, Depression, Suicidal, Stress
    """

    def __init__(self):
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

        # Validation step 1: check the file format
        _, file_extension = os.path.splitext(str(file_path))

        if not str(file_path).endswith('.csv'):
            # Create a readable error message even if extension is missing
            received = file_extension if file_extension else 'no file extension'
            raise ValueError(
                f'Invalid file format. Expected .csv, got {received}'
                f'(File: {os.path.basename(file_path)})'
            )

        try:
            # Load data and handle malformed rows if necessary
            df = pd.read_csv(file_path)
            self.report['original_count'] = len(df)
            logger.info(f'Loaded {len(df):,} rows')

            # Validation step 2: check column names
            allowed_text_cols = ['statement', 'text', 'Text']
            allowed_label_cols = ['status', 'label', 'Label']

            found_text = [col for col in df.columns if col in allowed_text_cols]
            found_label = [col for col in df.columns if col in allowed_label_cols]

            # Raise error if columns are invalid or missing
            if not found_text or not found_label:
                raise ValueError(
                    f'Invalid CSV columns. Found: {list(df.columns)}. '
                    f'Expected one text column from {allowed_text_cols} and one label column from {allowed_label_cols}.'
                )

            # Standardize the columns to 'text' and 'label'
            rename_map = {found_text[0]: 'text', found_label[0]: 'label'}
            df.rename(columns=rename_map, inplace=True)

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
            (df['text'].str.len() >= MIN_TEXT_LENGTH)
            & (df['word_count'] >= MIN_WORD_COUNT)
        ]
        df = df.drop(columns=['word_count'])
        removed = initial - len(df)
        self.report['removed_short_text'] = removed

        logger.info(
            f'Removed {removed:,} rows with "text" column having < {MIN_TEXT_LENGTH} characters or < {MIN_WORD_COUNT} words'
        )
        return df

    def _trim_long_text(self, df: pd.DataFrame) -> pd.DataFrame:
        long_count = (df['text'].str.len() > MAX_TEXT_LENGTH).sum()
        df['text'] = df['text'].str[:MAX_TEXT_LENGTH]
        self.report['trimmed_long_text'] = long_count
        logger.info(f'Trimmed {long_count:,} text to {MAX_TEXT_LENGTH} chars')
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
        invalid = df[~df['label'].isin(VALID_LABELS)]['label'].value_counts()
        if len(invalid) > 0:
            logger.info(f'Invalid labels found: {invalid.to_dict()}')

        initial = len(df)
        df = df[df['label'].isin(VALID_LABELS)]
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
        label_to_id = {label: idx for idx, label in enumerate(sorted(VALID_LABELS))}
        df['category_id'] = df['label'].map(label_to_id)

        for label in VALID_LABELS:
            label_lowercase = f'{label.replace("/", "_").replace("-", "_").lower()}'
            df[label_lowercase] = (df['label'] == label).astype(int)

        logger.info(f'Created encodings: {label_to_id}')
        return df
