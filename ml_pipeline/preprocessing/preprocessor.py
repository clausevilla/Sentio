import logging
import re
from typing import Dict, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

CHUNK_SIZE = 900


class DataPreprocessingPipeline:
    """
    Preprocesses cleaned text to prepare the data for ML model training.

    Assumes the input is a cleaned dataset.
    Creates a new 'text_preprocessed' column with processed text.
    """

    def __init__(self):
        self._download_nltk_resources()

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Dictionary for mapping the most common contractions
        self.contractions = {
            "ain't": 'am not',
            "aren't": 'are not',
            "can't": 'cannot',
            "can't've": 'cannot have',
            "could've": 'could have',
            "couldn't": 'could not',
            "couldn't've": 'could not have',
            "didn't": 'did not',
            "doesn't": 'does not',
            "don't": 'do not',
            "hadn't": 'had not',
            "hadn't've": 'had not have',
            "hasn't": 'has not',
            "haven't": 'have not',
            "he'd": 'he would',
            "he'd've": 'he would have',
            "he'll": 'he will',
            "he's": 'he is',
            "how'd": 'how did',
            "how'll": 'how will',
            "how's": 'how is',
            "i'd": 'i would',
            "i'd've": 'i would have',
            "i'll": 'i will',
            "i'm": 'i am',
            "i've": 'i have',
            "isn't": 'is not',
            "it'd": 'it would',
            "it'd've": 'it would have',
            "it'll": 'it will',
            "it's": 'it is',
            "let's": 'let us',
            "ma'am": 'madam',
            "might've": 'might have',
            "mightn't": 'might not',
            "must've": 'must have',
            "mustn't": 'must not',
            "needn't": 'need not',
            "oughtn't": 'ought not',
            "shan't": 'shall not',
            "she'd": 'she would',
            "she'd've": 'she would have',
            "she'll": 'she will',
            "she's": 'she is',
            "should've": 'should have',
            "shouldn't": 'should not',
            "that'd": 'that would',
            "that's": 'that is',
            "there'd": 'there would',
            "there's": 'there is',
            "they'd": 'they would',
            "they'll": 'they will',
            "they're": 'they are',
            "they've": 'they have',
            "wasn't": 'was not',
            "we'd": 'we would',
            "we'll": 'we will',
            "we're": 'we are',
            "we've": 'we have',
            "weren't": 'were not',
            "what'll": 'what will',
            "what're": 'what are',
            "what's": 'what is',
            "what've": 'what have',
            "where'd": 'where did',
            "where's": 'where is',
            "who'll": 'who will',
            "who's": 'who is',
            "won't": 'will not',
            "wouldn't": 'would not',
            "you'd": 'you would',
            "you'll": 'you will',
            "you're": 'you are',
            "you've": 'you have',
        }

        # Auto-generated report after preprocessing, may be useful
        self.report = {
            'rows_processed': 0,
            'avg_tokens_before': 0,
            'avg_tokens_after': 0,
        }

    def _download_nltk_resources(self):
        """Downloads required NLTK data if not already present."""
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f'Downloading NLTK resource: {resource}')
                nltk.download(resource, quiet=True)

    def preprocess_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocesses text in a DataFrame.

        Args:
            df: DataFrame with a 'text' column to be processed

        Returns:
            Tuple of (df_with_preprocessed_text, report_dict)
        """
        logger.info('=== Starting Data Preprocessing Pipeline ===')
        logger.info(f'Processing {len(df):,} rows')

        try:
            # Calculate average tokens before preprocessing
            df['_temp_tokens'] = df['text'].apply(lambda x: len(str(x).split()))
            self.report['avg_tokens_before'] = df['_temp_tokens'].mean()
            df = df.drop(columns=['_temp_tokens'])

            # Apply preprocessing
            df['text_preprocessed'] = df['text'].apply(self._preprocess_single_text)

            # Calculate average tokens after preprocessing
            df['_temp_tokens'] = df['text_preprocessed'].apply(lambda x: len(x.split()))
            self.report['avg_tokens_after'] = df['_temp_tokens'].mean()
            df = df.drop(columns=['_temp_tokens'])

            self.report['rows_processed'] = len(df)

            logger.info('=== Preprocessing complete ===')
            logger.info(
                f'   Avg tokens: {self.report["avg_tokens_before"]:.1f} → {self.report["avg_tokens_after"]:.1f}'
            )

            return df, self.report

        except Exception as e:
            logger.error(f'Error in preprocessing: {str(e)}', exc_info=True)
            raise

    def _preprocess_single_text(self, text: str) -> str:
        """
        Preprocesses a single text string.

        Steps:
        1. Remove URLs, mentions, hashtag symbols
        2. Convert all text to lowercase
        3. Expand contractions
        4. Remove special characters and punctuation
        5. Remove extra whitespace
        6. Tokenize text
        7. Remove stopwords
        8. Lemmatize
        9. Remove numbers (may change if we decide they are meaningful for classification)
        """
        if pd.isna(text):
            return ''

        text = str(text)

        # 1.1 Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # 1.2 Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)

        # 1.3 Remove # symbols but keep the hashtag text
        text = text.replace('#', '')

        # 2. Convert to lowercase
        text = text.lower()

        # 3. Expand contractions
        text = self._expand_contractions(text)

        # 4. Remove special characters and punctuation (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)

        # 5. Remove extra whitespaces (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text).strip()

        # 6. Tokenize words
        tokens = word_tokenize(text)

        # 7. Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]

        # 8. Remove numbers
        tokens = [word for word in tokens if not word.isdigit()]

        # 9. Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # 10. Join tokens back into string
        return ' '.join(tokens)

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        # Pattern that matches any contraction
        pattern = re.compile(
            r'\b('
            + '|'.join(re.escape(key) for key in self.contractions.keys())
            + r')\b',
            re.IGNORECASE,
        )

        def replace(match):
            return self.contractions[match.group(0).lower()]

        return pattern.sub(replace, text)


def preprocess_cleaned_data(data_upload_id: int) -> Dict:
    """
    This is the entry point for preprocessing cleaned data.
    Loads data, preprocesses the text, removes short texts, updates the database.

    Args:
        data_upload_id: ID of DataUpload record

    Returns:
        Dict with 'success', 'row_count', or 'error'
    """
    from apps.ml_admin.models import DatasetRecord, DataUpload

    try:
        upload = DataUpload.objects.get(id=data_upload_id)
        logger.info(
            f'Preprocessing data for upload ID {data_upload_id}: {upload.file_name}'
        )

        # Load cleaned data
        records = DatasetRecord.objects.filter(data_upload=upload)

        if not records.exists():
            raise ValueError('No cleaned data found. Run cleaning pipeline first.')

        # Convert to DataFrame
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

        logger.info(f'Loaded {len(df):,} records from database')

        # Run preprocessing steps
        preprocessor = DataPreprocessingPipeline()
        df_processed, report = preprocessor.preprocess_dataframe(df)

        # Identify and remove rows with too few word in the preprocessed text
        df_processed['word_count_proc'] = (
            df_processed['text_preprocessed'].str.split().str.len()
        )
        df_keep = df_processed[df_processed['word_count_proc'] >= 3].copy()
        df_delete = df_processed[df_processed['word_count_proc'] < 3].copy()

        deleted_count = len(df_delete)

        # Delete short records from the database in chunks
        if deleted_count > 0:
            ids_to_delete = df_delete['id'].tolist()
            for i in range(0, len(ids_to_delete), CHUNK_SIZE):
                chunk = ids_to_delete[i : i + CHUNK_SIZE]
                DatasetRecord.objects.filter(id__in=chunk).delete()

            logger.info(
                f'Removed {deleted_count:,} records that became < 3 words after preprocessing.'
            )

        # Update the good records in the database
        _update_database(df_keep)

        report['removed_post_preprocessing'] = deleted_count
        report['final_count_after_preprocessing'] = len(df_keep)

        logger.info('=== Preprocessing complete ===')

        return {'success': True, 'row_count': len(df_processed), 'report': report}

    except DataUpload.DoesNotExist:
        error = f'Upload ID {data_upload_id} not found'
        logger.error(error)
        return {'success': False, 'error': error}

    except Exception as e:
        error = f'Preprocessing failed: {str(e)}'
        logger.exception(error)
        return {'success': False, 'error': error}


def _update_database(df: pd.DataFrame):
    """Update DatasetRecord text field with preprocessed text."""
    from apps.ml_admin.models import DatasetRecord

    # Process in chunks to stay within SQLite limits
    id_to_text = dict(zip(df['id'], df['text_preprocessed']))
    all_ids = list(id_to_text.keys())
    total_updated = 0

    for i in range(0, len(all_ids), CHUNK_SIZE):
        chunk_ids = all_ids[i : i + CHUNK_SIZE]

        records = DatasetRecord.objects.filter(id__in=chunk_ids)

        records_to_update = []
        for record in records:
            if record.id in id_to_text:
                record.text = id_to_text[record.id]
                records_to_update.append(record)

        if records_to_update:
            DatasetRecord.objects.bulk_update(
                records_to_update, ['text'], batch_size=CHUNK_SIZE
            )
            total_updated += len(records_to_update)

    logger.info(f'=== Updated {total_updated:,} records with preprocessed text ===')
