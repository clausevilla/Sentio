import logging
import re
from typing import Dict, Tuple

import nltk
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


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
        import swifter as swifter

        logger.info('=== Starting Data Preprocessing Pipeline ===')
        logger.info(f'Processing {len(df):,} rows')

        try:
            # Calculate average tokens before preprocessing
            df['_temp_tokens'] = df['text'].apply(lambda x: len(str(x).split()))
            self.report['avg_tokens_before'] = df['_temp_tokens'].mean()
            df = df.drop(columns=['_temp_tokens'])

            # Apply preprocessing
            df['text_preprocessed'] = df['text'].swifter.apply(
                self._preprocess_single_text
            )

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
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        lemmatized = [
            self.lemmatizer.lemmatize(t, self.get_wordnet_pos(pos)) for t, pos in tags
        ]

        # 10. Join tokens back into string
        return ' '.join(lemmatized)

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # safe fallback

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
