# Author: Julia McCall

import json
import logging
import re
from typing import Dict, Tuple

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# NLTK RESOURCES - DOWNLOAD ONCE AT MODULE LOAD

_nltk_initialized = False


def _ensure_nltk_resources():
    """Download NLTK resources once at module load."""
    global _nltk_initialized
    if _nltk_initialized:
        return

    # Just download quietly - nltk.download handles duplicates gracefully
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logger.warning(f'Failed to download NLTK resource {resource}: {e}')

    _nltk_initialized = True
    logger.info('NLTK resources initialized')


# Download immediately when module is imported
_ensure_nltk_resources()


class DataPreprocessingPipeline:
    """
    Preprocesses cleaned text to prepare the data for ML model training.

    Assumes the input is a cleaned dataset.
    Creates a new 'text_preprocessed' column with processed text.
    """

    def __init__(self):
        # NLTK resources already downloaded at module level
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        DATA_PATH = 'ml_pipeline/data/strings.json'

        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            TEXT_DATA = json.load(f)

        # Stopwords to keep to preserve mental health context
        self.keep_words = set(TEXT_DATA.get('keep_words', []))

        # Remove the 'keep_words' from the standard stopword set
        self.refined_stop_words = self.stop_words - self.keep_words

        # Dictionary for mapping the most common contractions
        self.contractions = TEXT_DATA['contraction_dictionary']

        # Auto-generated report after preprocessing, may be useful
        self.report = {
            'rows_processed': 0,
            'avg_tokens_before': 0,
            'avg_tokens_after': 0,
        }

    def preprocess_dataframe(
        self, df: pd.DataFrame, model_type: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocesses text in a DataFrame.

        Args:
            df: DataFrame with a 'text' column to be processed

        Returns:
            Tuple of (text_preprocessed, report_dict)
        """
        import swifter as swifter

        logger.info('=== Starting Data Preprocessing Pipeline ===')
        logger.info(f'Processing {len(df):,} rows')

        # Perform specific set of preprocessing steps depending on model type
        if model_type == 'traditional':
            # For classic ML models, all preprocessing steps included
            df['text_preprocessed'] = df['text'].swifter.apply(
                self._preprocess_traditional
            )
        elif model_type == 'rnn':
            # For RNN, less preprocessing
            df['text_preprocessed'] = df['text'].swifter.apply(self._preprocess_rnn)
        elif model_type == 'transformer':
            # For transformer, just basic cleanup steps
            df['text_preprocessed'] = df['text'].swifter.apply(
                self._preprocess_transformer
            )
        else:
            raise ValueError(
                "Invalid model type. Must be 'traditional', 'rnn', or 'transformer'"
            )

        return df, self.report

    # Preprocessing branch for traditional ML (logistic regression)
    def _preprocess_traditional(self, text):
        if pd.isna(text) or text == '':
            return ''

        text = self._expand_contractions(str(text).lower())
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.refined_stop_words]
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        return ' '.join(tokens)

    # Preprocessing branch for RNN (LSTM)
    def _preprocess_rnn(self, text):
        if pd.isna(text) or text == '':
            return ''

        # No stopword removal and no lemmatization
        text = self._expand_contractions(str(text).lower())
        text = re.sub(r'([?.!,])', r' \1 ', text)
        text = re.sub(r'[^a-z\s?.!,]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Preprocessing branch for transformer
    def _preprocess_transformer(self, text):
        if pd.isna(text) or text == '':
            return ''

        # Very minimal preprocessing to preserve context
        text = str(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _preprocess_single_text(self, text: str) -> str:
        """
        Used in unit testing.
        Performs all preprocessing steps on a single text string.

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
        tokens = [word for word in tokens if word not in self.refined_stop_words]

        # 8. Remove numbers
        tokens = [word for word in tokens if not word.isdigit()]

        # 9. Lemmatize
        tags = pos_tag(tokens)
        lemmatized = [
            self.lemmatizer.lemmatize(t, self.get_wordnet_pos(pos)) for t, pos in tags
        ]

        # 10. Join tokens back into string
        return ' '.join(lemmatized)

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        Helper function for getting the grammatical type of the token.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

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
