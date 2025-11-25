from typing import Any, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


class LogisticRegressionModel:
    """
    Logistic Regression with TF-IDF for rect classification.
    Used as baseline model for classification in the Sentio app.
    """

    DEFAULT_CONFIG = {
        'max_iter': 1000,
        'C': 1.0,
        'solver': 'lbfgs',
        'tfidf': {
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'max_features': None,
        },
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        args:
            config: Optional dict override DEFAULT_CONFIG
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.pipeline = None
        self.build()

    def build(self):
        """Build sklearn pipeline"""

        tfidf_config = self.config['tfidf']

        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=tfidf_config['ngram_range'],
            min_df=tfidf_config['min_df'],
            max_df=tfidf_config['max_df'],
            max_features=tfidf_config['max_features'],
        )

        classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=self.config['max_iter'],
            C=self.config['C'],
            solver=self.config['solver'],
            random_state=42,
        )

        self.pipeline = Pipeline([('tfidf', vectorizer), ('classifier', classifier)])

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Train the model

        args:
            X_train: Raw text data
            y_train: String labels
        """

        self.pipeline.fit(X_train, y_train)
        return self

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate on test set.

        Returns:
            Dict with accuracy, precision, recall, f1_score
        """
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        print('\nLogistic Regression Classification Report:')
        print(classification_report(y_test, y_pred, zero_division=0))

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }
