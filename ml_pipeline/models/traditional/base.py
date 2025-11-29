# Author: Marcus Berggren
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


class BaseTextClassifier(ABC):
    """Base class for TF-IDF + classifier pipelines."""

    DEFAULT_TFIDF_CONFIG = {
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'max_features': None,
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = self._merge_config(config or {})
        self.pipeline = None
        self._build()

    @property
    @abstractmethod
    def DEFAULT_CONFIG(self) -> Dict[str, Any]:
        """Subclass-specific defaults (is merged with tfidf defaults)."""
        pass

    @abstractmethod
    def _create_classifier(self):
        """Return configured sklearn classifier instance."""
        pass

    def _build(self):
        tfidf_cfg = self.config.get('tfidf', {})
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=tfidf_cfg.get('ngram_range', (1, 2)),
            min_df=tfidf_cfg.get('min_df', 2),
            max_df=tfidf_cfg.get('max_df', 0.95),
            max_features=tfidf_cfg.get('max_features'),
        )
        self.pipeline = Pipeline(
            [
                ('tfidf', vectorizer),
                ('classifier', self._create_classifier()),
            ]
        )

    def fit(self, X_train: pd.Series, y_train: pd.Series):
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        classes = self.pipeline.classes_

        try:
            if len(classes) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                y_test_binarized = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(
                    y_test_binarized,
                    y_pred_proba,
                    average='weighted',
                    multi_class='ovr',
                )
        except ValueError:
            auc = None

        cm = confusion_matrix(y_test, y_pred, labels=classes)

        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc) if auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': classes.tolist(),
            'classification_report': classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            ),
        }

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {'tfidf': self.DEFAULT_TFIDF_CONFIG.copy(), **self.DEFAULT_CONFIG}
        for key, value in user_config.items():
            if (
                isinstance(value, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    @abstractmethod
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """Return feature importance (implementation varies by model type)."""
        pass
