# Author: Marcus Berggren
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseTextClassifier


class LogisticRegressionModel(BaseTextClassifier):
    """
    Logistic regression classifier for text classification.
    """

    @property
    def DEFAULT_CONFIG(self):
        return {'max_iter': 1000, 'C': 1.0, 'solver': 'lbfgs'}

    def _create_classifier(self):
        return LogisticRegression(
            class_weight='balanced',
            max_iter=self.config['max_iter'],
            C=self.config['C'],
            solver=self.config['solver'],
            random_state=42,
        )

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, any]:
        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['classifier']
        feature_names = vectorizer.get_feature_names_out()

        per_class_features = {
            class_id: {
                feature_names[i]: float(coef[i])
                for i in np.argsort(coef)[-top_n:][::-1]
            }
            for class_id, coef in enumerate(classifier.coef_)
        }

        return {'type': 'per_class', 'data': per_class_features}
