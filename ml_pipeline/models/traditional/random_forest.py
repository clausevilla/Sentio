# Author: Marcus Berggren
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseTextClassifier


class RandomForestModel(BaseTextClassifier):
    @property
    def DEFAULT_CONFIG(self):
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1,
        }

    def _create_classifier(self):
        return RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            max_features=self.config['max_features'],
            class_weight='balanced',
            random_state=42,
            n_jobs=self.config['n_jobs'],
        )

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, any]:
        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['classifier']

        feature_names = vectorizer.get_feature_names_out()
        importances = classifier.feature_importances_

        indices = np.argsort(importances)[-top_n:][::-1]

        global_importance = {feature_names[i]: float(importances[i]) for i in indices}

        return {'type': 'global', 'data': global_importance}
