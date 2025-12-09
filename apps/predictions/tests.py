# Author: Julia McCall

from unittest.mock import MagicMock, patch

import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase

from apps.ml_admin.models import ModelVersion
from apps.predictions import services
from apps.predictions.models import PredictionResult, TextSubmission


class PredictionTests(TestCase):
    # ----- Text statistics -----
    def test_calculate_metrics_general(self):
        text = 'I am feeling sad, anxious, and angry today.'

        anxiety_level, negativity_level, emotional_intensity, _, _ = (
            services.calculate_metrics(text)
        )

        self.assertGreater(anxiety_level, 0)
        self.assertGreater(negativity_level, 0)
        self.assertGreater(emotional_intensity, 0)

    def test_calculate_metrics_high_anxiety(self):
        text = "I don't know... maybe... it is just hard..."

        anxiety, _, _, _, _ = services.calculate_metrics(text)

        self.assertGreaterEqual(anxiety, 50)  # because of ellipses

    def test_calculate_metrics_emotional_intensity(self):
        text = 'I HATE THIS! WHY IS IT HAPPENING?!'

        _, _, emotional, _, _ = services.calculate_metrics(text)

        self.assertGreater(emotional, 50)  # ! and all caps

    # ----- Recommendations -----

    def test_get_recommendations(self):
        # Test cases: (prediction, confidence, anxiety, expected_fragment)
        test_cases = [
            ('normal', 0.8, 10, 'undoubtedly healthy'),
            ('stress', 0.8, 10, 'limit caffeine'),
            (
                'depression',
                0.8,
                10,
                'speaking with a mental health professional this week',
            ),
            ('suicidal', 0.8, 10, 'dialing 90101'),
            ('normal', 0.6, 10, 'seem to have a healthy'),
            ('stress', 0.3, 10, 'Model confidence is low'),
        ]

        for prediction, confidence, anxiety, expected_fragment in test_cases:
            with self.subTest(prediction=prediction, confidence=confidence):
                recs = services.get_recommendations(prediction, confidence, anxiety)
                full_text = ' '.join(recs)
                self.assertIn(expected_fragment, full_text)

    def test_get_recommendations_adds_anxiety_tip(self):
        recs = services.get_recommendations('normal', 0.9, 60)
        full_text = ' '.join(recs)
        self.assertIn('grounding techniques', full_text)

    # ----- ML logic with mock model -----

    @patch('apps.predictions.services.MODEL')
    def test_analyze_text(self, mock_model):
        mock_model.predict.return_value = ['stress']
        mock_model.predict_proba.return_value = [[0.1, 0.8, 0.1]]

        prediction, confidence = services.analyze_text('some text')

        self.assertEqual(prediction, 'stress')
        self.assertEqual(confidence, 0.8)

    @patch('apps.predictions.services.DataPreprocessingPipeline')
    def test_preprocess_user_input(self, mock_pipeline_cls):
        mock_instance = mock_pipeline_cls.return_value
        mock_df_result = pd.DataFrame({'text_preprocessed': ['clean text']})
        mock_instance.preprocess_dataframe.return_value = (mock_df_result, {})

        input_df = pd.DataFrame({'text': ['raw']})
        result = services.preprocess_user_input(input_df, 'logistic_regression')

        self.assertEqual(result.iloc[0], 'clean text')

    def test_preprocess_user_input_invalid_model(self):
        with self.assertRaises(ValueError):
            services.preprocess_user_input(pd.DataFrame(), 'super_fancy_ai')

    # ----- Integration with mock database and pipelines -----

    @patch('apps.predictions.services.save_prediction_to_database')
    @patch('apps.predictions.services.get_recommendations')
    @patch('apps.predictions.services.calculate_metrics')
    @patch('apps.predictions.services.analyze_text')
    @patch('apps.predictions.services.preprocess_user_input')
    @patch('apps.predictions.services.clean_user_input')
    @patch('apps.predictions.services.ModelVersion')
    def test_get_prediction_result(
        self,
        mock_mv,
        mock_clean,
        mock_preprocess,
        mock_analyze,
        mock_metrics,
        mock_recs,
        mock_save,
    ):
        user_text = 'I am stressed'
        mock_user = MagicMock()

        mock_mv_instance = MagicMock()
        mock_mv_instance.model_type = 'logistic_regression'
        mock_mv.objects.filter.return_value.first.return_value = mock_mv_instance
        mock_clean.return_value = pd.DataFrame()
        mock_df_processed = pd.DataFrame({'text_preprocessed': ['processed text']})
        mock_preprocess.return_value = mock_df_processed['text_preprocessed']
        mock_analyze.return_value = ('stress', 0.85)
        mock_metrics.return_value = (40, 20, 10, 5, 25)
        mock_recs.return_value = ['Take a break']

        result = services.get_prediction_result(mock_user, user_text)

        (
            prediction,
            confidence_percentage,
            _,
            anxiety,
            _,
            _,
            _,
            _,
        ) = result

        self.assertEqual(prediction, 'stress')
        self.assertEqual(confidence_percentage, 85)
        self.assertEqual(anxiety, 40)

        mock_save.assert_called_once()

        mock_preprocess.assert_called_with(
            mock_clean.return_value, 'logistic_regression'
        )

    # ----- Database save -----

    def test_save_prediction_to_database_creates_record(self):
        user = User.objects.create(username='testuser')
        mv = ModelVersion.objects.create(model_type='lr', is_active=True)

        text = 'Unique text input'
        recommendations = ['Do yoga']

        services.save_prediction_to_database(
            user, text, 'stress', 0.8, mv, recommendations
        )

        self.assertEqual(TextSubmission.objects.count(), 1)
        self.assertEqual(PredictionResult.objects.count(), 1)

        res = PredictionResult.objects.first()
        self.assertEqual(res.mental_state, 'stress')

        self.assertEqual(str(res.recommendations), str(recommendations))

    def test_save_prediction_ignores_template_text(self):
        user = User.objects.create(username='testuser')
        mv = MagicMock()

        template_text = (
            'I feel so empty inside. Nothing brings me joy anymore.'
            " I wake up each day wondering what's the point."
            " I used to love painting but now I can't even pick up a brush."
            ' My friends invite me out but I just make excuses.'
            " I'm tired all the time but can't sleep properly. Everything feels gray and meaningless."
        )

        services.save_prediction_to_database(
            user, template_text, 'depression', 0.9, mv, []
        )

        self.assertEqual(TextSubmission.objects.count(), 0)
