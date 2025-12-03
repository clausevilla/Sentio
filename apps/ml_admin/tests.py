import os
import tempfile

import pandas as pd
from django.test import TestCase

from ml_pipeline.data_cleaning.cleaner import DataCleaningPipeline

# from ml_pipeline.preprocessing.preprocessor import DataPreprocessingPipeline


class DataCleaningTests(TestCase):
    # Create a CSV file with edge cases to test the data cleaning pipeline
    def setUp(self):
        self.valid_labels = ['Normal', 'Depression', 'Suicidal', 'Stress']
        self.min_word_count = 3
        self.max_text_length = 5000
        self.min_text_length = 10

        self.test_data = [  # Correct test data for every label
            {'text': 'This is a normal text', 'label': 'Normal'},
            {'text': 'I am very depressed', 'label': 'Depression'},
            {'text': "I don't want to live", 'label': 'Suicidal'},
            {'text': 'I am so stressed about exams', 'label': 'Stress'},
            {
                'text': 'I feel really anxious',
                'label': 'Anxiety',
            },  # Anxiety should become Stress
            # Invalid statements
            {'text': 'Hi', 'label': 'Normal'},  # Too few words (< 3)
            {
                'text': 'loveSEM' * 6000,
                'label': 'Normal',
            },  # Text is too long ( > 5000)
            {'text': '   ', 'label': 'Normal'},  # Whitespace only
            {'text': '', 'label': 'Normal'},  # Empty text
            {'text': 'hello', 'label': 'Normal'},  # Text is too short (<10)
            # Invalid labels
            {'text': 'This is a valid text', 'label': ''},
            {'text': 'This is valid too', 'label': None},
            # Test char encoding issues
            {'text': 'Testing â€œencodingÃ¼ \u2026', 'label': 'Normal'},
            {'text': 'This is a valid text', 'label': 'Normal'},
            # Test duplicates
            {'text': 'This text is twice', 'label': 'Depression'},
            {'text': 'This text is twice', 'label': 'Depression'},
        ]

    def create_csv_file(self):
        csv_file = 'text,label\n'  # Create CSV file
        for row in self.test_data:
            text = row['text'] if row['text'] is not None else ''
            label = row['label'] if row['label'] is not None else ''
            csv_file += f'"{text}","{label}"\n'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_file)
            return f.name

    # Test complete dataflow of the pipeline (source file -> cleaning pipeline)
    def test_data_flow(self):
        temp_file_path = self.create_csv_file()

        try:
            # Verify source file exists and contains test data
            self.assertTrue(os.path.exists(temp_file_path), 'Error: CSV file not found')

            test_df = pd.read_csv(temp_file_path)  # Read CSV
            initial_row_count = len(
                test_df
            )  # Keep track of the row count before processing
            print(f'{initial_row_count} rows before cleaning')

            # Run pipeline
            cleaner = DataCleaningPipeline()
            df_cleaned, report = cleaner.clean_file(temp_file_path)
            self.assertTrue(len(df_cleaned) > 0, 'Error: pipeline execution failed')

            self.assertIn(
                'final_count',
                report,
                'Error: missing final_count',  # Check final count exists
            )
            # Check there is same or fewer rows after cleaning
            self.assertLessEqual(
                len(df_cleaned),
                initial_row_count,
                'Error: row count should decrease after cleaning',
            )

            print(f'{len(df_cleaned)} records successfully processed')

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    # Check data is in the expected schema
    def test_data_schema(self):
        temp_file_path = self.create_csv_file()

        try:
            # Run pipeline
            cleaner = DataCleaningPipeline()
            df_cleaned, report = cleaner.clean_file(temp_file_path)
            self.assertTrue(len(df_cleaned) > 0, 'Error: pipeline execution failed')

            expected_columns = [
                'text',
                'label',
                'category_id',
                'normal',
                'depression',
                'suicidal',
                'stress',
            ]

            for column in expected_columns:
                self.assertTrue(
                    column in df_cleaned.columns, f'Missing field(s): {column}'
                )

            print('Correct Schema: All expected columns are present')

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    # Check data has no incorrect values (overly large/small statements, duplicates, invalid labels)
    # and correctly processes char encoding

    def test_invalid_text_values(self):
        temp_file_path = self.create_csv_file()

        try:
            cleaner = DataCleaningPipeline()
            df_cleaned, report = cleaner.clean_file(temp_file_path)
            self.assertTrue(len(df_cleaned) > 0, 'Error: pipeline execution failed')

            for text in df_cleaned['text']:
                self.assertGreaterEqual(
                    len(text), self.min_text_length
                )  # Check min chars
                self.assertGreaterEqual(
                    len(text.split()), self.min_word_count
                )  # Min words
                self.assertLessEqual(len(text), self.max_text_length)  # Max chars

            print('Text validation passed: Correct word length for every record')

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_duplicate_values(self):
        temp_file_path = self.create_csv_file()

        try:
            cleaner = DataCleaningPipeline()
            df_cleaned, report = cleaner.clean_file(temp_file_path)
            self.assertTrue(len(df_cleaned) > 0, 'Error: pipeline execution failed')

            texts = df_cleaned['text'].tolist()

            duplicate_texts = [text for text in set(texts) if texts.count(text) > 1]

            self.assertEqual(
                len(duplicate_texts), 0, f'Error: Found duplicates: {duplicate_texts}'
            )
            print('Text validation passed: No duplicates')

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_invalid_labels(self):
        temp_file_path = self.create_csv_file()

        try:
            cleaner = DataCleaningPipeline()
            df_cleaned, report = cleaner.clean_file(temp_file_path)
            self.assertTrue(len(df_cleaned) > 0, 'Error: pipeline execution failed')

            invalid_labels = []

            for text, label in zip(df_cleaned['text'], df_cleaned['label']):
                if label not in self.valid_labels:
                    invalid_labels.append(
                        f"'{label}' found in: '{text[:50]}...'"  # Get corresponding text of the invalid label
                    )

            self.assertEqual(
                len(invalid_labels),
                0,
                'Invalid labels found:\n' + '\n'.join(invalid_labels),
            )

            labels = df_cleaned['label'].tolist()  # Check for any Anxiety label
            self.assertNotIn('Anxiety', labels)

            print('Labelling correct: All labels are valid')

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_char_encoding(self):
        temp_file_path = self.create_csv_file()

        try:
            cleaner = DataCleaningPipeline()
            df_cleaned, report = cleaner.clean_file(temp_file_path)
            self.assertTrue(len(df_cleaned) > 0, 'Error: pipeline execution failed')

            for text in df_cleaned['text']:
                self.assertNotIn('â€œ', text)  # Verify encoding fixes were applied
                self.assertNotIn('Ã¼', text)
                self.assertNotIn('\u2026', text)

            print(f'Correct encoding: {len(df_cleaned)} records cleaned')

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
