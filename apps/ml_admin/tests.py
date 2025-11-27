import os
import tempfile

import pandas as pd
from django.test import TestCase

from ml_pipeline.data_cleaning.cleaner import (
    run_cleaning_pipeline,
)
from apps.ml_admin.models import DatasetRecord, DataUpload


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
            {'text': 'hello', 'label': 'Normal'}, # Text is too short (<10)
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

        self.csv_file = 'text,label\n'  # Create CSV file
        for row in self.test_data:
            text = row['text'] if row['text'] is not None else ''
            label = row['label'] if row['label'] is not None else ''
            self.csv_file += f'"{text}","{label}"\n'

    def create_test_upload(self):  # Method to create a DataUpload instance
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(self.csv_file)
            temp_file_path = f.name

        upload = DataUpload.objects.create(file_path=temp_file_path)
        return upload, temp_file_path

    # Test complete dataflow of the pipeline (source file -> cleaning pipeline -> database)
    def test_data_flow(self):
        upload, temp_file_path = self.create_test_upload()

        try:
            # Verify source file exists and contains test data
            self.assertTrue(os.path.exists(temp_file_path), 'Error: CSV file not found')

            test_df = pd.read_csv(temp_file_path)  # Read CSV
            initial_row_count = len(
                test_df
            )  # Keep track of the row count before processing
            print(f"{initial_row_count} rows before cleaning")

            # Run pipeline
            outcome = run_cleaning_pipeline(upload.id)
            self.assertTrue(outcome['success'], 'Error: pipeline execution failed')
            self.assertIn('row_count', outcome, 'Error: missing row_count')  # Check rowcount exists
            self.assertIn('report', outcome, 'Error: missing report')
            # Check there is same or fewer rows after cleaning
            self.assertLessEqual(outcome['row_count'], initial_row_count, 'Error: row count should decrease after cleaning')

            # Check data was loaded to the database:
            records = DatasetRecord.objects.filter(data_upload=upload)
            self.assertEqual(len(records), outcome['row_count'], 'Error: mismatch in database record count')
            print(f"{len(records)} records successfully saved in database")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    # Check data is in the expected schema
    def test_data_schema(self):
        upload, temp_file_path = self.create_test_upload()

        try:
            # Run pipeline
            outcome = run_cleaning_pipeline(upload.id)  # Pass specific upload ID
            self.assertTrue(outcome['success'], 'Error: pipeline execution failed')

            # Get records from database
            records = DatasetRecord.objects.filter(data_upload=upload)
            self.assertGreater(len(records), 0, 'Error: no record found in database')

            expected_fields = [
                'id',
                'text',
                'label',
                'category_id',
                'normal',
                'depression',
                'suicidal',
                'stress',
                'dataset_type',
                'data_upload',
                'imported_at',
            ]

            for record in records:
                for field in expected_fields:
                    self.assertTrue(hasattr(record, field), f'Missing field(s): {field}')

            print("Correct Schema: All fields present")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


# Check data has no incorrect values (overly large/small statements, duplicates, invalid labels)
# and correctly processes char encoding

    def test_invalid_text_values(self):
        upload, temp_file_path = self.create_test_upload()

        try:
            outcome = run_cleaning_pipeline(upload.id)  # Run pipeline
            self.assertTrue(outcome['success'], 'Error: pipeline execution failed')

            records = DatasetRecord.objects.filter(data_upload=upload)

            for record in records:
                self.assertGreaterEqual(len(record.text), self.min_text_length)  # Check min chars
                self.assertGreaterEqual(len(record.text.split()), self.min_word_count) # Min words
                self.assertLessEqual(len(record.text), self.max_text_length) # Max chars


            print("Text validation passed: Correct word length for every record")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


    def test_duplicate_values(self):
        upload, temp_file_path = self.create_test_upload()

        try:
            outcome = run_cleaning_pipeline(upload.id)  # Run pipeline
            self.assertTrue(outcome['success'], 'Error: pipeline execution failed')

            records = DatasetRecord.objects.filter(data_upload=upload)
            texts = [record.text for record in records]

            duplicate_texts = [text for text in set(texts) if texts.count(text) > 1]

            self.assertEqual(len(duplicate_texts), 0, f'Error: Found duplicates: {duplicate_texts}')
            print("Text validation passed: No duplicates")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


    def test_invalid_labels(self):
        upload, temp_file_path = self.create_test_upload()

        try:
            outcome = run_cleaning_pipeline(upload.id)  # Run pipeline
            self.assertTrue(outcome['success'], 'Error: pipeline execution failed')

            records = DatasetRecord.objects.filter(data_upload=upload)
            invalid_labels = []

            for record in records:
                if record.label not in self.valid_labels:
                    invalid_labels.append(f"'{record.label}' found in: '{record.text[:50]}...'")

            self.assertEqual(len(invalid_labels), 0, 'Invalid labels found:\n' + '\n'.join(invalid_labels))


            labels = [record.label for record in records] # Check for any Anxiety label
            self.assertNotIn('Anxiety', labels)

            print("Labelling correct: All labels are valid")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


    def test_char_encoding(self):
        upload, temp_file_path = self.create_test_upload()

        try:
            outcome = run_cleaning_pipeline(upload.id)  # Run pipeline
            self.assertTrue(outcome['success'], 'Error: pipeline execution failed')

            records = DatasetRecord.objects.filter(data_upload=upload)

            for record in records:
                self.assertNotIn('â€œ', record.text)    # Verify encoding fixes were applied
                self.assertNotIn('Ã¼', record.text)
                self.assertNotIn('\u2026', record.text)

            print(f"Correct encoding: {len(records)} records cleaned")

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

