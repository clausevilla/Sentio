import logging

import pandas as pd

from apps.ml_admin.models import DatasetRecord, DataUpload

logger = logging.getLogger(__name__)


def run_cleaning_pipeline(data_upload_id):
    """
    Accepts an ID, finds the file, cleans the data, and saves
    the cleaned data to the database.
    """
    # --- 1. Cleaning Pipeline Setup and Configuration ---

    # 1.1 Get the file the admin uploaded
    try:
        upload_instance = DataUpload.objects.get(id=data_upload_id)
        file_path = upload_instance.file_path  # Uses the dynamic path from DB
    except DataUpload.DoesNotExist:
        return {'success': False, 'error': 'Upload not found'}

    # 1.2 Load Data from File
    try:
        df_combined = pd.read_csv(file_path)
    except Exception as e:
        return {'success': False, 'error': str(e)}

    # 1.3 Configure Constants
    MIN_WORD_COUNT = 3
    MAX_TEXT_LENGTH = 5000
    VALID_LABELS = ['Normal', 'Depression', 'Suicidal', 'Stress']

    # --- 2. Create Combined Dataset ---

    # 2.1 Load and check the data
    try:
        df_combined = pd.read_csv(file_path)
    except Exception as e:
        return {'success': False, 'error': str(e)}

    # Check for missing values
    print('Missing values per column:')
    print(df_combined.isnull().sum())

    # Check label distribution
    print('\nOriginal label distribution:')
    print(df_combined['label'].value_counts())

    print('\nFinal label distribution:')
    label_dist = df_combined['label'].value_counts()
    print(label_dist)

    print('\nAs percentages:')
    print((label_dist / len(df_combined) * 100).round(2))

    # --- 3. Data Cleaning ---

    # 3.1 Remove rows with missing labels
    initial_count = len(df_combined)
    df_combined = df_combined.dropna(subset=['label'])
    removed = initial_count - len(df_combined)

    print(f'Removed {removed:,} rows with missing labels')
    print(f'Remaining number of rows: {len(df_combined):,}')

    # 3.2 Handle rows with missing, short, or too long texts
    initial_count = len(df_combined)

    # Remove rows with missing texts
    df_combined = df_combined.dropna(subset=['text'])

    # Remove texts that are just whitespace
    df_combined['text'] = df_combined['text'].astype(str)
    df_combined = df_combined[df_combined['text'].str.strip() != '']

    removed = initial_count - len(df_combined)
    print(f'Removed {removed:,} rows with empty text')
    print(f'Remaining number of rows: {len(df_combined):,}')

    # Remove rows with too short text
    df_combined['word_count'] = df_combined['text'].str.split().str.len()
    count_before_short = len(df_combined)
    df_combined = df_combined[df_combined['word_count'] >= MIN_WORD_COUNT]
    removed_short = count_before_short - len(df_combined)
    df_combined = df_combined.drop(columns=['word_count'])

    print(f'\nRemoved {removed_short:,} text with fewer than 3 words')
    print(f'Remaining rows: {len(df_combined):,}')

    # Trim text to maximum allowed length
    num_long_text = df_combined[df_combined['text'].str.len() > MAX_TEXT_LENGTH]
    print(f'Found {len(num_long_text):,} text longer than {MAX_TEXT_LENGTH} characters')

    df_combined['text'] = df_combined['text'].str[:MAX_TEXT_LENGTH]

    if len(num_long_text) > 0:
        print(f'\nTrimmed {len(num_long_text):,} text to {MAX_TEXT_LENGTH} characters')
    else:
        print(f'No text needed trimming (max length already <= {MAX_TEXT_LENGTH})')

    # 3.3 Clean up labels

    # Check counts before combining
    print('Before combining:')
    print(f'Anxiety: {len(df_combined[df_combined["label"] == "Anxiety"]):,}')
    print(f'Stress:  {len(df_combined[df_combined["label"] == "Stress"]):,}')

    # Combine Anxiety and Stress into Stress
    df_combined['label'] = df_combined['label'].replace(
        {'Anxiety': 'Stress', 'Stress': 'Stress'}
    )

    print('\nAfter combining:')
    print(f'Stress: {len(df_combined[df_combined["label"] == "Stress"]):,}')

    # Keep only valid labels
    invalid_labels = df_combined[~df_combined['label'].isin(VALID_LABELS)][
        'label'
    ].value_counts()
    print('Labels to be removed:')
    print(invalid_labels)

    initial_count = len(df_combined)
    df_combined = df_combined[df_combined['label'].isin(VALID_LABELS)]
    removed = initial_count - len(df_combined)

    print(f'\nRemoved {removed:,} rows with invalid labels')
    print(f'Remaining number of rows: {len(df_combined):,}')

    # 3.4 Check for character encoding issues
    encoding_fixes = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'â€¦': '...',
        'Â': '',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã ': 'à',
        'Ã¢': 'â',
        'Ã´': 'ô',
        'Ã«': 'ë',
        'Ã¯': 'ï',
        'Ã¼': 'ü',
        'Ã±': 'ñ',
        'Ã§': 'ç',
        '\x00': '',
        '\ufeff': '',
        '\u2018': "'",
        '\u2019': "'",
        '\u201a': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u201e': '"',
        '\u2013': '-',
        '\u2014': '--',
        '\u2026': '...',
        '\u00a0': ' ',
    }

    issues_found = {}
    for bad_char, good_char in encoding_fixes.items():
        count = df_combined['text'].str.contains(bad_char, regex=False).sum()
        if count > 0:
            issues_found[bad_char] = count
            print(f"Encoding issues found: '{bad_char}' in {count:,} rows")

    if not issues_found:
        print('No common encoding issues found')

    # Fix issues
    if issues_found:
        for bad_char, good_char in encoding_fixes.items():
            df_combined['text'] = df_combined['text'].str.replace(
                bad_char, good_char, regex=False
            )

    # Normalize whitespaces
    df_combined['text'] = (
        df_combined['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    )

    # 3.5 Remove rows with duplicate text
    duplicate_count = df_combined['text'].duplicated().sum()
    print(f'Number of duplicate text: {duplicate_count:,}')

    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['text'], keep='first')
    removed = initial_count - len(df_combined)

    print(f'\nRemoved {removed:,} duplicate text')
    print(f'Remaining rows: {len(df_combined):,}')

    # 3.6 Reset the row indices
    df_combined = df_combined.reset_index(drop=True)
    print(f'Fixed row indeces\nNumber of rows in the new dataset: {len(df_combined):,}')
    print(df_combined.head())
    print(df_combined.tail())

    # 3.7 Create new label columns

    # Numeric encoding column (transformer)
    label_to_id = {label: idx for idx, label in enumerate(sorted(VALID_LABELS))}
    df_combined['category_id'] = df_combined['label'].map(label_to_id)
    print('\nLabel mapping for numeric encoding (category_id column):')
    for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
        print(f'  {idx}: {label}')

    # One-hot encoded columns
    for label in VALID_LABELS:
        # Create column name like normal, depression etc.
        col_name = f'{label.replace("/", "_").replace("-", "_").lower()}'
        df_combined[col_name] = (df_combined['label'] == label).astype(int)

    print(f'\nColumns: {list(df_combined.columns)}')

    # --- 4. Check the Cleaned Dataset ---
    print(f'Cleaned dataset shape: {df_combined.shape}')
    print(f'Columns: {list(df_combined.columns)}')

    print('\nFinal label distribution:')
    label_dist = df_combined['label'].value_counts()
    print(label_dist)

    print('\nAs percentages:')
    print((label_dist / len(df_combined) * 100).round(2))

    # --- 5. Save Cleaned Data to Database ---
    # Connects the cleaning pipeline to the database for the next pipeline to pick up
    try:
        records = []
        for _, row in df_combined.iterrows():
            records.append(
                DatasetRecord(
                    data_upload=upload_instance,
                    text=row['text'],
                    label=row['label'],
                    # TODO: add category_id mapping logic here
                )
            )
        DatasetRecord.objects.bulk_create(records)

        return {'success': True, 'row_count': len(records)}

    except Exception as e:
        return {'success': False, 'error': str(e)}
