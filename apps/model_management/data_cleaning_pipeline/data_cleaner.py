import os

import pandas as pd


def run_cleaning_pipeline():
    # --- 1. Cleaning Pipeline Setup and Configuration ---
    INPUT_FILE = '../../data/stressor-data.csv'
    REDDIT_FILE = '../../data/stressed-anxious-cleaned.csv'
    OUTPUT_FILE = '../../data/stressor-data-cleaned.csv'
    MIN_WORD_COUNT = 3
    MAX_TEXT_LENGTH = 5000
    VALID_LABELS = ['Normal', 'Depression', 'Suicidal', 'Stress']

    # Check if input files exist
    print(f'Input file exists: {os.path.exists(INPUT_FILE)}')
    print(f'Reddit file exists: {os.path.exists(REDDIT_FILE)}')

    # --- 2. Create Combined Dataset ---

    # 2.1 Load and check the original and Reddit data
    try:
        df = pd.read_csv(INPUT_FILE)
        df_reddit = pd.read_csv(REDDIT_FILE)

        print(f'\nOriginal dataset shape: {df.shape}')
        print(f'Original columns: {list(df.columns)}')

        print(f'\nReddit dataset shape: {df_reddit.shape}')
        print(f'Reddit columns: {list(df_reddit.columns)}')

    except FileNotFoundError as e:
        print(f'Error loading files: {e}')
        return

    # 2.2 Combine Datasets

    # Drop index column from original dataset to match Reddit dataset
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Re-label Reddit dataset to match column names
    df_reddit_relabeled = pd.DataFrame(
        {
            'statement': df_reddit['Text'],
            'status': df_reddit['is_stressed/anxious'].map({1: 'Stress', 0: 'Normal'}),
        }
    )
    print(f'Re-labeled {len(df_reddit_relabeled):,} rows of Reddit posts')

    # Combine datasets
    df_combined = pd.concat([df, df_reddit_relabeled], ignore_index=True)

    print(f'Original dataset: {len(df):,} rows')
    print(f'Reddit dataset: {len(df_reddit_relabeled):,} rows')
    print(f'Combined dataset: {len(df_combined):,} rows')

    # Check for missing values
    print('Missing values per column:')
    print(df_combined.isnull().sum())

    # Check label distribution
    print('\nOriginal label distribution:')
    print(df_combined['status'].value_counts())

    print('\nFinal label distribution:')
    label_dist = df_combined['status'].value_counts()
    print(label_dist)

    print('\nAs percentages:')
    print((label_dist / len(df_combined) * 100).round(2))

    # --- 3. Data Cleaning ---

    # 3.1 Remove rows with missing labels
    initial_count = len(df_combined)
    df_combined = df_combined.dropna(subset=['status'])
    removed = initial_count - len(df_combined)

    print(f'Removed {removed:,} rows with missing labels')
    print(f'Remaining number of rows: {len(df_combined):,}')

    # 3.2 Handle rows with missing, short, or too long statements
    initial_count = len(df_combined)

    # Remove rows with missing statements
    df_combined = df_combined.dropna(subset=['statement'])

    # Remove statements that are just whitespace
    df_combined['statement'] = df_combined['statement'].astype(str)
    df_combined = df_combined[df_combined['statement'].str.strip() != '']

    removed = initial_count - len(df_combined)
    print(f'Removed {removed:,} rows with empty statements')
    print(f'Remaining number of rows: {len(df_combined):,}')

    # Remove rows with too short statements
    df_combined['word_count'] = df_combined['statement'].str.split().str.len()
    count_before_short = len(df_combined)
    df_combined = df_combined[df_combined['word_count'] >= MIN_WORD_COUNT]
    removed_short = count_before_short - len(df_combined)
    df_combined = df_combined.drop(columns=['word_count'])

    print(f'\nRemoved {removed_short:,} statements with fewer than 3 words')
    print(f'Remaining rows: {len(df_combined):,}')

    # Trim statements to maximum allowed length
    num_long_statements = df_combined[
        df_combined['statement'].str.len() > MAX_TEXT_LENGTH
    ]
    print(
        f'Found {len(num_long_statements):,} statements longer than {MAX_TEXT_LENGTH} characters'
    )

    df_combined['statement'] = df_combined['statement'].str[:MAX_TEXT_LENGTH]

    if len(num_long_statements) > 0:
        print(
            f'\nTrimmed {len(num_long_statements):,} statements to {MAX_TEXT_LENGTH} characters'
        )
    else:
        print(
            f'No statements needed trimming (max length already <= {MAX_TEXT_LENGTH})'
        )

    # 3.3 Clean up labels

    # Check counts before combining
    print('Before combining:')
    print(f'Anxiety: {len(df_combined[df_combined["status"] == "Anxiety"]):,}')
    print(f'Stress:  {len(df_combined[df_combined["status"] == "Stress"]):,}')

    # Combine Anxiety and Stress into Stress
    df_combined['status'] = df_combined['status'].replace(
        {'Anxiety': 'Stress', 'Stress': 'Stress'}
    )

    print('\nAfter combining:')
    print(f'Stress: {len(df_combined[df_combined["status"] == "Stress"]):,}')

    # Keep only valid labels
    invalid_labels = df_combined[~df_combined['status'].isin(VALID_LABELS)][
        'status'
    ].value_counts()
    print('Labels to be removed:')
    print(invalid_labels)

    initial_count = len(df_combined)
    df_combined = df_combined[df_combined['status'].isin(VALID_LABELS)]
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
        count = df_combined['statement'].str.contains(bad_char, regex=False).sum()
        if count > 0:
            issues_found[bad_char] = count
            print(f"Encoding issues found: '{bad_char}' in {count:,} rows")

    if not issues_found:
        print('No common encoding issues found')

    # Fix issues
    if issues_found:
        for bad_char, good_char in encoding_fixes.items():
            df_combined['statement'] = df_combined['statement'].str.replace(
                bad_char, good_char, regex=False
            )

    # Normalize whitespaces
    df_combined['statement'] = (
        df_combined['statement'].str.replace(r'\s+', ' ', regex=True).str.strip()
    )

    # 3.5 Remove rows with duplicate statement text
    duplicate_count = df_combined['statement'].duplicated().sum()
    print(f'Number of duplicate statements: {duplicate_count:,}')

    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['statement'], keep='first')
    removed = initial_count - len(df_combined)

    print(f'\nRemoved {removed:,} duplicate statements')
    print(f'Remaining rows: {len(df_combined):,}')

    # 3.6 Reset the row indices
    df_combined = df_combined.reset_index(drop=True)
    print(f'Fixed row indeces\nNumber of rows in the new dataset: {len(df_combined):,}')
    print(df_combined.head())
    print(df_combined.tail())

    # 3.7 Create new label columns

    # Numeric encoding column (transformer)
    status_to_id = {status: idx for idx, status in enumerate(sorted(VALID_LABELS))}
    df_combined['status_id'] = df_combined['status'].map(status_to_id)
    print('\nStatus mapping for numeric encoding (status_id column):')
    for status, idx in sorted(status_to_id.items(), key=lambda x: x[1]):
        print(f'  {idx}: {status}')

    # One-hot encoded columns
    for status in VALID_LABELS:
        # Create column name like status_Normal, status_Depression etc.
        col_name = f'status_{status.replace("/", "_").replace("-", "_")}'
        df_combined[col_name] = (df_combined['status'] == status).astype(int)

    print(f'\nColumns: {list(df_combined.columns)}')

    # --- 4. Check the Cleaned Dataset ---
    print(f'Cleaned dataset shape: {df_combined.shape}')
    print(f'Columns: {list(df_combined.columns)}')

    print('\nFinal label distribution:')
    label_dist = df_combined['status'].value_counts()
    print(label_dist)

    print('\nAs percentages:')
    print((label_dist / len(df_combined) * 100).round(2))

    # --- 5. Save the Cleaned Data ---
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned dataset saved to '{OUTPUT_FILE}'")


if __name__ == '__main__':
    run_cleaning_pipeline()
