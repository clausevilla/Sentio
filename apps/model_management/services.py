import csv
from django.db import transaction
from apps.model_management.models import DatasetRecord, DataUpload

def import_csv_dataset(file_path, data_upload, dataset_type='train', batch_size=1000):
    """
    Imports a CSV file into DatasetRecord model.
    :param file_path: path to CSV file
    :param data_upload: DataUpload instance
    """
    batch = []

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:      # Map known columns to model fields
            record = DatasetRecord(
                text=row.get('text', ''),
                label=int(row['label']) if row.get('label') else None,
                subreddit=row.get('subreddit', ''),
                confidence=float(row['confidence']) if row.get('confidence') else None,
                features={k: v for k, v in row.items() if k not in ['text', 'label', 'subreddit', 'confidence']},
                dataset_type=dataset_type, # (train, test, or unlabeled)
                data_upload=data_upload
            )
            batch.append(record)

            # Batch insertion for performance (insert in bulks)
            if len(batch) >= batch_size:
                DatasetRecord.objects.bulk_create(batch, ignore_conflicts=True)
                batch = []

        if batch:
            DatasetRecord.objects.bulk_create(batch, ignore_conflicts=True)