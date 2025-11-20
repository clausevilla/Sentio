import csv
from apps.model_management.models import DatasetRecord, DataUpload

CATEGORY_MAP = {  # Define category mapping
    "normal": 0,
    "depression": 1,
    "suicidal": 2,
    "anxiety": 3,
    "bipolar": 4,
    "stress": 5,
}

def import_csv_dataset(file_path, data_upload, dataset_type='train', batch_size=5000):
    """
    Imports a CSV file into DatasetRecord model using one-hot-encoding and category_id.
    :param file_path: path to CSV file
    :param data_upload: DataUpload instance
    """
    batch = []

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:      # Map known columns to model fields
            label = row.get('status', '').strip().lower()
            category_id = CATEGORY_MAP.get(label, None)

            one_hot_categories = {
                "normal": 0,
                "depression": 0,
                "suicidal": 0,
                "anxiety": 0,
                "bipolar": 0,
                "stress": 0,
                } # Initialise all columns to 0

            if category_id is not None:
                one_hot_categories[label] = 1

            record = DatasetRecord(
                text=row.get('statement', ''),
                label=row.get('status', ''),
                category_id=category_id,
                dataset_type=dataset_type, # (train, test, or unlabeled)
                data_upload=data_upload,
                **one_hot_categories #unpack into model fields
            )
            batch.append(record)

            # Batch insertion for performance (insert in bulks)
            if len(batch) >= batch_size:
                DatasetRecord.objects.bulk_create(batch, ignore_conflicts=True)
                batch = []

        if batch:
            DatasetRecord.objects.bulk_create(batch, ignore_conflicts=True)