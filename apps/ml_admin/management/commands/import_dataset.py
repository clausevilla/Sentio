# Authors: Claudia Sevilla, Julia McCall

from django.core.management.base import BaseCommand

from apps.ml_admin.models import DataUpload
from apps.ml_admin.services import import_csv_dataset


class Command(BaseCommand):  # Import CSV dataset into DatasetRecord
    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str)
        parser.add_argument(
            '--dataset_type',
            type=str,
            default='train',
            choices=['train', 'test', 'unlabeled'],
        )

    def handle(self, *args, **options):
        file_path = options['file_path']
        dataset_type = options['dataset_type']

        # Create or get a DataUpload record to track provenance
        upload = DataUpload.objects.create(file_path=file_path)

        import_csv_dataset(file_path, data_upload=upload, dataset_type=dataset_type)
        self.stdout.write(
            self.style.SUCCESS(f'Imported {dataset_type} dataset from {file_path}')
        )
