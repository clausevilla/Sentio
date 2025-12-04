from django.core.management.base import BaseCommand

from apps.ml_admin.services import register_model


class Command(BaseCommand):
    help = 'Register an externally trained model'

    def add_arguments(self, parser):
        parser.add_argument('model_file_path', type=str)
        parser.add_argument('--name', type=str, default=None)
        parser.add_argument('--active', action='store_true')

    def handle(self, *args, **options):
        model = register_model(
            model_file_path=options['model_file_path'],
            version_name=options['name'],
            set_active=options['active'],
        )

        self.stdout.write(
            self.style.SUCCESS(
                f'Registered: {model.version_name} (id={model.id}, accuracy={model.accuracy})'
            )
        )
