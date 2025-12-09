#!/bin/sh

# Author: Julia McCall , Lian Shi

echo "Waiting for postgres to be ready"
while ! nc -z $SQL_HOST $SQL_PORT; do
  sleep 0.1
done
echo "Started PostgreSQL"

echo "Creating migrations..."
python manage.py makemigrations --noinput

echo "Starting database migrations"
python manage.py migrate

echo "Collecting static files"
python manage.py collectstatic --no-input

# Download models from GCS
echo "Downloading ML models from GCS..."
python << PYTHON_END
from google.cloud import storage
import os

bucket_name = os.environ.get('GCS_BUCKET', 'sentio-m_l-models')
models_dir = '/app/ml_models'
os.makedirs(models_dir, exist_ok=True)

try:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix='models/')
    downloaded = 0
    for blob in blobs:
        if blob.name.endswith('.pkl') or blob.name.endswith('.joblib'):
            filename = blob.name.split('/')[-1]
            local_path = f"{models_dir}/{filename}"
            blob.download_to_filename(local_path)
            print(f"Downloaded: {blob.name} -> {local_path}")
            downloaded += 1

    if downloaded == 0:
        print("No models found in GCS. Will use local models if available.")
    else:
        print(f"Downloaded {downloaded} models from GCS")
except Exception as e:
    print(f"Warning: Could not download models from GCS: {e}")
    print("Will use local models if available")
PYTHON_END

# Create superuser if needed
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
  echo "Creating superuser..."
  python manage.py shell << PYTHON_END
from django.contrib.auth import get_user_model
import os
User = get_user_model()
username = os.environ.get('DJANGO_SUPERUSER_USERNAME')
email = os.environ.get('DJANGO_SUPERUSER_EMAIL', '')
password = os.environ.get('DJANGO_SUPERUSER_PASSWORD')
if not User.objects.filter(username=username).exists():
    User.objects.create_superuser(username, email, password)
    print(f'Superuser {username} created!')
else:
    print(f'Superuser {username} already exists')
PYTHON_END
fi

echo "Starting Gunicorn"
exec gunicorn sentio.wsgi:application --bind 0.0.0.0:8000 --timeout 300
