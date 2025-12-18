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

# No longer download models from GCS (active model is stored in memory)

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
exec gunicorn sentio.wsgi:application --bind 0.0.0.0:8000 --timeout 300 --workers 4 --threads 2 --worker-class gthread --max-requests 1000 --max-requests-jitter 100
