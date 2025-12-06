#!/bin/sh

# Author: Julia McCall

echo "Waiting for postgres to be ready"
while ! nc -z $SQL_HOST $SQL_PORT; do
  sleep 0.1
done
echo "Started PostgreSQL"

echo "Starting database migrations"
python manage.py migrate

echo "Collecting static files"
python manage.py collectstatic --no-input

echo "Starting Gunicorn"
exec gunicorn sentio.wsgi:application --bind 0.0.0.0:8000 --timeout 300