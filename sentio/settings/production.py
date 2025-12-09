# Authors: Marcus Berggren, Julia McCall

import os

from .base import *

# Secret key from the Docker environment variable
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-fallback-key')
DEBUG = False
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')
DATABASES = {
    'default': {
        'ENGINE': os.environ.get('SQL_ENGINE', 'django.db.backends.postgresql'),
        'NAME': os.environ.get('SQL_DATABASE', 'sentio_db'),
        'USER': os.environ.get('SQL_USER', 'sentio_user'),
        'PASSWORD': os.environ.get('SQL_PASSWORD', 'super_secret_password'),
        'HOST': os.environ.get('SQL_HOST', 'db'),
        'PORT': os.environ.get('SQL_PORT', '5432'),
    }
}  # PostgreSQL database configuration
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'sentio-ml-models')  # Google Cloud Storage
MODEL_DIR = '/app/media/ml-models'  # Points to internal Docker path
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}  # Docker logs
USE_GCS = True
