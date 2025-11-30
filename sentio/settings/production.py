import os

from .base import *

DEBUG = False
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')

# Likely adding PostgreSQL for production later
DATABASES = {}

GCS_BUCKET = os.environ('GCS_BUCKET', 'sentio-ml-models')
MODEL_DIR = '/tmp/models'
