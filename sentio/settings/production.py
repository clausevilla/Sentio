import os

from .base import *

DEBUG = False
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')

# Likely adding PostgreSQL for production later
DATABASES = {}
