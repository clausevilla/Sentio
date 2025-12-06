# Author: Marcus Berggren, Julia McCall

from .base import *

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-t35qj&0r68chrbu%kdn-4_ovfgqo&@1p8snqtj$3^*-3lnt0w_'

DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'data' / 'db.sqlite3',  # BASE_DIR is defined in base.py
    }
}

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        # Prints logs to the console
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        # Saves logs to a file named 'debug.log' in the project root
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'debug.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        # Catch-all logger
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
