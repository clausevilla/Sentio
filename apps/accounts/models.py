# Author: Lian Shi


"""
Models for user accounts

Note: We're using Django's built-in User model which already includes:
- username (CharField, max 150 chars, unique)
- email (EmailField)
- password (automatically hashed by Django)
- date_joined (DateTimeField, auto-set on creation)
- is_active (BooleanField, default True)
- is_staff (BooleanField, default False)
- is_superuser (BooleanField, default False)
- last_login (DateTimeField, nullable)

The built-in User model provides everything we need for basic authentication.

"""

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


class UserConsent(models.Model):
    """
    Model to track user consent for data processing.
    Required for GDPR compliance.
    """

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='consent')
    has_consented = models.BooleanField(default=False)
    consent_at = models.DateTimeField(null=True, blank=True)
    consent_revoked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'User Consent'
        verbose_name_plural = 'User Consents'

    def __str__(self):
        status = 'Consented' if self.has_consented else 'Not Consented'
        return f'{self.user.username} - {status}'

    def give_consent(self):
        """Record that the user has given consent."""
        self.has_consented = True
        self.consent_at = timezone.now()
        self.consent_revoked_at = None
        self.save()

    def revoke_consent(self):
        """Record that the user has revoked consent."""
        self.has_consented = False
        self.consent_revoked_at = timezone.now()
        self.save()
