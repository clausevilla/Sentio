from django.contrib.auth.models import User
from django.db import models


class ModelVersion(models.Model):
    """
    Tracks different versions of trained ML models.

    Each version stores performance metrics and file location.
    Only one version should be active (is_active=True) at a time.
    Uses SET_NULL for created_by to preserve model history even if user deleted.
    """

    version_name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    model_file_path = models.CharField(max_length=255)
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=False)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)


class DataUpload(models.Model):
    """
    Tracks CSV files uploaded by admins for model training.

    Stores file metadata and validation status. Row count is calculated
    after upload. Uses SET_NULL for uploaded_by to preserve upload history
    even if user account is deleted.
    """

    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=255)
    row_count = models.IntegerField(null=True, blank=True)
    is_validated = models.BooleanField(default=False)
    validation_errors = models.TextField(null=True, blank=True)


class TrainingJob(models.Model):
    """
    Represents a model training/retraining operation initiated by an admin.

    Tracks the lifecycle of training from start to completion or failure.
    CASCADE delete on data_upload because training job is meaningless without its data.
    OneToOne relationship with resulting_model ensures one job produces at most one model.
    """

    STATUS_CHOICES = {
        'PENDING': 'Pending',
        'RUNNING': 'Running',
        'COMPLETED': 'Completed',
        'FAILED': 'Failed',
    }
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    data_upload = models.ForeignKey(DataUpload, on_delete=models.CASCADE)
    resulting_model = models.OneToOneField(
        ModelVersion, on_delete=models.SET_NULL, null=True
    )
    error_message = models.TextField(null=True, blank=True)
    initiated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
