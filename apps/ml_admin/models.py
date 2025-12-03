# Author: Marcus Berggren, Claudia Sevilla Eslava, Julia McCall
from django.contrib.auth.models import User
from django.db import models


class ModelVersion(models.Model):
    """
    Tracks different versions of trained ML models.

    Each version stores performance metrics and file location.
    Only one version should be active (is_active=True) at a time.
    Uses SET_NULL for created_by to preserve model history even if user deleted.
    """

    MODEL_TYPES = [
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('lstm', 'LSTM'),
        ('transformer', 'Transformer'),
    ]

    model_type = models.CharField(max_length=30, choices=MODEL_TYPES)
    version_name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    model_file_path = models.CharField(max_length=255)
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    roc_plot_base64 = models.TextField(null=True, blank=True)
    confusion_matrix_base64 = models.TextField(null=True, blank=True)
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
    PROCESSING_STATUS = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    status = models.CharField(
        max_length=20, choices=PROCESSING_STATUS, default='pending'
    )

    def __str__(self):  # Print the date data was added
        return f'{self.uploaded_at.strftime("%Y-%m-%d")}'


class DatasetRecord(models.Model):
    """
    Dataset record with raw text, category label, and numeric category ID (0-6).

    Stores both text (for custom feature extraction) and pre-computed categories
    from the CSV.
    """

    DATASET_OPTIONS = (  # Used to separate the training data from the test data
        ('train', 'Training'),
        ('test', 'Test'),
        ('increment', 'Increment'),
    )

    text = models.TextField()
    label = models.CharField(
        max_length=50
    )  # Category label (Anxiety, depression, stress, etc.)

    category_id = models.IntegerField(
        null=True, blank=True
    )  # Numeric category id (0-5) corresponding to the label

    # One hot encoded columns for each category (0 or 1)
    normal = models.IntegerField(default=0)
    depression = models.IntegerField(default=0)
    suicidal = models.IntegerField(default=0)
    stress = models.IntegerField(default=0)

    # Indicates dataset type (train/test/unlabeled)
    dataset_type = models.CharField(
        max_length=10, choices=DATASET_OPTIONS, default='train'
    )

    # Tracking
    data_upload = models.ForeignKey(
        DataUpload, on_delete=models.CASCADE, related_name='records'
    )
    imported_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['category_id']),
            models.Index(fields=['data_upload']),
            models.Index(fields=['label']),
        ]

    def __str__(self):  # Formats how the model instances are represented as a string
        return f' {self.dataset_type} | {self.label} | {self.text[:30]}'


class TrainingJob(models.Model):
    """
    Represents a model training/retraining operation initiated by an admin.

    Tracks the lifecycle of training from start to completion or failure.
    CASCADE delete on data_upload because training job is meaningless without its data.
    OneToOne relationship with resulting_model ensures one job produces at most one model.
    """

    MODEL_TYPES = [
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('lstm', 'LSTM'),
        ('transformer', 'Transformer'),
    ]

    STATUS_CHOICES = {
        'PENDING': 'Pending',
        'RUNNING': 'Running',
        'COMPLETED': 'Completed',
        'FAILED': 'Failed',
    }
    model_type = models.CharField(max_length=30, choices=MODEL_TYPES)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    resulting_model = models.OneToOneField(
        ModelVersion, on_delete=models.SET_NULL, null=True
    )
    error_message = models.TextField(null=True, blank=True)
    initiated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
