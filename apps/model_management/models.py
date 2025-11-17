from django.contrib.auth.models import User
from django.db import models

# Create your models here.

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

class DatasetRecord(models.Model):
    """
    Dataset record with raw text, label, and pre-computed features.

    Stores both text (for custom feature extraction) and pre-computed
    linguistic features from the CSV (for experimentation).
    """

    DATASET_OPTIONS = (             # Used to separate the training data from the test data
        ('train', 'Training'),
        ('test', 'Test'),
        ('unlabeled', 'Unlabeled'),
    )

    # Raw data
    text = models.TextField()
    label = models.IntegerField()  # Target variable (0-4 stress level)

    # Metadata
    subreddit = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(null=True, blank=True)

    # Pre-computed features from Dreaddit (LIWC, sentiment, syntax, etc.)
    # Stored as JSON for flexibility
    features = models.JSONField(default=dict)

    # Indicates dataset type (train/test/unlabeled)
    dataset_type=models.CharField(max_length=10, choices=DATASET_OPTIONS, default='train')

    # Tracking
    data_upload = models.ForeignKey(
        DataUpload,
        on_delete=models.CASCADE,
        related_name='records'
    )
    is_active = models.BooleanField(default=True)
    imported_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['data_upload', 'is_active']),
            models.Index(fields=['label']),
        ]

    def __str__(self):   # Formats how the model instances are represented as a string
        return f" {self.dataset_type} | {self.label} | {self.text[:30]}"