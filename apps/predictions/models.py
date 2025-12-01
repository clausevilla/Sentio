from django.contrib.auth.models import User
from django.db import models

from apps.ml_admin.models import ModelVersion


class TextSubmission(models.Model):
    """
    Stores text submitted by users for stress analysis.

    CASCADE delete when user is deleted since submissions are user-specific data.
    Each submission generates exactly one PredictionResult via OneToOne relationship.
    """

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True, default=1
    )
    text_content = models.TextField(max_length=5000)
    submitted_at = models.DateTimeField(auto_now_add=True)


class PredictionResult(models.Model):
    """
    Stores analysis results for a text submission.

    OneToOne with submission ensures each submission has exactly one result.
    PROTECT on model_version prevents deletion of models that have made predictions,
    preserving audit trail. Includes confidence scores for each predicted metric.

    TODO: Possibly adding JSONField to recommendations for more flexible recommendation storage.
    """

    MENTAL_STATE_CHOICES = [
        ('normal', 'Normal'),
        ('stress', 'Stress'),
        ('depression', 'Depression'),
        ('suicidal', 'Suicidal'),
    ]

    submission = models.OneToOneField(TextSubmission, on_delete=models.CASCADE)
    model_version = models.ForeignKey(ModelVersion, on_delete=models.PROTECT)

    mental_state = models.CharField(
        max_length=20,
        choices=MENTAL_STATE_CHOICES,
        default='normal',
    )
    confidence = models.FloatField(default=0)

    anxiety_level = models.IntegerField(null=True, blank=True)
    negativity_level = models.IntegerField(null=True, blank=True)
    emotional_intensity = models.IntegerField(null=True, blank=True)

    recommendations = models.TextField()
    predicted_at = models.DateTimeField(auto_now_add=True)
