from django.contrib.auth.models import User
from django.db import models


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

    submission = models.OneToOneField(TextSubmission, on_delete=models.CASCADE)
    # model_version = models.ForeignKey(ModelVersion, on_delete=models.PROTECT)
    model_version = models.TextField()
    stress_level = models.IntegerField()
    prediction = models.TextField()
    emotional_tone = models.FloatField(default=0.0)
    social_confidence = models.FloatField(default=0.0)
    confidence = models.FloatField(default=0.0)
    recommendations = models.TextField()
    predicted_at = models.DateTimeField(auto_now_add=True)
