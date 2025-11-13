from django.db import models

# Create your models here.
class PostContent(models.Model):
    DATASET_OPTIONS = (             # Used to separate the training data from the test data
        ('train', 'Training'),
        ('test', 'Test'),
    )
    post_id = models.CharField(max_length=100, unique=True)
    text = models.TextField()   # textual content of the post
    stress_level = models.FloatField()
    stressor = models.CharField(max_length=100)  # eg. ptsd, relationships, etc.
    emotional_tone = models.CharField(max_length=200)
    social_confidence = models.FloatField()

    # Indicates if the text is part of the training or the test dataset
    dataset_type=models.CharField(max_length=10, choices=DATASET_OPTIONS, default='train')

    def __str__(self):   # Formats how the model instances are represented as a string
        return f"{self.post_id} | {self.dataset_type} | {self.text[:30]}"