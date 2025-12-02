# Author: Marcus Berggren

from django.contrib import admin

from .models import PredictionResult, TextSubmission

"""
Admin configurations for the predictions, possible to create, read, update, delete objects in the admin panel.
This is intended to be an interim solution until a dedicated admin UI is developed and real data is used.
"""


@admin.register(TextSubmission)
class TextSubmissionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'submitted_at', 'text_content_preview']
    list_filter = ['submitted_at', 'user']
    search_fields = ['text_content', 'user__username']

    def text_content_preview(self, obj):
        # Truncate long text to 50 characters for readability in list view
        return (
            obj.text_content[:50] + '...'
            if len(obj.text_content) > 50
            else obj.text_content
        )

    text_content_preview.short_description = 'Text Preview'


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'submission', 'model_version', 'stress_level', 'predicted_at']
    list_filter = ['predicted_at', 'stress_level', 'model_version']
    search_fields = ['submission__text_content']
