from django.contrib import admin

from .models import DatasetRecord, DataUpload, ModelVersion, TrainingJob

"""
Admin configurations for the model management, possible to create, read, update, delete objects in the admin panel.
This is intended to be an interim solution until a dedicated admin UI is developed and real data is used.
"""


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ['version_name', 'created_at', 'accuracy', 'created_by']
    list_filter = ['created_at']


@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    list_display = [
        'file_name',
        'uploaded_at',
        'uploaded_by',
        'is_validated',
        'row_count',
    ]
    list_filter = ['is_validated', 'uploaded_at']


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ['id', 'status', 'started_at', 'completed_at', 'initiated_by']
    list_filter = ['status', 'started_at']


@admin.register(DatasetRecord)
class DatasetRecordAdmin(admin.ModelAdmin):
    list_display = ['id', 'label', 'data_upload', 'dataset_type']
    list_filter = ['dataset_type', 'data_upload']
    search_fields = ['text']
    ordering = ['id']  # orders by ascending id
