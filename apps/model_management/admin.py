from django.contrib import admin, messages

from apps.model_management.data_cleaning_pipeline.data_cleaner import (
    run_cleaning_pipeline,
)

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

    def save_model(self, request, obj, form, change):
        """
        Triggered when the Admin chooses to train a new model.
        """
        # Save the job first so it exists in the DB with status 'PENDING'
        super().save_model(request, obj, form, change)

        # Check if the associated data is already cleaned
        data_upload = obj.data_upload

        if not data_upload.is_validated:
            self.message_user(
                request, 'Triggering Data Cleaning Pipeline', messages.INFO
            )

            # Run the data cleaning pipeline
            success = run_cleaning_pipeline(data_upload.id)

            if success:
                self.message_user(
                    request,
                    'Data Cleaning Complete',
                    messages.SUCCESS,
                )

                # --- TODO: TRIGGER NEXT PIPELINE (PREPROCESSING) HERE ---
            else:
                self.message_user(
                    request,
                    'Data Cleaning Failed',
                    messages.ERROR,
                )
        else:
            self.message_user(
                request,
                'Data was already cleaned. Proceeding to training.',
                messages.SUCCESS,
            )
            # --- TODO: TRIGGER NEXT PIPELINE (DATA PREPROCESSING) HERE ---


@admin.register(DatasetRecord)
class DatasetRecordAdmin(admin.ModelAdmin):
    list_display = ['id', 'label', 'data_upload', 'dataset_type']
    list_filter = ['dataset_type', 'data_upload']
    search_fields = ['text']
    ordering = ['id']  # orders by ascending id
