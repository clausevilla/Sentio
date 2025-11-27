from django.contrib import admin, messages

from ml_pipeline.data_cleaning.cleaner import run_cleaning_pipeline
from ml_pipeline.preprocessing.preprocessor import preprocess_cleaned_data

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
    """
    Introduces two methods of triggering the data cleaning pipeline:
    - automatic trigger when admin clicks "Save" on a newly uploaded dataset
    - manual trigger through the DataUpload List page
    """

    list_display = [
        'file_name',
        'uploaded_at',
        'uploaded_by',
        'is_validated',
        'row_count',
    ]
    list_filter = ['is_validated', 'uploaded_at']
    actions = ['trigger_cleaning_pipeline', 'trigger_preprocessing_pipeline']

    # --- Action for manual trigger of whole pipeline (cleaning and preprocessing) ---
    def trigger_cleaning_pipeline(self, request, queryset):
        success_count = 0
        for upload in queryset:
            result = run_cleaning_pipeline(upload.id)
            if result.get('success'):
                success_count += 1
                self.message_user(
                    request,
                    f'Success: {upload.file_name}: Cleaned and preprocessed {result.get("row_count")} rows.',
                    messages.SUCCESS,
                )
            else:
                self.message_user(
                    request,
                    f'Error in pipeline {upload.file_name}: {result.get("error")}',
                    messages.ERROR,
                )

    trigger_cleaning_pipeline.short_description = 'Clean and Preprocess Dataset'

    # --- Action for manual trigger (Preprocessing ONLY) ---
    # You can keep this if you ever need to re-run ONLY preprocessing without re-cleaning
    def trigger_preprocessing_pipeline(self, request, queryset):
        success_count = 0
        for upload in queryset:
            if not upload.is_validated:
                self.message_user(
                    request,
                    f'Skipping {upload.file_name}: Data not cleaned yet.',
                    messages.WARNING,
                )
                continue

            result = preprocess_cleaned_data(upload.id)
            if result.get('success'):
                success_count += 1
                self.message_user(
                    request,
                    f'Re-Preprocessed {upload.file_name}: {result.get("row_count")} rows.',
                    messages.SUCCESS,
                )
            else:
                self.message_user(
                    request,
                    f'Error preprocessing {upload.file_name}: {result.get("error")}',
                    messages.ERROR,
                )

    trigger_preprocessing_pipeline.short_description = 'Re-run Preprocessing Only'

    # --- Automatically trigger cleaning pipeline on "Save" when dataset is uploaded ---
    def save_model(self, request, obj, form, change):
        """
        Runs cleaning and preprocessing immediately when you click 'SAVE' on the Add DataUpload page.
        """
        # 1. Save the file to the database first
        super().save_model(request, obj, form, change)

        # 2. Automatically run the pipeline
        self.message_user(
            request,
            f"File saved. Starting data pipeline (cleaning & preprocessing) for '{obj.file_name}'...",
            messages.INFO,
        )

        result = run_cleaning_pipeline(obj.id)

        if result.get('success'):
            self.message_user(
                request,
                f'Pipeline success! Cleaned and preprocessed {result.get("row_count")} rows.',
                messages.SUCCESS,
            )
        else:
            self.message_user(
                request,
                f'Warning: File saved but pipeline failed! Error: {result.get("error")}',
                messages.WARNING,
            )


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
