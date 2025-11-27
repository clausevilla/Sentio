import threading

from django.contrib import admin, messages

from ml_pipeline.data_cleaning.cleaner import run_cleaning_pipeline

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
        'status',
        'uploaded_at',
        'uploaded_by',
        'is_validated',
        'row_count',
    ]
    list_filter = ['status', 'is_validated', 'uploaded_at']
    actions = ['trigger_cleaning_preprocessing_pipelines']

    # --- Action for manual trigger of both pipelines (in case there was a failure) ---
    def trigger_cleaning_preprocessing_pipelines(self, request, queryset):
        for upload in queryset:
            thread = threading.Thread(target=run_cleaning_pipeline, args=(upload.id,))
            thread.daemon = True
            thread.start()

        self.message_user(
            request,
            'Pipeline started in the background. Refresh page to see status updates.',
            messages.INFO,
        )

    trigger_cleaning_preprocessing_pipelines.short_description = (
        'Clean and preprocess selected datasets'
    )

    # --- Automatically trigger both data processing pipelines on "Save" when dataset is uploaded ---
    def save_model(self, request, obj, form, change):
        """
        Runs cleaning and preprocessing immediately when you click 'SAVE' on the Add DataUpload page.
        """
        super().save_model(request, obj, form, change)

        # Run cleaning and preprocessing pipelines in a background task (separate thread)
        def background_task(upload_id):
            run_cleaning_pipeline(upload_id)

        thread = threading.Thread(target=background_task, args=(obj.id,))
        thread.daemon = True  # Thread dies if server restarts
        thread.start()

        self.message_user(
            request,
            'File saved. Data pipeline started in the background.',
            messages.INFO,
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
    ordering = ['id']
