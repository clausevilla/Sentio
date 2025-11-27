"""
ML Admin Dashboard
"""

import json
import os
from datetime import timedelta

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .models import DatasetRecord, DataUpload, ModelVersion, TrainingJob

# Import cleaning pipeline
try:
    from ml_pipeline.data_cleaning.cleaner import run_cleaning_pipeline
    CLEANING_AVAILABLE = True
except ImportError:
    CLEANING_AVAILABLE = False

# Import predictions if available
try:
    from apps.predictions.models import PredictionResult
    PREDICTIONS_AVAILABLE = True
except ImportError:
    PREDICTIONS_AVAILABLE = False

ML_ALGORITHMS = {
    'logistic_regression': {'name': 'Logistic Regression', 'description': 'Fast, interpretable baseline', 'icon': 'fa-chart-line'},
    'bert': {'name': 'BERT', 'description': 'Pre-trained transformer', 'icon': 'fa-brain'},
    'rnn': {'name': 'RNN/LSTM', 'description': 'Recurrent neural network', 'icon': 'fa-network-wired'},
    'transformer': {'name': 'Custom Transformer', 'description': 'Custom architecture', 'icon': 'fa-microchip'},
}


# ============================================
# PAGE 1: Dashboard
# ============================================

@staff_member_required
def dashboard_view(request):
    active_model = ModelVersion.objects.filter(is_active=True).first()

    stats = {
        'models': ModelVersion.objects.count(),
        'datasets': DataUpload.objects.filter(is_validated=True).count(),
        'records': DatasetRecord.objects.count(),
        'users': User.objects.count(),
        'predictions': 0,
    }

    if PREDICTIONS_AVAILABLE:
        try:
            stats['predictions'] = PredictionResult.objects.count()
        except:
            pass

    jobs = {
        'pending': TrainingJob.objects.filter(status='PENDING').count(),
        'running': TrainingJob.objects.filter(status='RUNNING').count(),
        'completed': TrainingJob.objects.filter(status='COMPLETED').count(),
        'failed': TrainingJob.objects.filter(status='FAILED').count(),
    }

    recent_jobs = TrainingJob.objects.order_by('-started_at')[:5]
    recent_uploads = DataUpload.objects.order_by('-uploaded_at')[:5]

    # Dataset distribution for chart
    distribution = list(
        DatasetRecord.objects.values('label')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    return render(request, 'ml_admin/dashboard.html', {
        'active_model': active_model,
        'stats': stats,
        'jobs': jobs,
        'recent_jobs': recent_jobs,
        'recent_uploads': recent_uploads,
        'distribution': json.dumps(distribution),
    })


# ============================================
# PAGE 2: Data Management
# ============================================

@staff_member_required
def data_view(request):
    uploads = DataUpload.objects.order_by('-uploaded_at')

    # Get distribution for each upload (for inline display)
    uploads_with_stats = []
    for upload in uploads:
        dist = list(
            DatasetRecord.objects.filter(data_upload=upload)
            .values('label')
            .annotate(count=Count('id'))
        )
        uploads_with_stats.append({
            'upload': upload,
            'distribution': dist,
        })

    return render(request, 'ml_admin/data.html', {
        'uploads': uploads_with_stats,
        'total_uploads': uploads.count(),
        'total_records': DatasetRecord.objects.count(),
        'validated_count': DataUpload.objects.filter(is_validated=True).count(),
    })


@staff_member_required
@require_http_methods(['POST'])
def upload_csv_api(request):
    try:
        if 'csv_file' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No file provided'}, status=400)

        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            return JsonResponse({'success': False, 'error': 'File must be CSV'}, status=400)

        upload_dir = os.path.join('data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        filename = f'{timezone.now().strftime("%Y%m%d_%H%M%S")}_{csv_file.name}'
        file_path = os.path.join(upload_dir, filename)

        with open(file_path, 'wb+') as f:
            for chunk in csv_file.chunks():
                f.write(chunk)

        upload = DataUpload.objects.create(
            uploaded_by=request.user,
            file_name=csv_file.name,
            file_path=file_path,
            is_validated=False,
        )

        if CLEANING_AVAILABLE:
            result = run_cleaning_pipeline(upload.id)
            if result.get('success'):
                return JsonResponse({
                    'success': True,
                    'message': f'Processed {result.get("row_count", 0)} records',
                    'upload_id': upload.id,
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Cleaning failed'),
                }, status=400)

        return JsonResponse({
            'success': True,
            'message': 'Uploaded (run cleaning manually)',
            'upload_id': upload.id,
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@staff_member_required
@require_http_methods(['POST'])
def delete_upload_api(request, upload_id):
    try:
        upload = get_object_or_404(DataUpload, id=upload_id)
        if upload.file_path and os.path.exists(upload.file_path):
            os.remove(upload.file_path)
        upload.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@staff_member_required
def get_dataset_records_api(request, upload_id):
    """API to get records for modal display"""
    upload = get_object_or_404(DataUpload, id=upload_id)
    page = int(request.GET.get('page', 1))
    per_page = 20

    records = DatasetRecord.objects.filter(data_upload=upload).order_by('id')
    total = records.count()

    start = (page - 1) * per_page
    end = start + per_page
    records_page = records[start:end]

    return JsonResponse({
        'success': True,
        'records': [{'id': r.id, 'text': r.text[:200], 'label': r.label} for r in records_page],
        'total': total,
        'page': page,
        'pages': (total + per_page - 1) // per_page,
    })


# ============================================
# PAGE 3: Training
# ============================================

@staff_member_required
def training_view(request):
    jobs = TrainingJob.objects.order_by('-started_at')[:20]
    available_uploads = DataUpload.objects.filter(is_validated=True).order_by('-uploaded_at')
    active_model = ModelVersion.objects.filter(is_active=True).first()

    # Add record counts
    uploads_with_counts = []
    for upload in available_uploads:
        count = DatasetRecord.objects.filter(data_upload=upload).count()
        uploads_with_counts.append({'upload': upload, 'count': count})

    return render(request, 'ml_admin/training.html', {
        'jobs': jobs,
        'uploads': uploads_with_counts,
        'algorithms': ML_ALGORITHMS,
        'active_model': active_model,
        'running_count': TrainingJob.objects.filter(status='RUNNING').count(),
    })


@staff_member_required
@require_http_methods(['POST'])
def start_training_api(request):
    try:
        data = json.loads(request.body)
        upload_ids = data.get('upload_ids', [])
        algorithm = data.get('algorithm', 'logistic_regression')

        if not upload_ids:
            return JsonResponse({'success': False, 'error': 'No datasets selected'}, status=400)

        if algorithm not in ML_ALGORITHMS:
            return JsonResponse({'success': False, 'error': 'Invalid algorithm'}, status=400)

        uploads = DataUpload.objects.filter(id__in=upload_ids, is_validated=True)
        if uploads.count() != len(upload_ids):
            return JsonResponse({'success': False, 'error': 'Invalid datasets'}, status=400)

        if TrainingJob.objects.filter(status='RUNNING').exists():
            return JsonResponse({'success': False, 'error': 'Training already running'}, status=400)

        job = TrainingJob.objects.create(
            data_upload=uploads.first(),
            status='PENDING',
            initiated_by=request.user,
        )

        # TODO: Trigger actual training
        # from ml_pipeline.training import train_model
        # train_model.delay(job.id, upload_ids, algorithm)

        return JsonResponse({
            'success': True,
            'message': f'Started {ML_ALGORITHMS[algorithm]["name"]} training',
            'job_id': job.id,
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ============================================
# PAGE 4: Models
# ============================================

@staff_member_required
def models_view(request):
    models = ModelVersion.objects.order_by('-created_at')
    active_model = ModelVersion.objects.filter(is_active=True).first()

    # Add training job info to each model
    models_with_info = []
    for model in models:
        job = TrainingJob.objects.filter(resulting_model=model).first()
        models_with_info.append({'model': model, 'job': job})

    return render(request, 'ml_admin/models.html', {
        'models': models_with_info,
        'active_model': active_model,
        'total_models': models.count(),
    })


@staff_member_required
@require_http_methods(['POST'])
def activate_model_api(request, model_id):
    try:
        model = get_object_or_404(ModelVersion, id=model_id)
        ModelVersion.objects.update(is_active=False)
        model.is_active = True
        model.save()
        return JsonResponse({'success': True, 'message': f'{model.version_name} deployed'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@staff_member_required
@require_http_methods(['POST'])
def delete_model_api(request, model_id):
    try:
        model = get_object_or_404(ModelVersion, id=model_id)
        if model.is_active:
            return JsonResponse({'success': False, 'error': 'Cannot delete active model'}, status=400)
        if model.model_file_path and os.path.exists(model.model_file_path):
            os.remove(model.model_file_path)
        model.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)