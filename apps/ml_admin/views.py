# Author: Lian Shi
# Disclaimer: LLM has been used to help with initial structure of views.py, with manual tuning and adjustments made throughout.

"""
ML Admin Dashboard - 6 Pages: Dashboard, Data, Training, Models, Users, Analytics
"""

import json
import os
from datetime import timedelta
from venv import logger

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db.models import Avg, Count
from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .models import DatasetRecord, DataUpload, ModelVersion, TrainingJob

# Import full pipeline (cleaning + preprocessing)
try:
    from apps.ml_admin.services import trigger_full_pipeline_in_background

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# Import predictions
try:
    from apps.predictions.models import PredictionResult, TextSubmission

    PREDICTIONS_AVAILABLE = True
except ImportError:
    PREDICTIONS_AVAILABLE = False

ML_ALGORITHMS = {
    'logistic_regression': {
        'name': 'Logistic Regression',
        'description': 'Fast, interpretable baseline',
        'icon': 'fa-chart-line',
    },
    'random_forest': {
        'name': 'Random Forest',
        'description': 'Ensemble of decision trees',
        'icon': 'fa-tree',
    },
    'rnn': {
        'name': 'RNN/LSTM',
        'description': 'Recurrent neural network',
        'icon': 'fa-network-wired',
    },
    'transformer': {
        'name': 'Custom Transformer',
        'description': 'Custom architecture',
        'icon': 'fa-microchip',
    },
}


# ============================================
# PAGE 1: Dashboard
# ============================================


@staff_member_required
def dashboard_view(request):
    active_model = ModelVersion.objects.filter(is_active=True).first()
    active_model_job = None
    if active_model:
        active_model_job = TrainingJob.objects.filter(
            resulting_model=active_model
        ).first()

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
        except Exception as e:
            logger.exception(f'Failed to get prediction count: {e}')
            pass

    jobs = {
        'total': TrainingJob.objects.count(),
        'pending': TrainingJob.objects.filter(status='PENDING').count(),
        'running': TrainingJob.objects.filter(status='RUNNING').count(),
        'completed': TrainingJob.objects.filter(status='COMPLETED').count(),
        'failed': TrainingJob.objects.filter(status='FAILED').count(),
    }

    recent_jobs = TrainingJob.objects.order_by('-started_at')[:5]
    recent_uploads = DataUpload.objects.order_by('-uploaded_at')[:5]

    # Models for comparison chart (if > 1 model)
    models_for_comparison = None
    all_models = ModelVersion.objects.order_by('-created_at')[:10]
    if all_models.count() > 1:
        models_for_comparison = json.dumps(
            [
                {
                    'name': m.version_name,
                    'accuracy': float(m.accuracy) if m.accuracy else 0,
                    'f1': float(m.f1_score) if m.f1_score else 0,
                }
                for m in all_models
            ]
        )

    # Dataset overview
    dataset_overview = {
        'train': DatasetRecord.objects.filter(dataset_type='train').count(),
        'test': DatasetRecord.objects.filter(dataset_type='test').count(),
        'unlabeled': DatasetRecord.objects.filter(dataset_type='unlabeled').count(),
    }
    dataset_overview['total'] = sum(dataset_overview.values())

    # Label distribution (top 5)
    label_distribution = list(
        DatasetRecord.objects.values('label')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )

    return render(
        request,
        'ml_admin/dashboard.html',
        {
            'active_model': active_model,
            'active_model_job': active_model_job,
            'stats': stats,
            'jobs': jobs,
            'recent_jobs': recent_jobs,
            'recent_uploads': recent_uploads,
            'models_for_comparison': models_for_comparison,
            'dataset_overview': dataset_overview,
            'label_distribution': label_distribution,
        },
    )


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

        # Get counts by dataset_type
        training_count = DatasetRecord.objects.filter(
            data_upload=upload, dataset_type='train'
        ).count()
        test_count = DatasetRecord.objects.filter(
            data_upload=upload, dataset_type='test'
        ).count()
        unlabeled_count = DatasetRecord.objects.filter(
            data_upload=upload, dataset_type='unlabeled'
        ).count()

        uploads_with_stats.append(
            {
                'upload': upload,
                'distribution': dist,
                'training_count': training_count,
                'test_count': test_count,
                'unlabeled_count': unlabeled_count,
            }
        )

    # Overall distribution by label
    overall_distribution = list(
        DatasetRecord.objects.values('label')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    # Overall breakdown by dataset_type
    type_breakdown = {
        'training': DatasetRecord.objects.filter(dataset_type='train').count(),
        'test': DatasetRecord.objects.filter(dataset_type='test').count(),
        'unlabeled': DatasetRecord.objects.filter(dataset_type='unlabeled').count(),
    }

    total_records = DatasetRecord.objects.count()

    return render(
        request,
        'ml_admin/data.html',
        {
            'uploads': uploads_with_stats,
            'total_uploads': uploads.count(),
            'total_records': total_records,
            'validated_count': DataUpload.objects.filter(is_validated=True).count(),
            'overall_distribution': overall_distribution,
            'overall_distribution_json': json.dumps(overall_distribution),
            'type_breakdown': type_breakdown,
            'active_model': ModelVersion.objects.filter(is_active=True).first(),
        },
    )


@staff_member_required
@require_http_methods(['POST'])
def upload_csv_api(request):
    try:
        if 'csv_file' not in request.FILES:
            return JsonResponse(
                {'success': False, 'error': 'No file provided'}, status=400
            )

        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            return JsonResponse(
                {'success': False, 'error': 'File must be CSV'}, status=400
            )

        dataset_type = request.POST.get('dataset_type', 'train')
        if dataset_type not in ['train', 'test', 'unlabeled']:
            dataset_type = 'train'

        upload_dir = os.path.join('data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        filename = f'{timezone.now().strftime("%Y%m%d_%H%M%S")}_{csv_file.name}'
        file_path = os.path.join(upload_dir, filename)

        with open(file_path, 'wb+') as f:
            for chunk in csv_file.chunks():
                f.write(chunk)

        # TODO : please update here matching pipeline types in model later (each uploaded dataset have different preprocessing pipeline types)
        # waiting to be implemented later after preprocessing pipeline types are defined in model
        pipeline_type = request.POST.get('pipeline_type', 'full')

        upload = DataUpload.objects.create(
            uploaded_by=request.user,
            file_name=csv_file.name,
            file_path=file_path,
            is_validated=False,
            status='pending',
            pipeline_type=pipeline_type,
        )

        if PIPELINE_AVAILABLE:
            trigger_full_pipeline_in_background(upload.id, dataset_type)
            return JsonResponse(
                {
                    'success': True,
                    'message': 'Upload started. Processing in background...',
                    'upload_id': upload.id,
                }
            )
        else:
            return JsonResponse(
                {
                    'success': True,
                    'message': 'Uploaded (pipeline not available)',
                    'upload_id': upload.id,
                }
            )

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@staff_member_required
@require_http_methods(['GET'])
def get_upload_status_api(request, upload_id):
    """
    Get the current processing status of an upload.
    Used for polling during file upload/processing.
    """
    try:
        upload = get_object_or_404(DataUpload, id=upload_id)

        return JsonResponse(
            {
                'success': True,
                'upload_id': upload_id,
                'status': upload.status,
                'is_validated': upload.is_validated,
                'row_count': upload.row_count or 0,
                'file_name': upload.file_name,
            }
        )
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@staff_member_required
@require_http_methods(['GET'])
def get_upload_distribution_api(request, upload_id):
    """
    Get the label distribution for a specific upload.
    Used for the distribution modal.
    """
    try:
        upload = get_object_or_404(DataUpload, id=upload_id)

        # Get label distribution
        distribution = list(
            DatasetRecord.objects.filter(data_upload=upload)
            .values('label')
            .annotate(count=Count('id'))
            .order_by('-count')
        )

        # Get total count
        total = DatasetRecord.objects.filter(data_upload=upload).count()

        # Get type breakdown
        type_breakdown = {
            'training': DatasetRecord.objects.filter(
                data_upload=upload, dataset_type='train'
            ).count(),
            'test': DatasetRecord.objects.filter(
                data_upload=upload, dataset_type='test'
            ).count(),
            'unlabeled': DatasetRecord.objects.filter(
                data_upload=upload, dataset_type='unlabeled'
            ).count(),
        }

        return JsonResponse(
            {
                'success': True,
                'upload_id': upload_id,
                'file_name': upload.file_name,
                'distribution': distribution,
                'total': total,
                'type_breakdown': type_breakdown,
            }
        )
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
@require_http_methods(['GET'])
def get_upload_split_api(request, upload_id):
    """Get current train/test split for an upload"""
    upload = get_object_or_404(DataUpload, id=upload_id)

    breakdown = {
        'training': DatasetRecord.objects.filter(
            data_upload=upload, dataset_type='train'
        ).count(),
        'test': DatasetRecord.objects.filter(
            data_upload=upload, dataset_type='test'
        ).count(),
        'unlabeled': DatasetRecord.objects.filter(
            data_upload=upload, dataset_type='unlabeled'
        ).count(),
    }
    breakdown['total'] = sum(breakdown.values())

    return JsonResponse(
        {
            'success': True,
            'upload_id': upload_id,
            'file_name': upload.file_name,
            'breakdown': breakdown,
        }
    )


@staff_member_required
@require_http_methods(['POST'])
def update_upload_split_api(request, upload_id):
    import random

    try:
        upload = get_object_or_404(DataUpload, id=upload_id)
        data = json.loads(request.body)

        action = data.get('action')
        test_percent = data.get('test_percent', 20)

        records = list(DatasetRecord.objects.filter(data_upload=upload))

        if not records:
            return JsonResponse(
                {'success': False, 'error': 'No records found'}, status=400
            )

        if action == 'all_training':
            DatasetRecord.objects.filter(data_upload=upload).update(
                dataset_type='train'
            )
            message = f'All {len(records)} records set to Training'

        elif action == 'all_test':
            DatasetRecord.objects.filter(data_upload=upload).update(dataset_type='test')
            message = f'All {len(records)} records set to Test'

        elif action == 'split':
            from collections import defaultdict

            by_label = defaultdict(list)
            for record in records:
                by_label[record.label].append(record)

            training_ids = []
            test_ids = []

            for label, label_records in by_label.items():
                random.shuffle(label_records)
                n_test = max(1, int(len(label_records) * test_percent / 100))

                for i, record in enumerate(label_records):
                    if i < n_test:
                        test_ids.append(record.id)
                    else:
                        training_ids.append(record.id)

            DatasetRecord.objects.filter(id__in=training_ids).update(
                dataset_type='train'
            )
            DatasetRecord.objects.filter(id__in=test_ids).update(dataset_type='test')

            message = f'Split: {len(training_ids)} training, {len(test_ids)} test ({test_percent}%)'

        else:
            return JsonResponse(
                {'success': False, 'error': 'Invalid action'}, status=400
            )

        breakdown = {
            'training': DatasetRecord.objects.filter(
                data_upload=upload, dataset_type='train'
            ).count(),
            'test': DatasetRecord.objects.filter(
                data_upload=upload, dataset_type='test'
            ).count(),
            'unlabeled': DatasetRecord.objects.filter(
                data_upload=upload, dataset_type='unlabeled'
            ).count(),
        }

        return JsonResponse(
            {
                'success': True,
                'message': message,
                'breakdown': breakdown,
            }
        )

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@staff_member_required
def get_dataset_records_api(request, upload_id):
    """API to get records with sorting and filtering"""
    upload = get_object_or_404(DataUpload, id=upload_id)
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 20))
    sort_by = request.GET.get('sort_by', 'id')
    sort_dir = request.GET.get('sort_dir', 'asc')
    label_filter = request.GET.get('label', None)

    # Clamp per_page
    per_page = max(10, min(per_page, 100))

    # Validate sort column
    allowed_sort_columns = ['id', 'text', 'label']
    if sort_by not in allowed_sort_columns:
        sort_by = 'id'

    # Build order_by
    order_by = f'-{sort_by}' if sort_dir == 'desc' else sort_by

    # Base queryset
    records = DatasetRecord.objects.filter(data_upload=upload)

    # Apply label filter
    if label_filter and label_filter != 'all':
        records = records.filter(label__iexact=label_filter)

    # Order and count
    records = records.order_by(order_by)
    total = records.count()

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    records_page = records[start:end]

    return JsonResponse(
        {
            'success': True,
            'records': [
                {'id': r.id, 'text': r.text, 'label': r.label}  # Full text now
                for r in records_page
            ],
            'total': total,
            'page': page,
            'pages': max(1, (total + per_page - 1) // per_page),
            'per_page': per_page,
            'sort_by': sort_by,
            'sort_dir': sort_dir,
            'label': label_filter,
        }
    )


@staff_member_required
def get_dataset_distribution_api(request, upload_id):
    upload = get_object_or_404(DataUpload, id=upload_id)

    distribution = list(
        DatasetRecord.objects.filter(data_upload=upload)
        .values('label')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    total = sum(d['count'] for d in distribution)

    return JsonResponse(
        {
            'success': True,
            'upload_id': upload_id,
            'file_name': upload.file_name,
            'distribution': distribution,
            'total': total,
        }
    )


# ============================================
# PAGE 3: Training
# ============================================


@staff_member_required
def training_view(request):
    jobs = TrainingJob.objects.order_by('-started_at')[:20]
    available_uploads = DataUpload.objects.filter(is_validated=True).order_by(
        '-uploaded_at'
    )
    active_model = ModelVersion.objects.filter(is_active=True).first()

    uploads_with_counts = []
    for upload in available_uploads:
        training_count = DatasetRecord.objects.filter(
            data_upload=upload, dataset_type='train'
        ).count()

        if training_count > 0:
            dist = list(
                DatasetRecord.objects.filter(data_upload=upload, dataset_type='train')
                .values('label')
                .annotate(count=Count('id'))
            )
            uploads_with_counts.append(
                {
                    'upload': upload,
                    'count': training_count,
                    'distribution': dist,
                    'distribution_json': json.dumps(dist),
                }
            )

    test_set_info = {
        'total': DatasetRecord.objects.filter(dataset_type='test').count(),
        'distribution': list(
            DatasetRecord.objects.filter(dataset_type='test')
            .values('label')
            .annotate(count=Count('id'))
        ),
    }

    training_totals = {
        'records': DatasetRecord.objects.filter(dataset_type='train').count(),
        'uploads': len(uploads_with_counts),
    }

    all_models = ModelVersion.objects.order_by('-created_at')
    retrainable_models = []
    for model in all_models:
        model_info = {
            'id': model.id,
            'version_name': model.version_name,
            'accuracy': float(model.accuracy) if model.accuracy else None,
            'f1_score': float(model.f1_score) if model.f1_score else None,
            'created_at': model.created_at.strftime('%Y-%m-%d'),
            'is_active': model.is_active,
            'algorithm': getattr(model, 'algorithm', 'unknown'),
        }
        retrainable_models.append(model_info)

    return render(
        request,
        'ml_admin/training.html',
        {
            'jobs': jobs,
            'uploads': uploads_with_counts,
            'algorithms': ML_ALGORITHMS,
            'active_model': active_model,
            'running_count': TrainingJob.objects.filter(status='RUNNING').count(),
            'test_set_info': test_set_info,
            'test_set_json': json.dumps(test_set_info['distribution']),
            'training_totals': training_totals,
            'all_models': all_models,
            'retrainable_models_json': json.dumps(retrainable_models),
        },
    )


@staff_member_required
@require_http_methods(['POST'])
def start_training_api(request):
    """Start a new training job with configurable parameters"""
    try:
        data = json.loads(request.body)
        upload_ids = data.get('upload_ids', [])
        mode = data.get('mode', 'new')  # 'new' or 'retrain'
        algorithm = data.get('algorithm', 'logistic_regression')
        base_model_id = data.get('base_model_id')  # For retrain mode
        params = data.get('params', {})  # Training parameters

        if not upload_ids:
            return JsonResponse(
                {'success': False, 'error': 'No datasets selected'}, status=400
            )

        # Validate mode
        if mode not in ['new', 'retrain']:
            return JsonResponse(
                {'success': False, 'error': 'Invalid training mode'}, status=400
            )

        # Validate algorithm
        valid_algorithms = [
            'logistic_regression',
            'random_forest',
            'lstm',
            'transformer',
        ]
        if algorithm not in valid_algorithms:
            return JsonResponse(
                {'success': False, 'error': 'Invalid algorithm'}, status=400
            )

        # For retrain, validate base model and get its algorithm
        base_model = None
        if mode == 'retrain':
            if not base_model_id:
                return JsonResponse(
                    {
                        'success': False,
                        'error': 'No base model selected for retraining',
                    },
                    status=400,
                )
            try:
                base_model = ModelVersion.objects.get(id=base_model_id)
                # Use the base model's algorithm for retrain
                algorithm = base_model.model_type
            except ModelVersion.DoesNotExist:
                return JsonResponse(
                    {'success': False, 'error': 'Base model not found'}, status=400
                )

        # Validate uploads
        uploads = DataUpload.objects.filter(id__in=upload_ids, is_validated=True)
        if uploads.count() != len(upload_ids):
            return JsonResponse(
                {'success': False, 'error': 'Invalid datasets'}, status=400
            )

        # Check if training already running
        if TrainingJob.objects.filter(status='RUNNING').exists():
            return JsonResponse(
                {'success': False, 'error': 'Training already running'}, status=400
            )

        # Create training job with model_type
        job = TrainingJob.objects.create(
            model_type=algorithm,
            status='PENDING',
            initiated_by=request.user,
        )

        # Process params - convert string booleans to actual booleans
        processed_params = {}
        for key, value in params.items():
            if value == 'true':
                processed_params[key] = True
            elif value == 'false':
                processed_params[key] = False
            elif value == '' or value is None:
                processed_params[key] = None
            else:
                processed_params[key] = value

        # Create training configuration using helper function
        from .models import create_training_config

        config = create_training_config(
            training_job=job,
            algorithm=algorithm,
            custom_params=processed_params,
            base_model=base_model,
        )

        # TODO: Trigger actual training
        # from ml_pipeline.training import train_model
        # train_model.delay(job.id, upload_ids)

        algo_names = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'lstm': 'LSTM',
            'transformer': 'Transformer',
        }

        if mode == 'retrain':
            message = f'Started retraining {algo_names.get(algorithm, algorithm)} based on {base_model.version_name}'
        else:
            message = f'Started {algo_names.get(algorithm, algorithm)} training'

        return JsonResponse(
            {
                'success': True,
                'message': message,
                'job_id': job.id,
                'config_id': config.id,
                'mode': mode,
                'algorithm': algorithm,
            }
        )

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

    models_with_info = []
    for model in models:
        job = TrainingJob.objects.filter(resulting_model=model).first()

        training_records = 0
        training_labels = []
        if job and job.data_upload:
            training_records = DatasetRecord.objects.filter(
                data_upload=job.data_upload, dataset_type='train'
            ).count()
            training_labels = list(
                DatasetRecord.objects.filter(data_upload=job.data_upload)
                .values('label')
                .annotate(count=Count('id'))
                .order_by('-count')[:4]
            )

        models_with_info.append(
            {
                'model': model,
                'job': job,
                'training_records': training_records,
                'training_labels': training_labels,
            }
        )

    return render(
        request,
        'ml_admin/models.html',
        {
            'models': models_with_info,
            'active_model': active_model,
            'total_models': models.count(),
        },
    )


@staff_member_required
@require_http_methods(['POST'])
def activate_model_api(request, model_id):
    try:
        model = get_object_or_404(ModelVersion, id=model_id)
        ModelVersion.objects.update(is_active=False)
        model.is_active = True
        model.save()
        return JsonResponse(
            {'success': True, 'message': f'{model.version_name} deployed'}
        )
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@staff_member_required
@require_http_methods(['POST'])
def delete_model_api(request, model_id):
    try:
        model = get_object_or_404(ModelVersion, id=model_id)
        if model.is_active:
            return JsonResponse(
                {'success': False, 'error': 'Cannot delete active model'}, status=400
            )
        if model.model_file_path and os.path.exists(model.model_file_path):
            os.remove(model.model_file_path)
        model.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ============================================
# PAGE 5: Users
# ============================================


@staff_member_required
def users_view(request):
    users = User.objects.order_by('-date_joined')

    status = request.GET.get('status', '')
    if status == 'active':
        users = users.filter(is_active=True)
    elif status == 'inactive':
        users = users.filter(is_active=False)
    elif status == 'staff':
        users = users.filter(is_staff=True)

    paginator = Paginator(users, 20)
    page_obj = paginator.get_page(request.GET.get('page'))

    user_stats = {}
    if PREDICTIONS_AVAILABLE:
        try:
            for user in page_obj:
                user_stats[user.id] = PredictionResult.objects.filter(
                    submission__user=user
                ).count()
        except Exception as e:
            logger.exception(f'Failed to get user prediction counts: {e}')
            pass

    return render(
        request,
        'ml_admin/users.html',
        {
            'page_obj': page_obj,
            'status_filter': status,
            'user_stats': json.dumps(user_stats),
            'counts': {
                'total': User.objects.count(),
                'active': User.objects.filter(is_active=True).count(),
                'staff': User.objects.filter(is_staff=True).count(),
            },
            'active_model': ModelVersion.objects.filter(is_active=True).first(),
        },
    )


# ============================================
# PAGE 6: Analytics
# ============================================


@staff_member_required
def analytics_view(request):
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    two_weeks_ago = today - timedelta(days=13)

    prediction_stats = {
        'available': PREDICTIONS_AVAILABLE,
        'total': 0,
        'today': 0,
        'week': 0,
        'month': 0,
        'distribution': [],
        'daily_counts': [],
        'avg_confidence': None,
    }

    prediction_distribution_json = '[]'
    prediction_daily_json = '[]'

    if PREDICTIONS_AVAILABLE:
        try:
            prediction_stats['total'] = PredictionResult.objects.count()
            prediction_stats['today'] = PredictionResult.objects.filter(
                predicted_at__date=today
            ).count()
            prediction_stats['week'] = PredictionResult.objects.filter(
                predicted_at__date__gte=week_ago
            ).count()
            prediction_stats['month'] = PredictionResult.objects.filter(
                predicted_at__date__gte=month_ago
            ).count()

            distribution = list(
                PredictionResult.objects.values('mental_state')
                .annotate(count=Count('id'))
                .order_by('-count')
            )
            prediction_stats['distribution'] = distribution
            prediction_distribution_json = json.dumps(distribution)

            # Daily prediction counts (last 14 days) - FILL ALL DAYS
            daily_counts_db = dict(
                PredictionResult.objects.filter(predicted_at__date__gte=two_weeks_ago)
                .annotate(date=TruncDate('predicted_at'))
                .values('date')
                .annotate(count=Count('id'))
                .values_list('date', 'count')
            )

            daily_counts = []
            for i in range(13, -1, -1):
                day = today - timedelta(days=i)
                count = daily_counts_db.get(day, 0)
                daily_counts.append({'date': day.strftime('%m/%d'), 'count': count})

            prediction_stats['daily_counts'] = daily_counts
            prediction_daily_json = json.dumps(daily_counts)

            avg = PredictionResult.objects.aggregate(avg=Avg('confidence'))
            prediction_stats['avg_confidence'] = (
                round(avg['avg'] * 100, 1) if avg['avg'] else None
            )

        except Exception as e:
            prediction_stats['error'] = str(e)

    submission_stats = {
        'available': PREDICTIONS_AVAILABLE,
        'total': 0,
        'today': 0,
        'week': 0,
    }

    if PREDICTIONS_AVAILABLE:
        try:
            submission_stats['total'] = TextSubmission.objects.count()
            submission_stats['today'] = TextSubmission.objects.filter(
                submitted_at__date=today
            ).count()
            submission_stats['week'] = TextSubmission.objects.filter(
                submitted_at__date__gte=week_ago
            ).count()
        except Exception as e:
            logger.exception(f'Failed to get submission counts: {e}')
            pass

    user_stats = {
        'total': User.objects.count(),
        'active': User.objects.filter(is_active=True).count(),
        'new_week': User.objects.filter(date_joined__date__gte=week_ago).count(),
        'new_month': User.objects.filter(date_joined__date__gte=month_ago).count(),
    }

    # User signups (last 30 days) - FILL ALL DAYS
    signups_db = dict(
        User.objects.filter(date_joined__date__gte=month_ago)
        .annotate(date=TruncDate('date_joined'))
        .values('date')
        .annotate(count=Count('id'))
        .values_list('date', 'count')
    )

    user_signups = []
    for i in range(29, -1, -1):
        day = today - timedelta(days=i)
        count = signups_db.get(day, 0)
        user_signups.append({'date': day.strftime('%m/%d'), 'count': count})

    return render(
        request,
        'ml_admin/analytics.html',
        {
            'prediction_stats': prediction_stats,
            'prediction_distribution_json': prediction_distribution_json,
            'prediction_daily_json': prediction_daily_json,
            'submission_stats': submission_stats,
            'user_stats': user_stats,
            'user_signups': user_signups,
            'user_signups_json': json.dumps(user_signups),
            'active_model': ModelVersion.objects.filter(is_active=True).first(),
        },
    )

# ============================================
# APIs FOR TRANING JOB/DATASET UPLOAD STATUS CHECK
# ============================================

@staff_member_required
@require_http_methods(['GET'])
def get_jobs_status_api(request):
    """Bulk API for polling training job statuses."""
    from datetime import timedelta

    cutoff = timezone.now() - timedelta(hours=24)
    jobs = TrainingJob.objects.filter(started_at__gte=cutoff).order_by('-started_at')[:20]

    return JsonResponse({
        'success': True,
        'jobs': [
            {
                'id': job.id,
                'status': job.status,
                'model_type': getattr(job, 'model_type', None),
                'started_at': job.started_at.isoformat(),
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            }
            for job in jobs
        ]
    })


@staff_member_required
@require_http_methods(['GET'])
def get_uploads_status_api(request):
    """Bulk API for polling data upload statuses."""
    from datetime import timedelta

    cutoff = timezone.now() - timedelta(hours=24)
    uploads = DataUpload.objects.filter(uploaded_at__gte=cutoff).order_by('-uploaded_at')[:20]

    return JsonResponse({
        'success': True,
        'uploads': [
            {
                'id': upload.id,
                'status': upload.status,
                'file_name': upload.file_name,
                'row_count': upload.row_count,
            }
            for upload in uploads
        ]
    })