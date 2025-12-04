# Author: Lian Shi

"""
ML Admin URLs - 6 Pages + APIs
"""

from django.urls import path

from . import views

app_name = 'ml_admin'

urlpatterns = [
    # Pages
    path('', views.dashboard_view, name='dashboard'),
    path('data/', views.data_view, name='data'),
    path('training/', views.training_view, name='training'),
    path('models/', views.models_view, name='models'),
    path('users/', views.users_view, name='users'),
    path('analytics/', views.analytics_view, name='analytics'),
    # Data APIs
    path('api/upload/', views.upload_csv_api, name='upload_csv'),
    path(
        'api/data/<int:upload_id>/delete/',
        views.delete_upload_api,
        name='delete_upload',
    ),
    path(
        'api/data/<int:upload_id>/records/',
        views.get_dataset_records_api,
        name='get_records',
    ),
    path(
        'api/data/<int:upload_id>/split/', views.get_upload_split_api, name='get_split'
    ),
    path(
        'api/data/<int:upload_id>/split/update/',
        views.update_upload_split_api,
        name='update_split',
    ),
    path(
        'api/data/<int:upload_id>/status/',
        views.get_upload_status_api,
        name='get_status',
    ),
    path(
        'api/data/<int:upload_id>/distribution/',
        views.get_upload_distribution_api,
        name='get_distribution',
    ),
    path(
        'api/data/<int:upload_id>/status/',
        views.get_upload_status_api,
        name='get_status',
    ),
    path(
        'api/data/<int:upload_id>/distribution/',
        views.get_upload_distribution_api,
        name='get_distribution',
    ),
    # Training APIs
    path('api/training/start/', views.start_training_api, name='start_training'),
    # Model APIs
    path(
        'api/models/<int:model_id>/activate/',
        views.activate_model_api,
        name='activate_model',
    ),
    path(
        'api/models/<int:model_id>/delete/', views.delete_model_api, name='delete_model'
    ),
]
