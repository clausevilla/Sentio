# Author: Marcus Berggren, Lian Shi

from django.shortcuts import render
from django.urls import path

from apps.accounts import views


def test_error_page(request):
    return render(
        request,
        'accounts/error.html',
        {'message': 'This is a test error message!', 'retry_url': '/'},
        status=503,
    )


app_name = 'accounts'

urlpatterns = [
    # Authentication URLs
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    # User profile URLs
    path('profile/', views.profile_view, name='profile'),
    path('history/', views.history_view, name='history'),
    # API endpoints for profile features
    path('api/change-password/', views.change_password_api, name='change_password_api'),
    path('api/delete-all-data/', views.delete_all_data_api, name='delete_all_data_api'),
    path('api/delete-account/', views.delete_account_api, name='delete_account_api'),
    path('api/export-data/', views.export_data_api, name='export_data_api'),
    # API endpoint for chart data
    path('api/chart-data/', views.chart_data_api, name='chart_data_api'),
    # API endpoint for deleting a single analysis
    path(
        'api/delete-analysis/<int:analysis_id>/',
        views.delete_analysis_api,
        name='delete_analysis_api',
    ),
    # Registration validation APIs (real-time validation)
    path('api/check-username/', views.check_username_api, name='check_username_api'),
    path('api/check-email/', views.check_email_api, name='check_email_api'),
    path('api/register/', views.register_api, name='register_api'),
    # Consent and privacy URLs
    path('consent/', views.consent_view, name='consent'),
    path('privacy/', views.privacy_policy_view, name='privacy_policy'),
    path('test-error/', test_error_page, name='test_error'),  # For test only
]
