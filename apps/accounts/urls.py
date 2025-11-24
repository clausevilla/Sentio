from django.urls import path

from apps.accounts import views

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
]
