from django.urls import path

from apps.predictions import views

app_name = 'predictions'

urlpatterns = [
    path('input', views.input_view, name='input'),
    path('result', views.result_view, name='result'),
]
