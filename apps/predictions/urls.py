# Author: Lian Shi, Karl Byland

from django.urls import path

from apps.predictions import views

from .views import strings

app_name = 'predictions'

urlpatterns = [
    path('input', views.input_view, name='input'),
    path('result', views.result_view, name='result'),
    path('api/strings/', strings, name='strings'),
]
