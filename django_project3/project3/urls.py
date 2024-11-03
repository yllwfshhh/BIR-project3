from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('index/', views.main_page, name='main_page'),  # Map '/index' to main page view

]

