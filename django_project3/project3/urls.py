from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('index/', views.main_page, name='main_page'),  # Map '/index' to main page view
    path('check-similarity/', similarity_view, name='check_similarity'),
    path('predict-cbow/', predict_cbow_view, name='predict-cbow'),
    path('predict-sg/', predict_sg_view, name='predict-sg')
]

