from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('index/', views.main_page, name='main_page'),
    path('index/detail/<int:id>/', views.detail_page, name='detail_page'),
    path('similarity/', views.similarity_page, name='similarity_page')  # Map '/index' to main page view

]

