from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('index/', views.main_page, name='main_page'),  # Map '/index' to main page view
    path('get_input/', get_input, name='get_input'),
    path('similar_word/', similar_word_view, name='similar_word')
]

