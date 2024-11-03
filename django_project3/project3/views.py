from django.shortcuts import render
from .models import PubMedArticle
from project3.scripts.utils import *
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse


# Create your views here.
def main_page(request, word1=None, word2=None):

    if request.method == 'POST':
        word1 = request.POST.get('word1', word1)
        word2 = request.POST.get('word2', word2)

    top_k_words = top_k_frequency(get_all_words(), 10)
    cbow_score = None
    sg_score = None
    cbow_words = []
    sg_words = []

    if request.method == 'POST' and word1 and word2:
        cbow_score, sg_score = check_similarity(word1, word2)
        cbow_words = predict_cbow_word(word1, 10)
        sg_words = predict_sg_word(word1, 10)

    context = {
        'top_k_words': top_k_words,
        'cbow_score': cbow_score,  
        'sg_score': sg_score,     
        'cbow_words': cbow_words,  
        'sg_words': sg_words,     
        'word1': word1,
        'word2': word2,
    }
    
    return render(request, 'main.html', context)


# @csrf_exempt
# def get_input(request):
#     if request.method == 'POST':
#         word1 = request.POST.get('word1')
#         word2 = request.POST.get('word2')
#         return main_page(request, word1, word2)
#     return main_page(request)

    

@csrf_exempt
def similar_word_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        word1 = data.get('word1')
        cbow_words = predict_cbow_word(word1,10)   
        sg_words = predict_sg_word(word1,10)
        print(cbow_words)  
        print(sg_words)
       
        return JsonResponse({
            'cbow_words': cbow_words,
            'sg_words':sg_words
            })
    
