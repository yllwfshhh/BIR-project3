from django.shortcuts import render
from .models import PubMedArticle
from project3.scripts.utils import *
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse


# Create your views here.
def main_page(request):
    top_k_words = top_k_frequency(get_all_words(),10)
    context = {
        'top_k_words' : top_k_words,
    }
    return render(request, 'main.html',context)

@csrf_exempt
def similarity_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        word1 = data.get('word1')
        word2 = data.get('word2')
        similarity_score = check_similarity(word1, word2)    
        print(similarity_score)
        return JsonResponse({'similarity': float(similarity_score)})

@csrf_exempt
def predict_cbow_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        word1 = data.get('word1')
        word2 = data.get('word2')
        middle_word = predict_middle_word([word1, word2])    
        print("Middle word:",middle_word)
        return JsonResponse({'middle_word': middle_word})
    
@csrf_exempt
def predict_sg_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        word1 = data.get('word1')
        similar_words = predict_similar_word(word1,5)   
        print("Similar words:",similar_words)
        return JsonResponse({'similar_word': similar_words})