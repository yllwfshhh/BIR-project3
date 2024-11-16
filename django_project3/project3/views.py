from django.shortcuts import render, get_object_or_404
from .models import PubMedArticle
from project3.scripts.utils import *
from django.db.models import Q
import json
from django.http import JsonResponse


# Create your views here.

def main_page(request):
    query = request.GET.get('query', '')
    results = []
    if query:
        results = PubMedArticle.objects.filter(
            Q(title__icontains=query) | Q(abstract__icontains=query)
            )  
    for result in results:
        result.title_highlighted = highlight_query(result.title, query)
        result.abstract_highlighted = highlight_query(result.abstract, query)   
        title_count = len(re.findall(re.escape(query), result.title, re.IGNORECASE))
        abstract_count = len(re.findall(re.escape(query), result.abstract, re.IGNORECASE))
        result.keyword_count = title_count + abstract_count
    results = sorted(results, key=lambda x: x.keyword_count, reverse=True)

    return render(request, 'main.html', {'query': query, 
                                         'results': results,
                                         'results_count': len(results)
                                         })

def detail_page(request, id):
    query = request.GET.get('query', '')
    target = get_object_or_404(PubMedArticle, id=id)
    target.title_highlighted = highlight_query(target.title, query)
    target.abstract_highlighted = highlight_query(target.abstract, query)

    return render(request, 'detail.html', {'target': target, 'query': query})


def similarity_page(request, word1=None, word2=None):

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
        cbow_words = predict_cbow_word(word1, 20)
        sg_words = predict_sg_word(word1, 20)

    context = {
        'top_k_words': top_k_words,
        'cbow_score': cbow_score,  
        'sg_score': sg_score,     
        'cbow_words': cbow_words,  
        'sg_words': sg_words,     
        'word1': word1,
        'word2': word2,
    }
    
    return render(request, 'similarity.html', context)
