from django.shortcuts import render, get_object_or_404
from .models import PubMedArticle
from project3.scripts.utils import *
from django.db.models import Q
import json
from django.http import JsonResponse


# Create your views here.

def main_page(request):
    available_years = get_available_years()
    selected_year = request.GET.get('selected_year', None)
    query = request.GET.get('query', '')
    suggestions = suggest_corrections(query,set(get_all_words()))

    results = []
    if query :
        results = PubMedArticle.objects.filter(
            Q(title__icontains=query) | Q(abstract__icontains=query)
            )  
        if selected_year != "None":
            results = results.filter(pubdate__startswith=selected_year) | results.filter(pubdate__lt=f"{selected_year}-01-01") 

    for result in results:
        result.title_highlighted = highlight_query(result.title, query)
        result.abstract_highlighted = highlight_query(result.abstract, query)   
        result.keyword = keyword_count(result,query) 

    results = sorted(results, key=lambda x: x.keyword, reverse=True)

    return render(request, 'main.html', {'query': query, 
                                         'suggestions': suggestions,
                                         'results': results,
                                         'available_years': available_years,
                                         'selected_year': selected_year,
                                         'results_count': len(results)
                                         })

def detail_page(request, id):
    query = request.GET.get('query', '') # Get the query parameter from the URL
    target = get_object_or_404(PubMedArticle, id=id) # Get the id parameter from the URL

    context = {
                'target': target,
                'query': query
    }

    return render(request, 'detail.html', context)

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

def query_page(request):
    query = request.GET.get('query', '')
    print(query)
    all_pubmed = PubMedArticle.objects.all()
    all_abstracts = [each.abstract for each in all_pubmed]
    ranked_abstracts = ""
    if query:
        ranked_abstracts = rank_abstract(query, all_abstracts)
    # for abstract,score in ranked_abstracts:
    #     print(score,"----",abstract[:50])

    results = []
    for abstract, score in ranked_abstracts:
       target = PubMedArticle.objects.filter(abstract=abstract).first()  
       if target: 
            # Add the article details and the score to the results list
            results.append({
                'id': target.id,
                'pmid': target.pmid,
                'title': target.title,
                'abstract': target.abstract,
                'score': score
            })
    
    context = { 'results': results,
                'ranked_abstracts': ranked_abstracts,
                'results_count': len(ranked_abstracts)
    }
    return render(request, 'query.html',context)

def rank_sentence_page(request, id):
    query = request.GET.get('query', '')
    target = get_object_or_404(PubMedArticle, id=id)
    count_sentences, count_words, count_characters, count_ascii, count_non_ascii = statistic_count(target.abstract)
    context = {
                'target': target,
                'query': query,
                'count_sentences': count_sentences,
                'count_words': count_words,
                'count_characters': count_characters,
                'count_ascii': count_ascii,
                'count_non_ascii': count_non_ascii,
    }

    return render(request, 'rank_sentence.html', context)
