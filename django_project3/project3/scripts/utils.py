
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from project3.models import PubMedArticle
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from django.utils.safestring import mark_safe
import re
import Levenshtein


## base
def get_all_sentences():
    # Retrieve all abstracts from the database
    abstracts = PubMedArticle.objects.values_list('abstract', flat=True)  # Get all abstracts
    all_sentences = []
    for abstract in abstracts:
        if abstract:  # Ensure the abstract is not empty
            sentences = sent_tokenize(abstract)  # Split abstract into sentences
            all_sentences.extend(sentences)  # Add sentences to the list

    return all_sentences

def get_all_words():
    # Retrieve all abstracts from the database
    abstracts = PubMedArticle.objects.values_list('abstract', flat=True)
    abstract_lists =  preprocess_text([abstract for abstract in abstracts if abstract is not None])
    all_words = []
    for word_list in abstract_lists:
        all_words.extend(word_list)

    return all_words

def highlight_query(text, query):
    if query:
        highlighted = re.sub(f"({re.escape(query)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
        return mark_safe(highlighted)
    return text

## project1
def keyword_count(target,query):
    title_count = len(re.findall(re.escape(query), target.title, re.IGNORECASE))
    abstract_count = len(re.findall(re.escape(query), target.abstract, re.IGNORECASE))
    return title_count + abstract_count

def statistic_count(text):
    count_sentences = len(sent_tokenize(text))
    count_words = len(text.split())
    count_characters = len(text)
    count_ascii = 0
    for char in text:
        if ord(char) < 128:
            count_ascii += 1
    count_non_ascii = len(text)-count_ascii
    
    return count_sentences, count_words, count_characters, count_ascii, count_non_ascii

def get_available_years():
    years = PubMedArticle.objects.values('pubdate').distinct()
    unique_years = set()
    for each in years:
        year = str(each['pubdate'])[:4]
        if year[0] == '2' :
            unique_years.add(year)
    return sorted(unique_years)

## project2
def top_k_frequency(words,k):
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_word_freq[:k]

def suggest_corrections(query, candidates):
    suggestions = []
    for candidate in candidates:
        distance = Levenshtein.distance(query.lower(), candidate.lower())
        suggestions.append((candidate, distance))
    
    suggestions.sort(key=lambda x: x[1])
    return [s[0] for s in suggestions[:5]] 

# Preprocess the text data
def preprocess_text(documents):
    stop_words = set(stopwords.words('english'))
    preprocessed_documents = []
    for doc in documents:
        tokens = word_tokenize(doc)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        preprocessed_documents.append(tokens)
    return preprocessed_documents   # ['I','love','you']

## project3
def check_similarity(word1, word2):
    if word1 in CBOWmodel.wv.key_to_index and word2 in CBOWmodel.wv.key_to_index:
        cbow_score = CBOWmodel.wv.similarity(word1, word2)
        sg_score = SGmodel.wv.similarity(word1, word2)
        return cbow_score, sg_score
    return 0.0,0.0

def predict_cbow_word(keyword,k):
    if keyword in CBOWmodel.wv:
        similar_words = CBOWmodel.wv.most_similar(keyword, topn=k)
        print(k)
        words = [word for word, score in similar_words]
        words.insert(0,keyword)
        plot_embeddings(words,CBOWmodel,'project3/static/img/cbow_embeddings.png')
        return words
    return

def predict_sg_word(keyword,k):
    if keyword in CBOWmodel.wv:
        similar_words = SGmodel.wv.most_similar(keyword, topn=k)
        words = [word for word, score in similar_words]
        words.insert(0,keyword)
        plot_embeddings(words,SGmodel,'project3/static/img/sg_embeddings.png')
        return words
    return

def plot_embeddings(words,model,filename):
    vectors = model.wv[words]
    n_samples = len(vectors)
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    vectors_2d = tsne.fit_transform(vectors)
    

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = vectors_2d[i, :]
        plt.scatter(x, y)
        plt.text(x+0.1, y+0.1, word, fontsize=18)
    plt.title(f"t-SNE visualization of word vectors", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=16)
    plt.ylabel("t-SNE Dimension 2", fontsize=16)
    plt.grid()
    plt.savefig(filename,format='png')

CBOWmodel = Word2Vec(sentences = preprocess_text(get_all_sentences()), 
                     vector_size=100, 
                     window=2, 
                     min_count=1, 
                     sg=0)

SGmodel = Word2Vec(sentences=preprocess_text(get_all_sentences()), 
                     vector_size=100, 
                     window=2, 
                     min_count=1, 
                     sg=1)

## project4
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_abstract(query,abstracts):
    # Step 1: Compute TF-IDF scores
    all_text = [query] + abstracts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    query_vector = tfidf_matrix[0]  # First row corresponds to the query
    abstract_vectors = tfidf_matrix[1:] 

    # Step 2: Compute cosine similarity
    similarity_scores = cosine_similarity(query_vector, abstract_vectors).flatten()

    # Step 3: Rank abstracts by similarity to query
    ranked_indices = similarity_scores.argsort()[::-1]
    ranked_abstracts = [(abstracts[i], similarity_scores[i]) for i in ranked_indices if similarity_scores[i] > 0.05]
    

    # print("Ranked Abstracts and Sentences:")
    # for idx, (abstract, score) in enumerate(ranked_abstracts):
    #     print(f"\nAbstract (Score: {score:.4f}): {abstract}")
    #     # Split abstract into sentences
    #     sentences = sent_tokenize(abstract)
    #     # Compute weighted CBOW for sentences
    #     for i, sentence in enumerate(sentences):
    #         sentence_vector = compute_weighted_cbow(sentence, tfidf_weights[idx + 1], tfidf_feature_names, CBOWmodel)
    #         sentence_score = cosine_similarity([query_vector], [sentence_vector]).flatten()[0]
    #         print(f"  Sentence {i + 1}: {sentence} (Score: {sentence_score:.4f})")    

    return ranked_abstracts

# Function to compute weighted CBOW
def compute_weighted_cbow(text, tfidf_weights, feature_names, model):
    words = text.lower().split()
    weighted_embeddings = []
    
    for word in words:
        if word in model.wv and word in feature_names:
            # Get TF-IDF weight
            word_index = np.where(feature_names == word)[0]
            if word_index.size > 0:
                weight = tfidf_weights[word_index[0]]
                weighted_embeddings.append(model.wv[word] * weight)
    
    if not weighted_embeddings:
        return np.zeros(model.vector_size)  # Return zero vector if no valid words
    return np.mean(weighted_embeddings, axis=0)
