
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from project3.models import PubMedArticle
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np




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

def top_k_frequency(words,k):
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_word_freq[:k]

# Preprocess the text data
def preprocess_text(documents):
    stop_words = set(stopwords.words('english'))
    preprocessed_documents = []
    for doc in documents:
        tokens = word_tokenize(doc)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        preprocessed_documents.append(tokens)
    return preprocessed_documents   # ['I','love','you']


def check_similarity(word1, word2):
    if word1 in CBOWmodel.wv.key_to_index and word2 in CBOWmodel.wv.key_to_index:
        cbow_score = CBOWmodel.wv.similarity(word1, word2)
        sg_score = SGmodel.wv.similarity(word1, word2)
        return cbow_score, sg_score
    return 0.0,0.0

def predict_cbow_word(keyword,k):
    
    similar_words = CBOWmodel.wv.most_similar(keyword, topn=k)
    words = [word for word, score in similar_words]
    words.insert(0,keyword)
    plot_embeddings(words,CBOWmodel,'project3/static/img/cbow_embeddings.png')
    return words

def predict_sg_word(keyword,k):
    similar_words = SGmodel.wv.most_similar(keyword, topn=k)
    words = [word for word, score in similar_words]
    words.insert(0,keyword)
    plot_embeddings(words,SGmodel,'project3/static/img/sg_embeddings.png')
    return words

def plot_embeddings(words,model,filename):
    vectors = model.wv[words]
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1],), fontsize=18)
    plt.title(f"Similar words to '{words[0]}' in 2D space", fontsize=16)
    plt.xlabel("PCA Component 1", fontsize=16)
    plt.ylabel("PCA Component 2", fontsize=16)
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