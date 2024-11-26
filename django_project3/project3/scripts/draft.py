import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import os
import sys
print(sys.path)
import django
sys.path.append('/home/project3/django_project3')  # Use the absolute path to `django_project3`
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_project3.settings')
django.setup()
from project3.models import PubMedArticle


def get_all_sentences():
    # Retrieve all abstracts from the database
    abstracts = PubMedArticle.objects.values_list('abstract', flat=True)  # Get all abstracts
    all_sentences = []
    for abstract in abstracts:
        if abstract:  # Ensure the abstract is not empty
            sentences = sent_tokenize(abstract)  # Split abstract into sentences
            all_sentences.extend(sentences)  # Add sentences to the list

    return all_sentences

def preprocess_text(documents):
    # stop_words = set(stopwords.words('english'))
    preprocessed_documents = []
    for doc in documents:
        tokens = word_tokenize(doc)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower()]
        # tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        preprocessed_documents.append(tokens)
    return preprocessed_documents   # ['I','love','you']

# Load pretrained Word2Vec embeddings
sentences = get_all_sentences()

# Example query and abstracts
query = "will depression cause aging"
abstracts = [
    "With aging, there are normative changes to sleep physiology and circadian rhythmicity that may predispose older adults to sleep deficiency, whereas many health-related and psychosocial/behavioral factors may precipitate sleep deficiency. In this article, we describe age-related changes to sleep and describe how the health-related and psychosocial/behavioral factors typical of aging may converge in older adults to increase the risk for sleep deficiency. Next, we review the consequences of sleep deficiency in older adults, focusing specifically on important age-related outcomes, including mortality, cognition, depression, and physical function. Finally, we review treatments for sleep deficiency, highlighting safe and effective nonpharmacologic interventions.",
    "I like to eat apple. Apple is sweet. I dont like kiwi",
    "The results of this study suggest that a theater intervention for the older adults may be effective in preventing and improving depression and physical frailty in old age."
]
## with TFIDF
# Step 1: Compute TF-IDF scores
all_text = [query] + abstracts
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)
query_vector = tfidf_matrix[0]  # First row corresponds to the query
abstract_vectors = tfidf_matrix[1:]  

# Step 2: Compute cosine similarity
similarity_scores = cosine_similarity(query_vector, abstract_vectors).flatten()

# Rank abstracts by similarity to query
ranked_indices = similarity_scores.argsort()[::-1]
ranked_abstracts = [(abstracts[i], similarity_scores[i]) for i in ranked_indices ]

# Step 3: Rank sentences within each abstract
print("Ranked Abstracts and Sentences:")
for idx, (abstract, score) in enumerate(ranked_abstracts):
    print(f"\nAbstract (Score: {score:.4f}): {abstract}")
    
## CBOW & TFIDF
CBOWmodel = Word2Vec(sentences = preprocess_text(get_all_sentences()), 
                     vector_size=100, 
                     window=2, 
                     min_count=1, 
                     sg=0)


# Step1: TFIDF
abstract_sentences = sent_tokenize(abstracts[0])
tfidf_vectorizer = TfidfVectorizer(stop_words=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(abstract_sentences)
query_tfidf = tfidf_vectorizer.transform([query])
print("query tfidf:",query_tfidf)
print("sentences_tfid:",tfidf_matrix)
# Step2: CBOW


def get_word_embedding(word):
    if word in CBOWmodel.wv:
        return CBOWmodel.wv[word]
    else:
        return np.zeros(100)

def get_sentence_embedding(sentence):
    words = sentence.lower().split()
    sentence_embedding_sum = np.sum([get_word_embedding(word) for word in words], axis=0)
    sentence_embedding_mean = sentence_embedding_sum / len(words) if len(words) > 0 else np.zeros(100)  # Avoid division by zero
    return sentence_embedding_mean

query_cbow = get_sentence_embedding(query)
sentence_cbow_vectors = [get_sentence_embedding(sentence) for sentence in abstract_sentences]


# Step 3: Calculate relevance scores
tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
cbow_scores = [cosine_similarity([query_cbow], [sent_vec]).flatten()[0] for sent_vec in sentence_cbow_vectors]

# Combine scores
alpha = 0.7  # Adjust to prioritize TF-IDF or CBOW
final_scores = alpha * tfidf_scores + (1 - alpha) * np.array(cbow_scores)

ranked_sentences = sorted(zip(abstract_sentences, final_scores), key=lambda x: x[1], reverse=True)
for sentence, score in ranked_sentences:
    print(f"Sentence: {sentence} | Score: {score:.4f}\n")
