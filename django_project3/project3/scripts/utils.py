import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from project3.models import PubMedArticle
from gensim.models import Word2Vec


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
        # Tokenize, remove punctuation, and convert to lowercase
        tokens = word_tokenize(doc)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
        preprocessed_documents.append(tokens)
    return preprocessed_documents


def check_similarity(word1, word2):


    if word1 in CBOWmodel.wv.key_to_index and word2 in CBOWmodel.wv.key_to_index:
        return CBOWmodel.wv.similarity(word1, word2)
    return None

def predict_middle_word(context_words):
    
    context_embeddings = [CBOWmodel.wv[word] for word in context_words if word in CBOWmodel.wv]
    if not context_embeddings:
        return "Context words not found in vocabulary."
    context_vector = sum(context_embeddings) / len(context_embeddings)
    predicted_word = CBOWmodel.wv.similar_by_vector(context_vector, topn=1)[0][0]
    return predicted_word


def predict_similar_word(target_word,topn):
        if target_word in SGmodel.wv:
            surrounding_words = SGmodel.wv.most_similar(target_word, topn=topn)
            surrounding_words = [word for word, _ in surrounding_words]
            return surrounding_words
        else:
            return f"'{target_word}' is not in the vocabulary."

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