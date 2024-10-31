import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from database import get_abstract_sentences,get_all_words
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


# Download NLTK data files (if not already installed)
# nltk.download('punkt')
# nltk.download('stopwords')

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

# Predict middle word
def predict_middle_word(context_words, model):
    
    context_embeddings = [model.wv[word] for word in context_words if word in model.wv]
    if not context_embeddings:
        return "Context words not found in vocabulary."
    # Calculate the average vector of the context words
    context_vector = sum(context_embeddings) / len(context_embeddings)
    # Find the word in the vocabulary closest to the context vector
    predicted_word = model.wv.similar_by_vector(context_vector, topn=1)[0][0]
    return predicted_word

def predict_surrounding_words(target_word, model, topn=5):
    if target_word in model.wv:
        surrounding_words = model.wv.similar_by_word(target_word, topn=topn)
        return [word for word, similarity in surrounding_words]
    else:
        return f"'{target_word}' is not in the vocabulary."
    
def top_k_frequency(words,k):
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_word_freq[:k]

if __name__ == "__main__":
    db_name = "./pubmed.db"
    documents = get_abstract_sentences(db_name,"")
    words = get_all_words(db_name,"")
    top_k_words = top_k_frequency(words,10)
    word_lists = []

    preprocessed_documents = preprocess_text(documents)

    model = Word2Vec(sentences=preprocessed_documents, 
                     vector_size=100, 
                     window=2, 
                     min_count=1, 
                     sg=0)
    # Similarity between words
    word1, word2 = 'enterovirus', 'mice'
    if word1 in model.wv and word2 in model.wv:
        similarity = model.wv.similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity}")
    elif word1 in model.wv:
        print(f"'{word2}' is not in vocabulary.")
    elif word2 in model.wv:
        print(f"'{word1}' is not in vocabulary.")
    else:
        print(f"Both '{word1}' and '{word2}' are not in vocabulary.")

    # Example: predict middle word with context
    context_words = [word1,word2]
    # predicted_word = predict_middle_word(context_words, model)
    # print(f"Predicted middle word for context {context_words}: {predicted_word}")

    predicted_surrounding_words = predict_surrounding_words(word1, model)
    # print(f"Predicted surrounding words for '{word1}': {predicted_surrounding_words}")

    word_vectors = model.wv
    words = list(word_vectors.index_to_key[:10])  # Get the words
    vectors = word_vectors[words]  # Get the vectors for the words
  
    # Use t-SNE to reduce dimensions to 2D
    perplexity_value = min(5, len(words) - 1)  
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create a scatter plot
    plt.figure()
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

    # Add labels for each point
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8)

    plt.title("Word Embeddings Visualization")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid()
    plt.savefig('word_embeddings_visualization.png', format='png', bbox_inches='tight')