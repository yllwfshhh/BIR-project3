import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from database import get_abstract_sentences
import string

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

if __name__ == "__main__":
    db_name = "../pubmed.db"
    documents = get_abstract_sentences(db_name,"")
    preprocessed_documents = preprocess_text(documents)
    model = Word2Vec(sentences=preprocessed_documents, 
                     vector_size=100, 
                     window=2, 
                     min_count=1, 
                     sg=1)
    # Similarity between words
    word1, word2 = 'enterovirus', 'children'
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
    print(f"Predicted surrounding words for '{word1}': {predicted_surrounding_words}")