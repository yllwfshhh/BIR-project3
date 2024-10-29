import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import string

# Download NLTK data files (if not already installed)
nltk.download('punkt')
nltk.download('stopwords')

# Sample data: list of text documents
documents = [
    "Natural language processing is a fascinating field of study.",
    "Word embeddings are helpful for understanding text data.",
    "Gensim provides tools for training word2vec models efficiently.",
    "Machine learning algorithms learn from data."
]

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

# Preprocessed documents
preprocessed_documents = preprocess_text(documents)

# Train Word2Vec model with CBOW (sg=0)
model = Word2Vec(sentences=preprocessed_documents, vector_size=100, window=2, min_count=1, sg=0)

# Get vector for a word
word = 'language'
if word in model.wv:
    print(f"Vector for '{word}': {model.wv[word]}")
else:
    print(f"Word '{word}' not in vocabulary")

# Similarity between words
word1, word2 = 'language', 'text'
if word1 in model.wv and word2 in model.wv:
    similarity = model.wv.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity}")
else:
    print(f"One of the words ('{word1}' or '{word2}') is not in vocabulary.")
