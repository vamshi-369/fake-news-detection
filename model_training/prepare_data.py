import re
import pickle
from collections import Counter
import numpy as np

# Basic English stopwords list
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
])

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Preprocesses raw article/headline text."""
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    
    # 4. Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # 5. Keep only alphabetic characters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 6. Normalize whitespace
    words = text.split()
    
    # 7. Optionally remove stopwords
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]
        
    return " ".join(words)


class TextTokenizer:
    """Custom Tokenizer matching Keras-style interface for NLP sequence creation."""
    def __init__(self, num_words=10000, oov_token="<UNK>", pad_token="<PAD>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.word_index = {pad_token: 0, oov_token: 1}
        self.index_word = {0: pad_token, 1: oov_token}
        self.word_counts = Counter()

    def fit_on_texts(self, texts):
        """Builds vocabulary from a list of clean text strings."""
        for text in texts:
            words = text.split()
            self.word_counts.update(words)

        # Sort words by frequency
        most_common = self.word_counts.most_common(self.num_words - 2)
        idx = 2
        for word, _ in most_common:
            self.word_index[word] = idx
            self.index_word[idx] = word
            idx += 1

    def texts_to_sequences(self, texts):
        """Converts text strings to integer token sequences."""
        sequences = []
        oov_idx = self.word_index.get(self.oov_token, 1)
        for text in texts:
            words = text.split()
            seq = [self.word_index.get(w, oov_idx) for w in words]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences, maxlen=300, padding='post', truncating='post'):
        """Pads or truncates integer sequences to maxlen."""
        padded = []
        for seq in sequences:
            if len(seq) > maxlen:
                if truncating == 'post':
                    seq = seq[:maxlen]
                else:
                    seq = seq[-maxlen:]
            elif len(seq) < maxlen:
                pad_len = maxlen - len(seq)
                if padding == 'post':
                    seq = seq + [0] * pad_len
                else:
                    seq = [0] * pad_len + seq
            padded.append(seq)
        return np.array(padded, dtype=np.int64)

    def save(self, filepath):
        """Saves tokenizer state to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Loads tokenizer state from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
