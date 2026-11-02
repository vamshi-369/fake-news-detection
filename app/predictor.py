import os
import sys
import pickle
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_training")))
from prepare_data import clean_text, TextTokenizer

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class NewsLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim=128, hidden_dim1=64, hidden_dim2=32, dropout=0.3):
            super(NewsLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True, bidirectional=True)
            self.dropout1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True)
            self.dropout2 = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim2 * 2, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            embedded = self.embedding(x)
            out1, _ = self.lstm1(embedded)
            out1 = self.dropout1(out1)
            out2, _ = self.lstm2(out1)
            out2 = self.dropout2(out2)
            pooled = torch.mean(out2, dim=1)
            dense1 = self.relu(self.fc1(pooled))
            logits = self.fc2(dense1)
            prob = self.sigmoid(logits)
            return prob.squeeze(1)


class Predictor:
    """Thread-safe Singleton Predictor to avoid reloading model per-request."""
    _instance = None

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

        tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
        pth_path = os.path.join(model_dir, "model.pth")
        pkl_path = os.path.join(model_dir, "model.pkl")

        if os.path.exists(tokenizer_path):
            self.tokenizer = TextTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = None

        self.use_torch = False
        self.model = None

        if TORCH_AVAILABLE and os.path.exists(pth_path):
            try:
                vocab_size = len(self.tokenizer.word_index) + 1 if self.tokenizer else 5000
                self.model = NewsLSTM(vocab_size=vocab_size)
                self.model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
                self.model.eval()
                self.use_torch = True
                print("[Predictor] Successfully loaded PyTorch LSTM Model!")
            except Exception as e:
                print(f"[Predictor] Torch load exception ({e}). Falling back to neural model payload.")

        if not self.use_torch and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                payload = pickle.load(f)
                self.vectorizer = payload['vectorizer']
                self.model = payload['model']
                print("[Predictor] Successfully loaded Neural Classifier Payload!")

        if self.model is None:
            raise FileNotFoundError(f"No trained model artifact found in {model_dir}. Please run train_lstm.py first.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, text: str, max_len: int = 300):
        if not text or not text.strip():
            return "Real", 0.50, 0.50, 0

        clean = clean_text(text, remove_stopwords=True)
        token_count = len(text.split())

        if self.use_torch and self.tokenizer:
            seq = self.tokenizer.texts_to_sequences([clean])
            padded = self.tokenizer.pad_sequences(seq, maxlen=max_len)
            tensor_in = torch.tensor(padded, dtype=torch.long)
            with torch.no_grad():
                prob_val = float(self.model(tensor_in).numpy()[0])
        else:
            vec = self.vectorizer.transform([clean])
            prob_val = float(self.model.predict_proba(vec)[0][1])

        label = "Fake" if prob_val > 0.5 else "Real"
        confidence = prob_val if label == "Fake" else (1.0 - prob_val)
        confidence = max(confidence, 0.52)

        return label, round(float(confidence), 4), round(float(prob_val), 4), token_count


def predict_news(text: str, max_len: int = 300):
    predictor = Predictor.get_instance()
    return predictor.predict(text, max_len=max_len)
