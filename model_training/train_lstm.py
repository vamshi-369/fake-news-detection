import os
import sys
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_data import clean_text, TextTokenizer

np.random.seed(42)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(42)
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"[Notice] PyTorch unavailable ({e}). Using Deep Neural Network Classifier engine.")

DATASET = [
    # REAL NEWS (Label 0)
    ("NASA's James Webb Space Telescope discovers ancient galaxy forming stars at record speed.", 0),
    ("Federal Reserve maintains benchmark interest rates amidst steady inflation indicators.", 0),
    ("World Health Organization releases updated global guidelines for cardiovascular health.", 0),
    ("Scientists report breakthrough in solid-state lithium battery efficiency and energy density.", 0),
    ("United Nations climate conference reaches historic agreement on international carbon offset tracking.", 0),
    ("European Union passes comprehensive regulatory framework governing artificial intelligence models.", 0),
    ("Archaeologists uncover 3000 year old merchant vessel in the Mediterranean sea floor.", 0),
    ("Global semiconductor production expands as new fabrication facilities open across Europe and Asia.", 0),
    ("Cancer research institute announces promising phase three clinical trial results for targeted immunotherapy.", 0),
    ("Stock market indexes close higher following strong earnings reports from tech and retail sectors.", 0),
    ("Japan launches advanced meteorological satellite to enhance hurricane tracking and weather forecasting.", 0),
    ("Renewable energy generation surpasses coal usage across European power grids for second quarter.", 0),
    ("Department of Transportation allocates federal grants for high-speed rail corridor infrastructure.", 0),
    ("Marine biologists document recovery of coral reef ecosystems in protected oceanic marine reserves.", 0),
    ("Global literacy rates show steady improvement over the past decade according to UNESCO report.", 0),
    ("Electric vehicle adoption reaches milestone percentage in new car sales nationwide.", 0),
    ("Researchers develop low-cost water filtration membrane using recycled agricultural waste materials.", 0),
    ("Central bank announces digital currency pilot project with leading financial institutions.", 0),
    ("International Space Station completes orbit adjustment maneuver ahead of cargo supply mission.", 0),
    ("Agriculture department publishes seasonal crop yield projections indicating record wheat harvest.", 0),
    ("Medical researchers identify genetic marker linked to early onset Alzheimer's disease risk reduction.", 0),
    ("Supreme Court issues landmark ruling regarding consumer data privacy and digital search warrants.", 0),
    ("Civil engineers complete retrofit of major suspension bridge using carbon fiber reinforcement.", 0),
    ("Public university system expands tuition-free grant program for low-income undergraduate students.", 0),
    ("Oceanographic institute records decline in plastic microparticle concentration near coastal preserves.", 0),
    ("Global aviation industry implements new flight path optimization algorithms to reduce jet fuel emissions.", 0),
    ("Bipartisan senate committee approves funding bill for municipal clean drinking water systems.", 0),
    ("Epidemiology survey indicates seasonal influenza transmission rates remain within historical averages.", 0),
    ("Seismologists deploy real-time sensor network along major tectonic fault lines to improve warning lead time.", 0),
    ("National science foundation awards research fellowship grants for quantum computing error correction.", 0),

    # FAKE NEWS (Label 1)
    ("Secret government lab creates weather control machine that causes devastating hurricanes on demand!", 1),
    ("Miracle fruit discovered in South America cures all diseases overnight doctors are trying to ban it!", 1),
    ("Leaked documents confirm moon landing was staged in Hollywood basement using secret laser technology!", 1),
    ("Drinking boiled ocean water turns grey hair pitch black in three days according to underground scientists!", 1),
    ("Billionaire secret society installs microchips in tap water to control human brainwaves via cell towers!", 1),
    ("Scientists admit earth is actually hollow and inhabited by ancient race of giant reptilian beings!", 1),
    ("5G cell towers confirmed to transmit alien mind-altering frequencies during solar eclipses!", 1),
    ("Shocking scandal revealing famous celebrity replaced by android clone after disappearing last month!", 1),
    ("Government planning to replace all physical money with microchip implants mandatory by next week!", 1),
    ("Ancient Mayan prophecy predicts sky will turn purple tomorrow morning as portals open worldwide!", 1),
    ("Doctors reveal eating raw onion slices overnight completely immunizes body against all viruses forever!", 1),
    ("Secret space force base discovered on far side of moon housing time-traveling interdimensional portals!", 1),
    ("Major soft drink company caught putting mind control chemicals in soda cans to force election outcomes!", 1),
    ("Shocking photo proves dinosaurs still living secretly in unexplored rainforest cavern network!", 1),
    ("Whistleblower leaks evidence that clouds are artificial synthetic steam created to hide alien ships!", 1),
    ("Eating three cloves of garlic every morning guarantees instant weight loss of twenty pounds per week!", 1),
    ("Leaked audio records top officials admitting internet will be shut off permanently starting midnight!", 1),
    ("Billionaire builds secret underground city under North Pole to survive incoming asteroid collision!", 1),
    ("Scientist arrested after inventing free zero-point energy generator that destroys power company profits!", 1),
    ("Drinking cucumber juice cured stage four illness in twenty four hours pharmaceutical companies hiding truth!", 1),
    ("Secret military project using weather satellites to melt polar ice caps for oceanfront real estate!", 1),
    ("Famous tech founder admits smartphone screens emit rays that read user thoughts and dreams!", 1),
    ("Bizarre sea monster washed ashore with human hands and wings according to anonymous viral video!", 1),
    ("Government ordering mandatory surrender of all household pets to prepare for upcoming alien arrival!", 1),
    ("Secret frequency played on radio stations causes listeners to forget their bank account passwords!", 1),
    ("Eating pure silver dust boosts human intelligence by two hundred percent according to viral blog!", 1),
    ("Hollywood studio caught using deepfake holograms of historical figures to vote in municipal elections!", 1),
    ("Mysterious pink snow falling over major city contains microscopic tracking dust from space station!", 1),
    ("Scientists accidentally open portal to parallel universe inside underground particle accelerator basement!", 1),
    ("Leaked memo reveals government hiding giant gold deposit found under national park to inflate debt!", 1)
]

def augment_data(dataset, multiplier=5):
    augmented = []
    for text, label in dataset:
        augmented.append((text, label))
        words = text.split()
        for i in range(multiplier):
            if len(words) > 4:
                idx = (i * 2 + 1) % len(words)
                new_words = words[:idx] + words[idx+1:]
                augmented.append((" ".join(new_words), label))
    return augmented


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


def train():
    print("=== Fake News Model Training Pipeline ===")
    
    data = augment_data(DATASET, multiplier=6)
    print(f"Total dataset samples: {len(data)}")
    
    raw_texts = [item[0] for item in data]
    labels = np.array([item[1] for item in data], dtype=np.float32)
    clean_texts = [clean_text(t, remove_stopwords=True) for t in raw_texts]
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    tokenizer_path = os.path.join(models_dir, "tokenizer.pkl")
    model_pth_path = os.path.join(models_dir, "model.pth")
    model_pkl_path = os.path.join(models_dir, "model.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")
    
    tokenizer = TextTokenizer(num_words=5000)
    tokenizer.fit_on_texts(clean_texts)
    tokenizer.save(tokenizer_path)
    
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        clean_texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    
    if TORCH_AVAILABLE:
        print("[Training] Executing PyTorch Bidirectional LSTM model training...")
        sequences_train = tokenizer.texts_to_sequences(X_train_txt)
        sequences_test = tokenizer.texts_to_sequences(X_test_txt)
        X_train = tokenizer.pad_sequences(sequences_train, maxlen=300)
        X_test = tokenizer.pad_sequences(sequences_test, maxlen=300)
        vocab_size = len(tokenizer.word_index) + 1
        
        model = NewsLSTM(vocab_size=vocab_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        
        for epoch in range(1, 15):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                preds = model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
                
        torch.save(model.state_dict(), model_pth_path)
        model.eval()
        with torch.no_grad():
            test_probs = model(torch.tensor(X_test)).numpy()
        test_preds = (test_probs > 0.5).astype(int)
    else:
        print("[Training] Executing Deep Learning Multi-Layer Perceptron (Neural Classifier)...")
        tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
        X_train_vec = tfidf.fit_transform(X_train_txt)
        X_test_vec = tfidf.transform(X_test_txt)
        
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=200, random_state=42)
        mlp.fit(X_train_vec, y_train)
        
        with open(model_pkl_path, 'wb') as f:
            pickle.dump({'vectorizer': tfidf, 'model': mlp}, f)
            
        test_probs = mlp.predict_proba(X_test_vec)[:, 1]
        test_preds = (test_probs > 0.5).astype(int)
        
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds).tolist()
    
    print("\n" + "=" * 45)
    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print(f"Test Precision : {prec * 100:.2f}%")
    print(f"Test Recall    : {rec * 100:.2f}%")
    print(f"Test F1 Score  : {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(f" TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f" FN: {cm[1][0]} | TP: {cm[1][1]}")
    print("=" * 45)
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "engine": "PyTorch Bi-LSTM" if TORCH_AVAILABLE else "Neural Network Classifier",
        "test_samples": len(y_test)
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("\nArtifacts successfully exported to target models directory.")

if __name__ == '__main__':
    train()
