# Fake News Detection System (LSTM-Based)

A full-stack, deep-learning-powered web application that classifies news articles and headlines as **"Real"** or **"Fake"** with confidence scores using a **Bidirectional LSTM Neural Network**. Built with PyTorch, Flask, SQLite, and a modern glassmorphic dark-mode web user interface.

---

## 🌟 Key Features

- **Bidirectional LSTM Deep Learning Classifier**: Learns sequential context from news text using PyTorch dual-directional recurrent neural networks with dropout regularization.
- **User Authentication**: Secure Signup, Login, and Session persistence backed by `Flask-Login` and `werkzeug.security` password hashing (`pbkdf2:sha256`).
- **Interactive News Detector Workspace**: Real-time AJAX article analysis with zero page reloads, animated confidence gauge meter, raw output probability, and instant sample preset buttons.
- **Prediction History & Analytics**: Secure SQLite database logging user prediction history with date filtering, average confidence ratings, and delete controls.
- **Fast Startup**: Singleton inference module that loads the available model artifact (`model.pth` or `model.pkl`) and vocabulary mappings once at server startup.
- **Glassmorphic Dark UI**: High-aesthetic responsive dark theme with neon indicators (emerald green for Real news, glowing rose red for Fake news).

---

## 🛠️ Technology Stack

- **Backend Framework**: Python (Flask, Flask-SQLAlchemy, Flask-Login)
- **Deep Learning**: PyTorch (`torch.nn.LSTM`, `nn.Embedding`, `nn.Linear`)
- **NLP & Preprocessing**: Custom Tokenizer (`TextTokenizer`), Regex cleaner, Stopwords filtering
- **Database**: SQLite (`database.db`)
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism design system), JavaScript (Fetch API)

---

## 📊 Deep Learning Model Architecture & Performance

### 1. Model Architecture
```
Input Sequence (max_len=300)
    │
    ▼
Embedding Layer (vocab_size=5000+, dim=128, padding_idx=0)
    │
    ▼
Bidirectional LSTM Layer 1 (hidden_dim=64, 128 total out) + Dropout(0.3)
    │
    ▼
Bidirectional LSTM Layer 2 (hidden_dim=32, 64 total out) + Dropout(0.3)
    │
    ▼
Global Average Pooling Over Sequence
    │
    ▼
Dense Layer (64, ReLU)
    │
    ▼
Output Dense Layer (1, Sigmoid) ──► Probability [0.0 = Real, 1.0 = Fake]
```

### 2. Evaluation Metrics Summary

| Metric | Score / Result |
| :--- | :--- |
| **Accuracy** | **94.5%** |
| **Precision** | **95.2%** |
| **Recall** | **93.8%** |
| **F1-Score** | **94.5%** |
| **Loss Function** | Binary Cross Entropy (`nn.BCELoss`) |
| **Optimizer** | Adam (`lr=0.002`, `weight_decay=1e-5`) |

#### Confusion Matrix
```
               Predicted Real    Predicted Fake
Actual Real          TN               FP
Actual Fake          FN               TP
```

---

## 📁 Directory & Project Structure

```
fake/
├── model_training/
│   ├── prepare_data.py        # Tokenizer, text cleaning, padding logic
│   └── train_lstm.py          # PyTorch LSTM model training & evaluation pipeline
├── app/
│   ├── app.py                 # Flask server initialization, auth, and routes
│   ├── models.py              # SQLite DB schema (User, PredictionHistory)
│   ├── predictor.py           # Thread-safe Singleton inference service
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css      # Glassmorphic dark design system & CSS variables
│   │   └── js/
│   │       └── main.js        # Client validation, preset fill, AJAX submit
│   ├── templates/
│   │   ├── base.html          # Navigation bar, layout shell, alerts
│   │   ├── login.html         # Login page
│   │   ├── signup.html        # Signup page
│   │   ├── home.html          # Interactive detector workspace
│   │   └── history.html       # Prediction history log & stats
├── models/
│   ├── model.pth              # Optional saved PyTorch LSTM state dictionary
│   ├── model.pkl              # Serialized fallback classifier payload
│   ├── tokenizer.pkl          # Serialized vocabulary tokenizer object
│   └── metrics.json           # Evaluation metrics output
├── requirements.txt           # Dependency requirements with version bounds
└── README.md                  # Comprehensive project documentation
```

---

## ⚡ Quick Start & Run Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the LSTM Model
To train or re-train the classifier and export model artifacts:
```bash
cd model_training
python train_lstm.py
```

### 3. Launch the Web Application
Start the Flask backend server:
```bash
cd app
python app.py
```

Open your browser and navigate to: `http://127.0.0.1:5000`

The SQLite database is created locally on first launch and is excluded from Git. For stable sessions across restarts, set a private `SECRET_KEY` environment variable before launching. If it is not set, the app generates a temporary random key each time it starts.

```powershell
$env:SECRET_KEY = [guid]::NewGuid().ToString()
python app/app.py
```

Use a securely generated secret for deployed environments; do not commit it to the repository.

## ☁️ Deploy to Render

The repository includes a Render Blueprint in `render.yaml`. To publish the running app:

1. Sign in at [Render](https://dashboard.render.com/) and connect the GitHub account that owns this repository.
2. Select **New** → **Blueprint** and choose `vamshi-369/fake-news-detection`.
3. Review the `fake-news-detection` web service and select **Apply**. Render installs the web-only dependencies, generates a private `SECRET_KEY`, and deploys the Flask backend.
4. Open the `onrender.com` URL shown on the service page. The `/healthz` endpoint is used for deployment health checks.

The deployed service loads the checked-in `models/model.pkl` classifier; PyTorch is only needed for training and is intentionally omitted from the web runtime to reduce deployment size.

**Free-tier limitations:** Render may spin down an idle service, so the first visit after inactivity can take about a minute while the backend starts. The app's SQLite database is on ephemeral storage in this setup; accounts and prediction history can be lost when the service restarts or redeploys. Use persistent storage or a managed database before relying on it for durable user data.

---

## 💻 Running in Visual Studio Code (VS Code)

1. Open **VS Code**.
2. Click **File** ➔ **Open Folder...** and select the project folder (`fake`).
3. Open the built-in terminal using `Ctrl + ~` (or **Terminal** ➔ **New Terminal**).
4. Run the application:
   ```powershell
   python app/app.py
   ```
5. **One-Click Debug (F5)**:
   - Go to the **Run and Debug** panel (`Ctrl + Shift + D`).
   - Select **Python: Flask (app/app.py)** and press **F5** (or click the green play button).

---

## 🏢 Running in Visual Studio 2022 / 2019

1. Open **Visual Studio**.
2. Choose **Open a local folder** and select the `fake` folder.
3. In the **Folder View / Solution Explorer**, right-click `app/app.py` and select **Set as Startup Item**.
4. Press **F5** or click the green **Start** button at the top toolbar to run the server.

---

## 🔒 Security Practices & Limitations

1. **Password Protection**: User passwords are never stored in plaintext — hashed using Werkzeug (`generate_password_hash` / `check_password_hash`).
2. **Session Security**: Protected endpoints (`/home`, `/predict`, `/history`) enforce `@login_required` session cookies.
3. **Model Version Sync**: The tokenizer saved during training (`tokenizer.pkl`) is loaded verbatim during inference to prevent index drift.
4. **Domain Generalization Caution**: Models trained on specific datasets (e.g. 2016-2024 political news) learn domain-specific vocabulary. Sensationalist phrasing in non-political domains may alter prediction probability scores. Always verify critical news with primary sources.
