# F1-prediction-dashboard
I have created an F1 race winner predictor app using ML. You can predict a particular race winner or view it in simplified dashboard using app or on webpage.

# F1 Winner Predictor Bundle

This repository contains everything you need to train, serve, and interact with an F1 race winner prediction model:

* **Python Backend**

  * Data ingestion & feature engineering using FastF1
  * Model training & evaluation (Gradient Boosting Classifier)
  * Streamlit dashboard (light-themed) for ad-hoc predictions
  * FastAPI service to expose prediction endpoints for external clients

* **Flutter Mobile Client**

  * Simple UI to request predictions from the FastAPI backend
  * Podium view, sliders for what‑if analysis, full leaderboard
  * Buildable as an APK for Android devices

---

## Repository Structure

```
/ (root)
├── python/                         # Python backend & dashboard
│   ├── train_model.py             # Train & save model artifacts
│   ├── mymodel.py                 # Unified train/predict CLI script
│   ├── utils_f1.py                # Shared functions (cache, feature prep, predict)
│   ├── app.py                     # Streamlit dashboard (light theme)
│   ├── server/                    # FastAPI server code
│   │   ├── __init__.py
│   │   └── main.py                # API endpoints (/predict, /health)
│   ├── requirements.txt           # Python dependencies
│   └── .streamlit/config.toml     # Streamlit server config

└── flutter_app/                   # Flutter mobile client
    ├── pubspec.yaml               # Dart & Flutter dependencies
    └── lib/
        ├── main.dart              # App entrypoint
        ├── providers.dart         # API baseUrl provider
        ├── models/
        │   └── prediction.dart    # Prediction data model
        ├── services/
        │   └── api.dart           # HTTP client for FastAPI
        └── screens/
            └── home_screen.dart   # Main UI with sliders & podium
```

---

## Getting Started

### Prerequisites

* Python 3.9+
* Flutter SDK & Android toolchain (for mobile client)
* Git

### Python Backend Setup

1. **Navigate to the Python folder**

   ```bash
   cd python
   ```

2. **Install dependencies**

   ```bash
   sed -i '/^python/d' requirements.txt    # remove the python>=3.9 hint
   pip install -r requirements.txt
   ```

3. **Train the model**

   ```bash
   python train_model.py --train-start 2018 --train-end 2024 --test-year-start 2023
   ```

   This will generate:

   * `winner_model.pkl`
   * `feats.parquet`
   * `feats.cols.json`

4. **Run the FastAPI server**

   ```bash
   python -m uvicorn server.main:app --host 0.0.0.0 --port 8050
   ```

   * Health check: `GET http://localhost:8050/health`
   * Swagger UI:  `http://localhost:8050/docs`

5. **Launch the Streamlit dashboard (optional)**

   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```

   Browse to `http://localhost:8501`.

### Flutter Client Setup

1. **Navigate to the Flutter folder**

   ```bash
   cd ../flutter_app
   ```

2. **Configure the API base URL**
   Edit `lib/providers.dart`:

   ```dart
   const baseUrl = 'http://<YOUR_MACHINE_IP>:8050';
   ```

3. **Install Flutter dependencies**

   ```bash
   flutter pub get
   ```

4. **Run on device or emulator**

   ```bash
   flutter run
   ```

5. **Build a release APK**

   ```bash
   flutter build apk --release
   ```

   The APK will be at `build/app/outputs/flutter-apk/app-release.apk`.

---

## Usage

* **Streamlit UI**: interactive sliders, manual feature edits, download CSV.
* **FastAPI**: programmatic access via `/predict?year=2025&round=12&w_grid=1.0&...`
* **Flutter App**: mobile-friendly endpoint, podium view, full leaderboard.

---

*Happy coding and may the best driver win!*
