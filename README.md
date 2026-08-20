# 🍃 Apple Crop Disease Detector

> AI-powered leaf disease detection using ResNet18, FastAPI, and React — from a photo of a leaf to a diagnosis in seconds.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat&logo=vite&logoColor=white)

---

## ✨ Features

- 🔍 **ResNet18** fine-tuned on Apple leaf disease classes
- ⚡ **FastAPI** backend — sub-100ms inference response
- 🎨 **React + Vite** frontend with:
  - Drag-and-drop image upload
  - Dark / Light mode toggle
  - Animated confidence bars per class
  - Glassmorphism cards + floating orb background
  - Leaf-bounce logo animation + shimmer button effect
- 🌙 Full dark mode support
- 📱 Responsive design

---

## 📁 Project Structure

```
crop-disease-detector/
├── src/
│   ├── __init__.py           # Makes src a Python package
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app — POST /predict
│   │   └── inference.py      # Reusable inference logic
│   ├── model.py              # ResNet18 architecture
│   ├── data_loader.py        # Dataset + transforms
│   ├── data_split.py         # Train/val/test split script
│   ├── train.py              # Training loop
│   └── evaluate.py           # Test-set evaluation
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx           # Full UI with animations
│       └── App.css           # Design system CSS
├── tests/
│   ├── conftest.py
│   └── test_dummy.py
├── notebooks/
│   └── 01_explore_data.ipynb
└── models/                   # (git-ignored) — trained weights go here
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- Node.js 18+ (via [nvm](https://github.com/nvm-sh/nvm) recommended)
- Trained model weights at `models/best_model.pth`

---

### 1. Backend — FastAPI

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install fastapi "uvicorn[standard]" python-multipart torch torchvision pillow

# Start the API server
uvicorn src.api.main:app --reload
# → Runs on http://localhost:8000
```

Test it manually:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/leaf.jpg"
```

Expected response:
```json
{
  "predicted_class": "Apple___Apple_scab",
  "confidence": 97.34,
  "probabilities": {
    "Apple___Apple_scab": 97.34,
    "Apple___Black_rot": 1.22,
    "Apple___Cedar_apple_rust": 0.91,
    "Apple___healthy": 0.53
  }
}
```

---

### 2. Frontend — React + Vite

```bash
cd frontend
npm install
npm run dev
# → Opens http://localhost:5173
```

> **macOS tip:** If you see a `libsimdjson` dynamic library error with Node, switch to a stable Node version via nvm:
> ```bash
> nvm install --lts
> nvm use --lts
> cd frontend && npm install && npm run dev
> ```

---

## 🏋️ Training Pipeline

```bash
# 1. Split raw data into train/val/test
python src/data_split.py

# 2. Train the ResNet18 model
python src/train.py
# Saves best weights to models/best_model.pth

# 3. Evaluate on the test set
python src/evaluate.py
```

---

## 🧪 Tests

```bash
pytest -q
```

---

## 🤖 Model Details

| Property      | Value                                      |
|---------------|--------------------------------------------|
| Architecture  | ResNet18 (ImageNet1K_V1 pretrained)        |
| Input size    | 128×128 RGB                                |
| Classes       | 4 (Scab, Black Rot, Cedar Rust, Healthy)   |
| Normalisation | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]     |
| Fine-tuning   | Final FC layer unfrozen; all others frozen |

---

## 📄 License

MIT
# abc
