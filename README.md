# 🏡 House Price Predictor

This is a machine learning project that predicts house prices based on various features using the Ames Housing dataset. It includes a trained Random Forest model and an interactive web UI built using Streamlit.

---

## 📦 Features

- Trained on the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Supports preprocessing of categorical and numerical features
- Trained model and scaler saved as `.pkl` files
- Interactive predictions using a Streamlit app
- Clean and modular project structure

---

## 🚀 Getting Started

### 📁 Clone the Repository

```bash
git clone https://github.com/JAISE04/house-price-predictor.git
cd house-price-predictor
```

### 🛠️ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate  # macOS/Linux
```

### 📦 Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

```bash
python src/train_model.py
```

This will:
- Train a Random Forest model
- Save `model.pkl`, `scaler.pkl`, and `feature_columns.pkl` into the `models/` folder

---

## 🔍 Make Predictions on Test Set

```bash
python src/predict.py
```

This will create a submission file at `outputs/submission.csv`.

---

## 🖥️ Launch the Streamlit App

```bash
streamlit run app/app.py
```

---

## 📂 Project Structure

```
house-price-predictor/
├── app/                 # Streamlit frontend
├── data/                # Input datasets
├── models/              # Saved models
├── notebooks/           # Jupyter notebooks
├── outputs/             # Prediction results
├── src/                 # Python scripts
├── requirements.txt
└── README.md
```

---

## 📜 License

This project is licensed for educational purposes.
