# Breast Cancer Predictor (Streamlit App)

A simple machine learning web app that predicts whether a breast tumor is benign or malignant based on input features.  
The app is built using Logistic Regression and deployed with Streamlit.

---

## Features
- Interactive sliders for all input features
- Real-time prediction
- Probability output for benign vs malignant
- Radar chart visualization of input features

---

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly

---

## Project Structure

streamlit_cancerpredictor/
│
├── app/
│   └── main.py
├── model/
│   ├── model_v1.pkl
│   ├── scaler_v1.pkl
│   └── minmax.pkl
├── data/
│   └── data.csv
├── requirements.txt
└── README.md

---

## How to Run Locally

1. Clone the repository:
git clone https://github.com/your-username/streamlit_cancerpredictor.git
cd streamlit_cancerpredictor

2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run app/main.py

4. Open in browser:
http://localhost:8501

---

## Model
- Algorithm: Logistic Regression
- Dataset: Breast Cancer dataset (UCI / sklearn)
- Input: 30 numeric features
- Output: Benign or Malignant prediction with probabilities

---

## Live Demo
https://nebulav1cancerpredictor.streamlit.app
This link is not operational.
---

## Author
Priyansh