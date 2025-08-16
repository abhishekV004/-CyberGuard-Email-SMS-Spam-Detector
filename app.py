# import streamlit as st
# import pickle
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # Download required NLTK data (only first time)
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# # Load fitted vectorizer and trained model
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # Must be fitted TfidfVectorizer
# model = pickle.load(open('model.pkl', 'rb'))

# # Text preprocessing function
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     # Remove non-alphanumeric tokens
#     y = [i for i in text if i.isalnum()]

#     # Remove stopwords
#     y = [i for i in y if i not in stop_words]

#     # Stemming
#     y = [ps.stem(i) for i in y]

#     return " ".join(y)

# # Streamlit UI
# st.title("📩 Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     if input_sms.strip() == "":
#         st.warning("Please enter a message before predicting.")
#     else:
#         # 1. Preprocess
#         transformed_sms = transform_text(input_sms)

#         # 2. Vectorize
#         vector_input = tfidf.transform([transformed_sms])

#         # 3. Predict
#         result = model.predict(vector_input)[0]

#         # 4. Display result
#         if result == 1:
#             st.error("🚫 Spam")
#         else:
#             st.success("✅ Not Spam")

import os
import pathlib
import pickle
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Paths ---
BASE_DIR = pathlib.Path(__file__).resolve().parent

# --- Flask ---
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# --- NLTK data (vendor or cache inside repo) ---
NLTK_DIR = BASE_DIR / "nltk_data"
os.environ["NLTK_DATA"] = str(NLTK_DIR)  # look here first

# Try to use vendored data; fallback to download (not ideal on Render)
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=str(NLTK_DIR))
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", download_dir=str(NLTK_DIR))

ensure_nltk()

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# --- Load artifacts with absolute paths ---
with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/")
def predict():
    input_sms = request.form.get("message", "").strip()
    if not input_sms:
        return render_template("index.html", prediction="⚠ Please enter a message.")
    transformed = transform_text(input_sms)
    vector = tfidf.transform([transformed])
    result = model.predict(vector)[0]
    prediction = "🚫 Spam" if result == 1 else "✅ Not Spam"
    return render_template("index.html", prediction=prediction)

@app.get("/health")
def health():
    return "ok", 200

# Do NOT call app.run() here; Render will start via gunicorn.
