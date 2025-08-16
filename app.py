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

from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK data is available
nltk_packages = ["punkt", "punkt_tab", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# Load preprocessing tools
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Load fitted vectorizer and trained model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]

    # Remove stopwords
    y = [i for i in y if i not in stop_words]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        input_sms = request.form["message"]

        if input_sms.strip() != "":
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                prediction = "🚫 Spam"
            else:
                prediction = "✅ Not Spam"
        else:
            prediction = "⚠ Please enter a message."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
