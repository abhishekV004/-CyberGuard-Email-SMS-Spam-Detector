# 🛡️ CyberGuard -- Spam Detector

A machine learning powered web app to classify SMS/Emails as **Spam** or
**Not Spam**, built with **Flask, Python, scikit-learn, and NLTK**.

🌐 **Live Demo:** [CyberGuard Spam
Detector](https://cyberguard-email-sms-spam-detector.onrender.com/)

------------------------------------------------------------------------

## ✨ Features

-   🔍 Classifies text messages in real time as **Spam** or **Not
    Spam**\
-   🧠 Machine learning model trained using **TF-IDF + Multinomial Naive
    Bayes**\
-   🌐 Deployed on [Render](https://render.com)\
-   🖥️ Simple Flask backend with custom **Cyberpunk UI**

------------------------------------------------------------------------

## 📂 Project Structure

    ├── app.py               # Flask application
    ├── model.pkl            # Trained ML model
    ├── vectorizer.pkl       # TF-IDF Vectorizer
    ├── requirements.txt     # Dependencies
    ├── templates/
    │   └── index.html       # Frontend template (UI)
    |                        # Styling (if used)
    └── README.md

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   **Frontend:** HTML, CSS (Custom UI)
-   **Backend:** Flask (Python)
-   **ML/Preprocessing:** scikit-learn, NLTK , Numpy , Pandas
-   **Model:** TF-IDF + MultinomialNB
-   **Deployment:** Render (Gunicorn WSGI server)

------------------------------------------------------------------------

## 🚀 Getting Started

### 🔧 Prerequisites

-   Python 3.9+\
-   Virtual environment (`venv` or `conda`)

### 📥 Installation

Clone the repo:

``` bash
git clone https://github.com/abhishekV004/-CyberGuard-Email-SMS-Spam-Detector
cd spam-detector
```

Create virtual environment & activate:

``` bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### ▶️ Run Locally

``` bash
flask --app app run
```

Your app will be available at:\
👉 `http://127.0.0.1:5000/`

------------------------------------------------------------------------

### 🌐 Deployment on Render

1.  Push your project to GitHub.\
2.  Create a **new Web Service** on [Render](https://render.com).\
3.  Set:
    -   **Build Command:**

        ``` bash
        pip install -r requirements.txt
        ```

    -   **Start Command:**

        ``` bash
        gunicorn app:app --timeout 120
        ```
4.  Deploy 🚀

------------------------------------------------------------------------

## 🧪 Example Usage

-   Input: `Congratulations! You have won a $1000 Walmart gift card!`\

-   Output: 🚫 **Spam**

-   Input: `Hey, are we still meeting for lunch tomorrow?`\

-   Output: ✅ **Not Spam**

------------------------------------------------------------------------

## 📝 Notes

-   Ensure scikit-learn version matches training (recommended:
    `1.5.1`).\
-   If retraining, upgrade to latest version and regenerate `model.pkl`
    & `vectorizer.pkl`.\
-   NLTK stopwords/punkt must be downloaded (already handled in
    `app.py`).

------------------------------------------------------------------------

## ❤️ Credits

Made with Flask & Python by **Abhishek Vishwakarma** © 2025
