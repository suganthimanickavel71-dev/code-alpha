from flask import Flask, render_template, request, jsonify
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Download NLTK data
nltk.download("stopwords")


# -----------------------------
# FAQ DATA
# -----------------------------

faqs = [
    {
        "question": "What is the college timing?",
        "answer": "The college working hours are from 9:00 AM to 4:00 PM."
    },

    {
        "question": "When does the college start?",
        "answer": "The college starts at 9:00 AM."
    },

    {
        "question": "Where is the college located?",
        "answer": "The college is located in Tamil Nadu, India."
    },

    {
        "question": "How can I apply for admission?",
        "answer": "You can apply for admission through the college admission office or official website."
    },

    {
        "question": "What courses are available?",
        "answer": "The college offers courses in Computer Science, Information Technology, Electronics, Mechanical and other departments."
    },

    {
        "question": "How can I contact the college?",
        "answer": "You can contact the college through the official phone number or email address."
    },

    {
        "question": "Is there a placement department?",
        "answer": "Yes. The college has a placement department that supports students with training and recruitment."
    },

    {
        "question": "What is the attendance requirement?",
        "answer": "Students are generally required to maintain the minimum attendance percentage specified by the college."
    },

    {
        "question": "Is hostel facility available?",
        "answer": "Yes, hostel facilities are available for eligible students."
    },

    {
        "question": "Is there a library?",
        "answer": "Yes, the college has a library with academic books and learning resources."
    }
]


# -----------------------------
# NLP PREPROCESSING
# -----------------------------

stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()


def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords and perform stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Preprocess FAQ questions

faq_questions = [
    preprocess(faq["question"])
    for faq in faqs
]


# -----------------------------
# TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq_questions)


# -----------------------------
# FIND BEST FAQ
# -----------------------------

def get_answer(user_question):

    processed_question = preprocess(user_question)

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match = similarity.argmax()

    best_score = similarity[0][best_match]


    # Minimum similarity threshold

    if best_score < 0.20:

        return "Sorry, I don't understand your question. Please ask something related to the college."


    return faqs[best_match]["answer"]


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------
# CHAT API
# -----------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_question = data.get("message", "")

    if not user_question.strip():

        return jsonify({
            "answer": "Please enter a question."
        })


    answer = get_answer(user_question)


    return jsonify({
        "answer": answer
    })


# -----------------------------
# RUN APPLICATION
# -----------------------------

if __name__ == "__main__":

    app.run(debug=True)