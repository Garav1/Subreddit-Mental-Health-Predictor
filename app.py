import streamlit as st

# Set page config FIRST
st.set_page_config(page_title="Mental Health Assistant", layout="centered")

# Rest of imports
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

nltk.download("stopwords")
nltk.download("wordnet")

# Load preprocessed data
@st.cache_resource
def load_model():
    # Load dataset
    df = pd.read_csv("mental_disorders_reddit.csv", encoding='utf-8-sig')
    df = df.dropna(subset=["selftext"])
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    
    df["clean_text"] = df["selftext"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]
    
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["subreddit"])

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label_encoded"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, vectorizer, le, clean_text


model, vectorizer, le, clean_text = load_model()

# Streamlit UI
st.title("🧠 Mental Health Subreddit Predictor")
st.markdown("Enter a Reddit-like post and predict which mental health subreddit it might belong to.")

user_input = st.text_area("Write your thoughts here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)
        predicted_subreddit = le.inverse_transform(pred)[0]
        st.success(f"🏷️ This post most likely belongs to: **r/{predicted_subreddit}**")