from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import re
from nltk.stem import PorterStemmer
import uvicorn






# Load the stop words, TFIDF vectorizer, and Naive Bayes model from pickle files
stop_words = joblib.load('stop_words.pkl')
tfidf = joblib.load('tfidf.pkl')
nb_model = joblib.load('nb_model.pkl')
stemmer = PorterStemmer()


## step 1 :
def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize by splitting on whitespace
    tokens = text.split()
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(tokens)


class PredictionRequest(BaseModel):
    text: str





app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_sentiment(request: PredictionRequest):
    df = pd.DataFrame([request.text], columns=['text'])
    df['cleaned_text'] = df['text'].apply(clean_text)
    X_tfidf = tfidf.transform(df['cleaned_text'])
    y_pred = nb_model.predict(X_tfidf)
    y_pred_proba = nb_model.predict_proba(X_tfidf)
    return {"sentiment": str(y_pred[0]), "probabilities": str(y_pred_proba[0][1])}



