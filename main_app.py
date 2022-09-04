from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from os.path import dirname, join, realpath
import re
import joblib
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title = 'Sentiment Analysis API',
    description = 'A simple API that use model to predict the sentiment of the entered text'
)

with open(join(dirname(realpath(__file__)), 'model_pipeline.pkl'), 'rb') as f:
    model = joblib.load(f)

def text_cleaning(text, remove_stop_words = True, lemmatize_words = True):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)
    text = "".join([c for c in text if c not in punctuation])

    if remove_stop_words:
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text

@app.get("/predict/{text}")

def predict_sentiment(text: str):
    cleaned_text = text_cleaning(text)
    prediction = model.predict([cleaned_text])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_text])
    output_probability = '{:.2f}'.format(float(probas[:, output]))
    sentiments = {0: 'Negative', 1:'Positive'}
    result = {'Prediction': sentiments[output], 'Probability': output_probability}
    return {'output': result}