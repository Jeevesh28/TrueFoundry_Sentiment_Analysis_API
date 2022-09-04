import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from string import punctuation 
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
): nltk.download(dependency)

df = pd.read_csv('airline_sentiment_analysis.csv', index_col=0)
df.reset_index(inplace = True, drop = True)
df['airline'] = ''
df['airline'] = df['text'].str.findall(r'@([a-zA-Z0-9_]{1,50})').str[0].str.lower()
df['text'] = df['text'].str.replace(r'@[A-Za-z0-9_]+', '', regex=True).astype(str)
stop_words =  stopwords.words('english')

def text_cleaning(text, remove_stop_words = True, lemmatize_words = True):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) 
    text = ''.join([c for c in text if c not in punctuation])
    
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text

df['cleaned_text'] = df['text'].apply(text_cleaning)
X = df['cleaned_text']
df['airline_sentiment'] = df['airline_sentiment'].map({'negative': 0, 'positive': 1})
Y = df['airline_sentiment']

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.15, random_state = 42, shuffle = True, stratify = Y)
sentiment_classifier = Pipeline(steps=[('pre_processing',TfidfVectorizer(lowercase=True)), ('svm', svm.SVC(probability=True, C= 10, gamma= 'scale', kernel= 'rbf'))])
sentiment_classifier.fit(X_train,y_train)
joblib.dump(sentiment_classifier, 'model_pipeline.pkl')
print('Model Saved')