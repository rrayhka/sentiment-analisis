import re
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('dataset/dataset.csv')
data = data[["normalized", "sentiment"]]

# Load stopwords
with open('stopwords.txt', 'r') as file:
    stopwords = file.read().splitlines()

# Create and fit tf-idf vectorizer
tfidf = TfidfVectorizer(stop_words=stopwords)
X = tfidf.fit_transform(data['normalized'])

# Label encode the sentiments
le = LabelEncoder()
y = le.fit_transform(data['sentiment'])

# Train the SVM model
final_svm = SVC(kernel='linear')
final_svm.fit(X, y)

# Save the model and label encoder
pickle.dump(final_svm, open('models/final_svm_model.pkl', 'wb'))
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))
pickle.dump(tfidf, open('models/tfidf_vectorizer.pkl', 'wb'))

def load_svm_model():
    final_svm = pickle.load(open('models/final_svm_model.pkl', 'rb'))
    le = pickle.load(open('models/label_encoder.pkl', 'rb'))
    tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    return final_svm, le, tfidf

def predict_sentiment_svm(text, model, le, tfidf):
    text_transformed = tfidf.transform([text])
    prediction = model.predict(text_transformed)
    sentiment = le.inverse_transform(prediction)[0]
    return sentiment


# print(predict_sentiment_svm("halo ini produk bagus sekali", final_svm, le, tfidf))