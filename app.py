from flask import Flask, request, jsonify
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
import numpy as np
import joblib  # xrhsimopoiw joblib gia na kanw import ta svm kai vectorizer montela sto flask app
from bs4 import BeautifulSoup
from flask_cors import CORS  

# loadarw ta montela
svc = joblib.load('svc_model.pkl')  
vectorizer = joblib.load('vectorizer.pkl')  
mlp_model = keras.models.load_model('keras_model.h5')  

# xrhsimopoiw to idio preprocess function
stop_words = set(stopwords.words('english'))

def preprocess_review(review):
    # koitaw prwta ama sto text uparxoyn html tags
    if re.search('<.*?>', review):  
        review = BeautifulSoup(review, 'html.parser').get_text()  # ean uparxoun kanw parse me beatifulSoup
    review = review.lower() #kanw to text se lower
    review = re.sub("[^a-zA-Z]",' ',review)#afairw ola ta mh grammata
    review = re.sub(r"https\S+|www\S+|http\S+", '', review, flags = re.MULTILINE)#afairei ta urls
    review = re.sub(r'\@w+|\#', '', review)#afairei ola ta usernames h tags
    review = re.sub(r'[^\w\s]', '', review)#afairei to punctuation
    review_tokens = word_tokenize(review)#kanw tokenization
    filtered_review = [w for w in review_tokens if not w in stop_words]# psaxnei gia common stopwords kai ta afairei kathws den xreiazontai
    return " ".join(filtered_review) # telos ta filtrarizmena tokens gurnane se ena string

# dhmiourgw  flask app
app = Flask(__name__)
CORS(app)  # epitrepw ola ta routes
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:8000"}})  # epitrepw access sto frontend route

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    custom_review = data['review']
    
    # kanw preprocess to review 
    processed_review = preprocess_review(custom_review)
    
    # to kanw vectorize
    custom_review_vectorized = vectorizer.transform([processed_review])
    
    # kanw ta predicts kai me ta 2 montela
    svc_prediction = svc.predict(custom_review_vectorized)
    mlp_prediction = mlp_model.predict(custom_review_vectorized)
    
    # metatrepw to mlp prediction se binary 
    mlp_prediction_B = (mlp_prediction > 0.5).astype(int)
    
    return jsonify({
        'SVM Prediction': int(svc_prediction[0]),
        'MLP Prediction': int(mlp_prediction_B[0])
    })

if __name__ == '__main__':
    app.run(debug=True)