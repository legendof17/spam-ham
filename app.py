import streamlit as st
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())])
model = joblib.load('Email_Class')
st.title('Spam Ham Classifier')
ip = st.text_input('Enter your message')
op = model.predict([ip])
if st.button('Predict'):
  st.title(op[0])
