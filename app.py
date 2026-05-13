import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download once
nltk.download('punkt')
nltk.download('stopwords')

# load model
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# preprocessing functions
def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(txt):
    return ''.join([i for i in txt if not i.isdigit()])

def remove_stopwords(txt):
    words = word_tokenize(txt)
    filtered = [w for w in words if w not in stopwords.words('english')]
    return " ".join(filtered)

def preprocess(text):
    text = text.lower()
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    return text

# UI
st.title("😊 Emotion Detection App")
st.write("Enter a sentence and detect emotion")

user_input = st.text_area("Enter Text")

if st.button("Predict"):
    if user_input.strip() != "":
        clean_text = preprocess(user_input)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)

        st.success(f"Predicted Emotion: {prediction[0]}")
    else:
        st.warning("Please enter some text!")