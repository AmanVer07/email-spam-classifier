import streamlit as st

def login():
    st.title("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "spam123":
            st.session_state["authenticated"] = True
        else:
            st.error("❌ Invalid credentials")

if "authenticated" not in st.session_state:
    login()
    st.stop()


import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

@st.cache_resource
def download_nltk():
    nltk.download('stopwords')

download_nltk()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = wordpunct_tokenize(text)

    y = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            y.append(ps.stem(word))

    return " ".join(y)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
