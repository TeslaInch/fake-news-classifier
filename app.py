import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Classifier")
st.write("Enter news content to check if it's real or fake.")

user_input = st.text_area("News content:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        label = "🛑 Fake News" if prediction == 1 else "✅ Real News"
        st.success(f"Prediction: {label}")
