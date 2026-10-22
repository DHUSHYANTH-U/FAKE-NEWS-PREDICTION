import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

st.title("Fake News Detection")

title = st.text_input("Enter News Title")
content = st.text_area("Enter News Content")

if st.button("Predict"):
    if not title.strip() or not content.strip():
        st.error("Please enter both title and content to predict.")
    else:
        input_text = title + " " + content

        # Debug: Show length of input text
        st.write("Debug: Input text length:", len(input_text))

        # Vectorize input text
        input_vec = vectorizer.transform([input_text])

        # Debug: Show shape and non-zero count of vectorized input
        st.write("Debug: Vectorized input shape:", input_vec.shape)
        st.write("Debug: Non-zero entries in vector:", input_vec.nnz)

        # Predict class
        prediction = model.predict(input_vec)[0]

        # Debug: Show prediction probabilities
        probs = model.predict_proba(input_vec)[0]
        st.write(f"Debug: Probability REAL: {probs[model.classes_ == 'REAL'][0]:.4f}, FAKE: {probs[model.classes_ == 'FAKE'][0]:.4f}")

        st.success(f"The news is predicted as: {prediction}")

