import streamlit as st
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# Download stopwords if not available
nltk.download('stopwords')

# Load dataset
st.title("Fake News Detector")
st.write("Upload your dataset for fake news classification.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    news_df = pd.read_csv(uploaded_file)
    news_df = news_df.fillna(" ")

    if 'text' not in news_df.columns or 'label' not in news_df.columns:
        st.error("Dataset must have 'text' and 'label' columns.")
    else:
        # Preprocessing
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        def stemming(content):
            stemmed_content = re.sub("[^a-zA-Z]", " ", content)
            stemmed_content = stemmed_content.lower()
            stemmed_content = stemmed_content.split()
            stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
            return " ".join(stemmed_content)

        news_df['text'] = news_df['text'].apply(stemming)

        # Vectorization
        vector = TfidfVectorizer()
        X = vector.fit_transform(news_df['text'].values)
        y = news_df['label'].values

        # Splitting data
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

        # Model training
        model = LogisticRegression()
        model.fit(X_train, Y_train)

        # Show accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        # User input for prediction
        st.subheader("Test a News Article")
        input_text = st.text_area("Enter a news article")

        if input_text:
            input_data = vector.transform([input_text])
            prediction = model.predict(input_data)
            result = "Fake" if prediction[0] == 1 else "Real"
            st.write(f"The news is: **{result}**")
