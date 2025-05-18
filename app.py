import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# reading data
df = pd.read_csv("WELFake_Dataset.csv")
df = df.fillna(" ")
# news_df["content"] = news_df["author"] + " " + news_df["title"]
X = df.drop("label", axis=1)
y = df["label"]

# define stemming function
ps = PorterStemmer()


def stemming(content):
    stemmed_text = re.sub("[^a-zA-Z]", " ", content)
    stemmed_text = stemmed_text.lower()
    stemmed_text = stemmed_text.split()
    stemmed_text = [
        ps.stem(word)
        for word in stemmed_content
        if not word in stopwords.words("english")
    ]
    stemmed_text = " ".join(stemmed_text)
    return stemmed_text


# apply stemming function to content column
# news_df["content"] = news_df["content"].apply(stemming)

# vectorize data
x = df["content"].values
y = df["label"].values
vector = TfidfVectorizer()
vector.fit(X)
Xx = vector.transform(X)

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# fitttings logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)


# web ui
st.title("Fake News Detector")
input_text = st.text_input("Enter news Article")


def prediction(input_text):
    # conver the input in machine readable
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]


if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write("The News is Fake")
    else:
        st.write("The News Is Real")
