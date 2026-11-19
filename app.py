import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("📰 Fake News Detector")
st.write("Enter any news headline and check if it's REAL or FAKE!")

data = {
    'text': [
        "Government opens a new public hospital",
        "Scientists discover new planet",
        "School announces new exam dates",
        "PM announces new scheme for farmers",
        "Cricket team wins world cup final",
        "India launches new satellite",
        "Mumbai police arrests thief",
        "New metro line starts in Pune",
        "Rakhi Sawant will perform on Mars for aliens",
        "Man turns into dog after eating pizza",
        "Aliens will visit Earth tomorrow",
        "Rakhi Sawant bought the Moon and will charge rent",
        "Man buys Moon and will charge rent",
        "Aliens landed in Mumbai yesterday",
        "Boy becomes superhero after eating Maggi",
        "Dog starts talking and becomes PM",
        "Ghost seen in school at night",
        "World will end tomorrow"
    ],
    'label': [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
}
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
model = LogisticRegression()
model.fit(X, df['label'])

news = st.text_input("Enter News Headline:")

if st.button("Check"):
    if news:
        pred = model.predict(vectorizer.transform([news]))[0]
        if pred == 1:
            st.success("This is REAL NEWS! ✅")
        else:
            st.error("This is FAKE NEWS! ❌")
    else:
        st.warning("Please enter a headline first!")