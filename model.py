import pandas as pd
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib

path_df = "D://CODING-CAMP CAPSTONE//Model Content Base//content base//places_df.csv"

places_df = pd.read_csv(path_df)

def preprocess(text):
    text = str(text).lower()
    text = text.replace(',', ' ')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model():
    places = pd.read_csv("places_df.csv")
    places['category'] = places['category'].fillna('').apply(lambda x: x.replace(',', ' '))
    places['combined'] = (
    places['category'].fillna('') + ' ' +
    places['place_description'].fillna('') + ' ' +
    places['city'].fillna('')
    )   
    places['combined'] = places['combined'].apply(preprocess)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(places['combined'])


    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(tfidf_matrix, "model/tfidf_matrix.pkl")
    places.to_csv("model/places.csv", index=False)

    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()


