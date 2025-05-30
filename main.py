from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


places = pd.read_csv("model/places.csv")
vectorizer = joblib.load("model/vectorizer.pkl")
tfidf_matrix = joblib.load("model/tfidf_matrix.pkl")

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_n: int = 5 #ambil 5 rekumendasi teratas

@app.post("/recommend/")
def recommend_places(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

  
    query_vec = vectorizer.transform([query])

    
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

   
    top_indices = similarity_scores.argsort()[-req.top_n:][::-1]

  
    results = []
    for idx in top_indices:
        results.append({
            "place_name": places.iloc[idx]["place_name"],
            "city": places.iloc[idx]["city"],
            "category": places.iloc[idx]["category"],
            "description": places.iloc[idx]["place_description"],
            "similarity_score": float(similarity_scores[idx])
        })

    return {"query": req.query, "recommendations": results}
