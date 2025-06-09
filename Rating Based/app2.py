from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

model = tf.keras.models.load_model("./recommender_model")
user_encoder = joblib.load("./user_encoder.pkl")
place_encoder = joblib.load("./place_encoder.pkl")

place_df = pd.read_csv("./places.csv")  
place_df = place_df[place_df['place_id'].isin(place_encoder.classes_)].reset_index(drop=True)

app = FastAPI()
class PlaceRating(BaseModel):
    place_id: int = Field(..., example=101)
    rating: Optional[float] = Field(None, example=4.5)

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., example=5)
    history: List[PlaceRating] = Field(default_factory=list, example=[
        {"place_id": 101, "rating": 4.5},
        {"place_id": 102, "rating": 3.0}
    ])
    top_k: int = Field(5, example=5)

@app.post("/recommend")
def recommend_places(request: RecommendationRequest):
    user_id_raw = request.user_id
    k = request.top_k
    history = request.history

    if user_id_raw not in user_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"User ID '{user_id_raw}' tidak ditemukan.")

    try:
        user_id = user_encoder.transform([user_id_raw])[0]

        history_place_ids = [h.place_id for h in history]
        known_history_places = [pid for pid in history_place_ids if pid in place_encoder.classes_]
        encoded_history_places = place_encoder.transform(known_history_places) if known_history_places else []

        candidate_places = place_df[~place_df['place_id'].isin(known_history_places)].reset_index(drop=True)
        candidate_place_ids = place_encoder.transform(candidate_places['place_id'])

        user_place_pairs = tf.constant([[user_id, pid] for pid in candidate_place_ids], dtype=tf.int64)
        preds = model.predict(user_place_pairs, verbose=0).flatten()

        top_k_idx = preds.argsort()[-k:][::-1]
        top_places = candidate_places.iloc[top_k_idx].copy()
        top_places['predicted_rating'] = preds[top_k_idx]

        results = top_places[[
            'place_id', 'place_name', 'city', 'category', 'place_description', 'predicted_rating'
        ]].to_dict(orient="records")

        return {"user_id": user_id_raw, "recommendations": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
