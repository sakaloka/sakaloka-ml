from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# Load model & encoders
model = tf.keras.models.load_model("./recommender_model")
user_encoder = joblib.load("./user_encoder.pkl")
place_encoder = joblib.load("./place_encoder.pkl")

# Load tempat
place_df = pd.read_csv("./places.csv")  
place_df = place_df[place_df['place_id'].isin(place_encoder.classes_)].reset_index(drop=True)

# FastAPI app
app = FastAPI()

# Request schema
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 5

@app.post("/recommend")
def recommend_places(request: RecommendationRequest):
    user_id_raw = request.user_id
    k = request.top_k

    if user_id_raw not in user_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"User ID '{user_id_raw}' tidak ditemukan.")

    try:
        user_id = user_encoder.transform([user_id_raw])[0]
        all_place_ids = place_encoder.transform(place_df['place_id'])
        user_array = tf.constant([[user_id, pid] for pid in all_place_ids], dtype= tf.int64)
        preds = model.predict(user_array, verbose=0).flatten()

        top_k_idx = preds.argsort()[-k:][::-1]
        top_places = place_df.iloc[top_k_idx].copy()
        top_places['predicted_rating'] = preds[top_k_idx]

        results = top_places[[
            'place_id', 'place_name', 'city', 'category', 'place_description', 'predicted_rating'
        ]].to_dict(orient="records")

        return {"user_id": user_id_raw, "recommendations": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
