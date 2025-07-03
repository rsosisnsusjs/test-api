from fastapi import FastAPI, Request
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

app = FastAPI()

model = joblib.load("moodmeal_model.pkl")
mlb = joblib.load("moodmeal_mlb.pkl")

@app.post("/recommend")
async def recommend(request: Request):
    try:
        data = await request.json()
        multi_cols = [
            "flavor", "texture", "emotion", "post_feeling", "meal_time",
            "hunger_level", "eating_style", "cuisine", "budget", "location_type"
        ]
        for col in multi_cols:
            if col in data:
                if not isinstance(data[col], list):
                    data[col] = [x.strip() for x in str(data[col]).split(",")]
            else:
                data[col] = []

        df = pd.DataFrame([data])
        multi_encoded = pd.DataFrame(mlb.transform(df[multi_cols]), columns=mlb.classes_)
        pred = model.predict(multi_encoded)
        return {"recommended_menu": pred[0]}
    except Exception as e:
        return {"error": str(e)}
