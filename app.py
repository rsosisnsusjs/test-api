from fastapi import FastAPI, Request
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("moodmeal_model.pkl")

multi_cols = ["flavor", "texture", "emotion", "post_feeling", "meal_time", "hunger_level", "eating_style", "cuisine", "budget", "location_type"]
mlb_dict = {col: joblib.load(f"moodmeal_mlb_{col}.pkl") for col in multi_cols}


@app.post("/recommend")
async def recommend(request: Request):
    try:
        data = await request.json()
        
        # Normalize input เป็น list ทุกคอลัมน์
        for col in multi_cols:
            if col in data:
                if not isinstance(data[col], list):
                    data[col] = [x.strip() for x in str(data[col]).split(",")]
            else:
                data[col] = []

        df = pd.DataFrame([data])

        # แปลงข้อมูลทีละคอลัมน์
        encoded_list = []
        for col in multi_cols:
            enc = mlb_dict[col].transform(df[col])
            cols = [f"{col}_{cls}" for cls in mlb_dict[col].classes_]
            encoded_list.append(pd.DataFrame(enc, columns=cols))

        X = pd.concat(encoded_list, axis=1)

        pred = model.predict(X)
        return {"recommended_menu": pred[0]}

    except Exception as e:
        return {"error": str(e)}
