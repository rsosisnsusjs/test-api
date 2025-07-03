from fastapi import FastAPI, Request
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

app = FastAPI()

model = joblib.load("moodmeal_model.pkl")
mlb = joblib.load("moodmeal_mlb.pkl")

@app.post("/recommend")
async def recommend(request: Request):
    data = await request.json()

    # รายการคอลัมน์ที่ต้อง normalize ให้เป็น list เสมอ
    multi_cols = [
        "flavor", "texture", "emotion", "post_feeling", "meal_time", 
        "hunger_level", "eating_style", "cuisine", "budget", "location_type"
    ]
    # แปลงค่าให้เป็น list ถ้ายังไม่ใช่ (เพื่อ binarize ให้เหมือนตอน train)
    for col in multi_cols:
        if col in data:
            if not isinstance(data[col], list):
                data[col] = [x.strip() for x in str(data[col]).split(",")]
        else:
            data[col] = []  # กรณีไม่มี key ใน data

    df = pd.DataFrame([data])

    # ใช้ mlb ที่โหลดมา แปลง multi-label features
    multi_encoded = pd.DataFrame(mlb.transform(df[multi_cols]), columns=[f"{cls}" for cls in mlb.classes_])

    # ใช้ model ทำนายเมนู
    pred = model.predict(multi_encoded)
    return {"recommended_menu": pred[0]}
