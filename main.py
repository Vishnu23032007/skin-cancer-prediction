from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import pickle

app = FastAPI()

model = tf.keras.models.load_model("skin_cancer_cnn_model.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    label = label_encoder.inverse_transform(predicted_class)[0]

    return JSONResponse(content={"prediction": label})
@app.get("/")
def read_root():
    return {"message": "Skin Cancer Prediction API is live!"}

