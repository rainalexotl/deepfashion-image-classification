from fastapi import FastAPI, UploadFile
from PIL import Image
from io import BytesIO

from model import load_model, make_prediction

app = FastAPI()
model = load_model()

@app.get('/')
async def health_check():
    return {'status': 'running'}

@app.post('/predict')
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    prediction = make_prediction(model, image)
    return {"prediction": prediction}