from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
    model: str

models = {
    "local_3": DistilBertForSequenceClassification.from_pretrained('./models/local_3'),
    "local_5": DistilBertForSequenceClassification.from_pretrained('./models/local_5'),
    "collab_5": DistilBertForSequenceClassification.from_pretrained('./models/collab_5'),
    "collab_7": DistilBertForSequenceClassification.from_pretrained('./models/collab_7'),

    "2_collab_3": DistilBertForSequenceClassification.from_pretrained('./models/2_collab_3')
}

tokenizers = {
    "local_3": DistilBertTokenizer.from_pretrained('./models/local_3'),
    "local_5": DistilBertTokenizer.from_pretrained('./models/local_5'),
    "collab_5": DistilBertTokenizer.from_pretrained('./models/collab_5'),
    "collab_7": DistilBertTokenizer.from_pretrained('./models/collab_7'),

    "2_collab_3": DistilBertTokenizer.from_pretrained('./models/2_collab_3'),
}

@app.post("/predict/")
async def predict(message: Message):
    model_name = message.model
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    model = models[model_name]
    tokenizer = tokenizers[model_name]
    
    inputs = tokenizer(message.text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    
    return {"prediction": prediction}
