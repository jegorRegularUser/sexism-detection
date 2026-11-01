import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def translate_if_russian(text: str) -> str:
    try:
        language = detect(text)
        if language == "ru":
            return GoogleTranslator(source="ru", target="en").translate(text)
        return text
    except LangDetectException:
        return text
    except Exception:
        # translation failures must not break prediction
        return text


class Message(BaseModel):
    text: str
    model: str


AVAILABLE_MODELS = (
    "sexism-distilbert-v2",  # trained with ml/train.py, see README
    "local_3",
    "local_5",
    "collab_5",
    "collab_7",
    "2_collab_3",
)

# Models are lazy-loaded on first use: startup stays fast and RAM is only
# spent on models people actually ask for.
_loaded: dict[str, tuple] = {}


def _load_threshold(path: str) -> float:
    """Trained models ship their tuned decision threshold."""
    meta = os.path.join(path, "threshold.json")
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as fh:
                return float(json.load(fh)["threshold"])
        except (json.JSONDecodeError, KeyError):
            pass
    return 0.5


def _get_model(name: str):
    if name not in _loaded:
        path = f"./models/{name}"
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.eval()
            _loaded[name] = (tokenizer, model, _load_threshold(path))
        except OSError:
            raise HTTPException(status_code=400, detail="Model not found")
    return _loaded[name]


# Sync endpoint on purpose: torch inference would block the event loop, so
# FastAPI runs this in its threadpool instead.
@app.post("/predict/")
def predict(message: Message):
    if message.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Model not found")

    tokenizer, model, threshold = _get_model(message.model)

    text = translate_if_russian(message.text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=128, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    sexist_probability = float(probs[0][1].item())
    prediction = int(sexist_probability >= threshold)
    return {
        "prediction": prediction,
        "probability": float(probs[0][prediction].item()),
        "sexist_probability": sexist_probability,
        "model": message.model,
    }
