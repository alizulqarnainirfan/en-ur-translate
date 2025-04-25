# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re 

# Load the model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-ur'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# FastAPI app setup
app = FastAPI()

# Request model for input
class TranslationRequest(BaseModel):
    text: str

# Function to preprocess input (optional)
def preprocess(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9?]','',text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Strip leading/trailing spaces
    text = text.lower()  # Convert to lowercase
    return text

# Translation function
def translate_text(text: str) -> str:
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate translation
    output = model.generate(**inputs)
    # Decode and return translated text
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_text

# FastAPI POST endpoint for translation
@app.post("/translate/")
async def translate(request: TranslationRequest):
    # Preprocess and translate the text
    preprocessed_text = preprocess(request.text)
    translated = translate_text(preprocessed_text)
    return {"original_text": request.text, "translated_text": translated}
