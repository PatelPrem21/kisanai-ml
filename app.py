from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import io
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

SEVERITY = {
    'Apple___Apple_scab': 'medium',
    'Apple___Black_rot': 'high',
    'Apple___Cedar_apple_rust': 'medium',
    'Apple___healthy': 'none',
    'Blueberry___healthy': 'none',
    'Cherry_(including_sour)___Powdery_mildew': 'medium',
    'Cherry_(including_sour)___healthy': 'none',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'medium',
    'Corn_(maize)___Common_rust_': 'medium',
    'Corn_(maize)___Northern_Leaf_Blight': 'high',
    'Corn_(maize)___healthy': 'none',
    'Grape___Black_rot': 'high',
    'Grape___Esca_(Black_Measles)': 'high',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'medium',
    'Grape___healthy': 'none',
    'Orange___Haunglongbing_(Citrus_greening)': 'high',
    'Peach___Bacterial_spot': 'medium',
    'Peach___healthy': 'none',
    'Pepper,_bell___Bacterial_spot': 'high',
    'Pepper,_bell___healthy': 'none',
    'Potato___Early_blight': 'medium',
    'Potato___Late_blight': 'high',
    'Potato___healthy': 'none',
    'Raspberry___healthy': 'none',
    'Soybean___healthy': 'none',
    'Squash___Powdery_mildew': 'medium',
    'Strawberry___Leaf_scorch': 'medium',
    'Strawberry___healthy': 'none',
    'Tomato___Bacterial_spot': 'high',
    'Tomato___Early_blight': 'medium',
    'Tomato___Late_blight': 'high',
    'Tomato___Leaf_Mold': 'medium',
    'Tomato___Septoria_leaf_spot': 'medium',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'medium',
    'Tomato___Target_Spot': 'medium',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'high',
    'Tomato___Tomato_mosaic_virus': 'high',
    'Tomato___healthy': 'none'
}

def load_model():
    model = efficientnet_b0(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(1280, len(CLASSES))
    model.load_state_dict(torch.load('/app/models/best_model.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image).unsqueeze(0)

def get_treatment_advice(disease: str) -> dict:
    prompt = f"""
You are an agricultural expert. A farmer's crop has been diagnosed with: {disease}

Give specific, practical advice for this exact disease/condition.
If the crop is healthy (contains 'healthy'), give tips to maintain health and prevent disease.

Return ONLY a valid JSON object, no markdown, no explanation:
{{
    "treatment_steps": ["specific step 1", "specific step 2", "specific step 3"],
    "fertilizer": "specific fertilizer recommendation",
    "prevention": "specific prevention advice",
    "urgency": "immediate OR within 3 days OR within a week OR no action needed"
}}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert agricultural advisor. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

app = FastAPI()
model = load_model()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
def health():
    return {"status": "ok", "model": "EfficientNetB0", "classes": len(CLASSES)}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = preprocess(image)

    with torch.inference_mode():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

    # Top 3 predictions from model
    top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)

    confidence_val = round(top3_probs[0][0].item() * 100, 2)
    disease = CLASSES[top3_indices[0][0].item()]

    # Low confidence check
    if confidence_val < 60:
        return {
            "status"    : "low_confidence",
            "confidence": confidence_val,
            "message"   : "Image unclear. Please take a closer photo of the affected leaf in good lighting."
        }

    crop = disease.split('_')[0]
    is_healthy = 'healthy' in disease.lower()

    # Main prediction advice
    main_advice = get_treatment_advice(disease)

    # 2nd and 3rd predictions advice
    similar_list = []
    for i in range(1, 3):
        sim_disease    = CLASSES[top3_indices[0][i].item()]
        sim_confidence = round(top3_probs[0][i].item() * 100, 2)
        sim_advice     = get_treatment_advice(sim_disease)
        similar_list.append({
            "name"           : sim_disease.replace('_', ' '),
            "confidence"     : sim_confidence,
            "severity"       : SEVERITY[sim_disease],
            "treatment_steps": sim_advice["treatment_steps"],
            "fertilizer"     : sim_advice["fertilizer"],
            "prevention"     : sim_advice["prevention"],
        })

    return {
        "status"          : "success",
        "crop"            : crop,
        "disease"         : disease.replace('_', ' '),
        "is_healthy"      : is_healthy,
        "confidence"      : confidence_val,
        "severity"        : SEVERITY[disease],
        "treatment_steps" : main_advice["treatment_steps"],
        "fertilizer"      : main_advice["fertilizer"],
        "prevention"      : main_advice["prevention"],
        "urgency"         : main_advice["urgency"],
        "disclaimer"      : "Model trained on lab images. Real-world accuracy may vary.",
        "similar_diseases": similar_list
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await analyze(file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)