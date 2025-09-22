# app.py

# IMPORT render_template to serve your HTML page
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models # Import models for Vision Transformer
from PIL import Image
import json
import os
import requests
import torch.serialization
import google.generativeai as genai

# ----------------- Flask App -----------------
app = Flask(__name__)
CORS(app)

# ----------------- Paths -----------------
model_path = "ml_models/plant_disease_model.pth"
classes_path = 'ml_models/classes.json'


# ----------------- Gemini API Setup -----------------
try:
    # IMPORTANT: It's better to load keys from environment variables
    API_KEY = "" # Replace with your actual key
    genai.configure(api_key=API_KEY)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model_gemini = None 

# ----------------- Load Class Names -----------------
with open(classes_path, "r") as f:
    classes = json.load(f)

# ----------------- Model Architecture (Disease Model) -----------------
class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(4))
        self.res1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(4))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(4))
        self.res2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# --- Load Models ---
torch.serialization.add_safe_globals([CNN_NeuralNet])
disease_model = torch.load(model_path, map_location="cpu", weights_only=False)
disease_model.eval()

# --- MODIFICATION: Load Vision Transformer (ViT) as the "Specialist Guard" ---
weights = models.ViT_B_16_Weights.IMAGENET1K_V1
guard_model = models.vit_b_16(weights=weights)
guard_model.eval()

# --- Get the list of all 1000 class names from ImageNet ---
imagenet_classes = weights.meta["categories"]

# --- Create our "master list" of plant-related classes ---
PLANT_RELATED_CLASSES = {
    # Generic ImageNet classes that are plant-related
    'maize', 'corn', 'oak tree', 'maple tree', 'daisy', 'yellow lady\'s slipper',
    'pot', 'flowerpot', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 
    'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 
    'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange',
    'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate',
    'acorn', 'hip', 'rapeseed', 'leaf beetle', 'lacewing', 'earthstar', 'stinkhorn',
    'velvet foot',
    # Classes extracted specifically from your disease list
    'apple', 'blueberry', 'cherry', 'grape', 'peach', 'pepper', 'potato', 
    'raspberry', 'soybean', 'squash', 'tomato'
}


# Confidence threshold for the guard model's prediction.
GUARD_CONFIDENCE_THRESHOLD = 0.10 # Lowered threshold as it's one of 1000 classes


# --- Define separate transforms for each model ---
disease_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- MODIFICATION: Get the official transforms for the ViT model ---
guard_transform = weights.transforms()


# ----------------- Root Route to serve index.html -----------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------- Weather Risk Route -----------------
@app.route("/weather_risk", methods=["POST"])
def weather_risk():
    data = request.get_json()
    lat, lon = data.get('lat'), data.get('lon')
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required."}), 400
    
    WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" # Replace with your key
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        temp, humidity = weather_data['main']['temp'], weather_data['main']['humidity']
        risk = "High" if humidity > 80 and 18 <= temp <= 28 else "Medium" if humidity > 70 and 15 <= temp <= 30 else "Low"
        return jsonify({"temperature": temp, "humidity": humidity, "risk": risk})
    except requests.exceptions.RequestException as e:
        print(f"Weather API Request Error: {e}")
        return jsonify({"error": "Could not fetch weather data."}), 500
    except KeyError:
        print("Weather API Error: Unexpected JSON structure")
        return jsonify({"error": "Could not parse weather data."}), 500

# ----------------- Prediction API (MODIFIED) -----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    risk = request.form.get("risk", "Not available")

    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    # ===============================================================================
    # STEP 1: "Full Knowledge" Guard check using the original Vision Transformer
    # ===============================================================================
    img_tensor_guard = guard_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs_guard = guard_model(img_tensor_guard)
        probabilities_guard = torch.nn.functional.softmax(outputs_guard, dim=1)
        confidence_guard, predicted_idx_guard = torch.max(probabilities_guard, 1)
        
        # Get the name of the predicted class (e.g., 'daisy', 'car', 'pot')
        predicted_class_name = imagenet_classes[predicted_idx_guard.item()]

    # --- For debugging: print what the guard model sees ---
    print(f"Guard Model Prediction: '{predicted_class_name}' with confidence {confidence_guard.item():.2f}")

    # --- MODIFIED Robust check ---
    # Is the predicted class name in our master list of plant-related things?
    # AND is the confidence high enough?
    is_plant = predicted_class_name in PLANT_RELATED_CLASSES and confidence_guard.item() > GUARD_CONFIDENCE_THRESHOLD

    if not is_plant:
        return jsonify({
            "disease": "Not a Plant",
            "confidence": 0,
            "care_plan": f"The model identified this as a '{predicted_class_name.replace('_', ' ')}', which does not seem to be a plant. Please upload a clear image of a plant leaf."
        })

    # ===============================================================================
    # STEP 2: If it's a plant, proceed to disease classification
    # ===============================================================================
    img_t_disease = disease_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs_disease = disease_model(img_t_disease)
        probabilities = torch.nn.functional.softmax(outputs_disease, dim=1)
        confidence, predicted_disease = torch.max(probabilities, 1)

    predicted_class = classes[predicted_disease.item()]
    confidence_score = round(confidence.item() * 100, 2)
    clean_disease_name = predicted_class.replace('_', ' ').replace('___', ' ').title()
    DISEASE_CONFIDENCE_THRESHOLD = 90.0

    if confidence_score < DISEASE_CONFIDENCE_THRESHOLD:
        clean_disease_name = "Unrecognized Condition"
        care_plan = "The model is not confident enough for a reliable diagnosis. Please ensure the uploaded image is a clear, well-lit photo of the affected plant leaf."
    else:
        care_plan = "Could not generate a care plan at this time."
        if model_gemini:
            prompt = f"""
            Generate a farmer-friendly care plan for a plant with "{clean_disease_name}".
            The current environmental risk is "{risk}".

            IMPORTANT: Use the following markdown structure EXACTLY. Do not add any introductory or concluding sentences outside of this structure.

            ### Immediate Actions
            - [Action 1]
            - [Action 2]
            - ...

            ### Preventive Measures
            - [Measure 1]
            - [Measure 2]
            - ...

            ### Long-Term Solutions
            - [Solution 1]
            - [Solution 2]
            - ...
            """
            try:
                gemini_response = model_gemini.generate_content(prompt)
                care_plan = gemini_response.text
            except Exception as e:
                print(f"Error fetching care plan from Gemini: {e}")
                care_plan = "An error occurred while generating the care plan."

    return jsonify({
        "disease": clean_disease_name,
        "confidence": confidence_score,
        "care_plan": care_plan
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

