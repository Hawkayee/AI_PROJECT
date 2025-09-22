from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model/cnn_model.h5")

# Replace with your dataset classes
classes = ["Healthy", "Early Blight", "Late Blight"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"
    file = request.files["file"]
    filepath = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(filepath)

    # Preprocess
    img = image.load_img(filepath, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    result = classes[class_index]

    return render_template("result.html", prediction=result, image=filepath)

if __name__ == "__main__":
    app.run(debug=True)
