from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import tensorflow as tf
import cv2
from modelsummary import generate_model_summary


app = Flask(__name__)
CORS(app)

# Load the trained Keras model
model = tf.keras.models.load_model('trained_model_m.keras')
class_names = ['Bacterial spot rot', 'Black Rot', 'Disease Free', 'Downy Mildew']

@app.route('/model_summary', methods=['GET'])
def get_model_summary():
    # Generate the model summary
    summary = generate_model_summary()  # Assuming you have a function to generate the summary
    
    # Ensure that summary is a list of strings
    if not isinstance(summary, list):
        return jsonify({'error': 'Model summary is not in the expected format'}), 500
    
    return jsonify({'summary': summary})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['file']
        # Read the image using OpenCV
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize image to the required input size of the model
        img_resized = cv2.resize(img_rgb, (128, 128))
        # Convert image to array
        input_arr = np.array([img_resized])
        # Make prediction using the loaded model
        prediction = model.predict(input_arr)
        # Get the predicted class index
        result_index = np.argmax(prediction)
        # Get the predicted class name
        predicted_class = class_names[result_index]
        return jsonify({'prediction': predicted_class})

