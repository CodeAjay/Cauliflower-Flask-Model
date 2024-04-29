from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# Load the trained Keras model
model = tf.keras.models.load_model('trained_model.keras')
class_names = ['Bacterial spot rot', 'Black Rot', 'Disease Free', 'Downy Mildew']

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

if __name__ == '__main__':
    app.run(debug=True)
