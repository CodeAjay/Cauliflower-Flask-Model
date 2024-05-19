from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained cauliflower classifier model
cauliflower_model = tf.keras.models.load_model('path/to/cauliflower_model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/classify_cauliflower', methods=['POST'])
def classify_cauliflower():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make prediction using the loaded model
        prediction = cauliflower_model.predict(img_array)

        # Return the prediction result
        if prediction[0] > 0.5:
            return jsonify({'result': 'Cauliflower'})
        else:
            return jsonify({'result': 'Non-Cauliflower'})

if __name__ == '__main__':
    app.run(debug=True)
