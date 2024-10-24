from flask import Flask, request, jsonify, render_template
import os
import cv2
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model = tf.keras.models.load_model('C:\\Users\\mkura\\CongressionalAppChallenge\\Pharyngitis.hdf5')

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to serve the HTML form
@app.route('/')
def index():
    return render_template('InputImage.html')

# Upload route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Secure the filename and save the file to the upload folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded image
        image = cv2.imread(filepath)
        target_size = (64, 64)
        resized_image = cv2.resize(image, target_size)
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension

        # Get prediction from the model
        prediction = model.predict(resized_image)[0][0]  # Assuming model outputs a probability

        # Prepare detailed messages based on the prediction
        if prediction > 0.5:
            # No pharyngitis detected message
            message = """Good News:
            
Your results indicate that there are no signs of bacterial pharyngitis (strep throat) detected.

What to Do Next:

- Monitor Your Symptoms: While bacterial pharyngitis is not present, you may still experience sore throat due to viral infections or other factors. If symptoms worsen or persist, consider seeking medical advice.

- Stay Hydrated: Drink plenty of fluids. Warm teas and soups can help.

- Rest: Rest your voice and body to help recovery.

- Over-the-Counter Relief: You may use pain relievers or lozenges for discomfort. 

- When to Seek Medical Attention: If you experience high fever, difficulty swallowing, or symptoms that worsen, consult a healthcare provider.

Stay informed and take care of your health!"""
        else:
            # Pharyngitis detected message
            message = """Alert:
            
Your results indicate that you may have bacterial pharyngitis (strep throat).

What to Do Next:

- Consult a Healthcare Professional: Contact a healthcare provider for diagnosis and treatment. Bacterial pharyngitis often requires antibiotics.

- Monitor Your Symptoms: Track symptoms such as fever or difficulty swallowing and inform your provider.

- Stay Hydrated: Drink fluids and rest.

- Avoid Contagion: Prevent spreading the infection by avoiding contact with others, washing hands frequently, and covering your mouth when coughing or sneezing.

- Over-the-Counter Relief: Pain relievers can help ease discomfort. 

- Why Early Detection Matters: Untreated bacterial pharyngitis can lead to serious complications like rheumatic fever. Early detection helps in effective treatment.

Stay informed and take care of your health!"""

        # Return the result as JSON
        return jsonify({
            "success": True,
            "message": message,
            "prediction": float(prediction),
        })

    return jsonify({"success": False, "message": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)