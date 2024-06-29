from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load your trained Keras model
model1 = load_model("model1.h5", compile=False)
model2 = load_model("model2.h5", compile=False)
model3 = load_model("model3.h5", compile=False)

# Load the labels
class_names1 = open("labels1.txt", "r").readlines()
class_names2 = open("labels2.txt", "r").readlines()
class_names3 = open("labels3.txt", "r").readlines()

def model_predict(img_path, model,class_names):
    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name[2:], confidence_score

@app.route('/')
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Ensure the 'uploads' folder exists
        uploads_folder = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)

        # Save the file to ./uploads
        file_path = os.path.join(uploads_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        class_name1, confidence_score1 = model_predict(file_path, model1,class_names1)
        print(class_name1,len(class_name1))
        if(class_name1[:-1]=="Skin"):
            class_name2,confidence_score2=model_predict(file_path,model2,class_names2)
            print(class_name2)
            if(class_name2[:-1]=="cancer skin"):
                class_name3,confidence_score3=model_predict(file_path,model3,class_names3)
                result=class_name3[:-1]
                return result
            else:
                result=f"Class: {class_name2},Confidence Score: {confidence_score2} The given skin image did not have any type of cancer."
                return result
        else:
            result = f"Class: {class_name1}, Confidence Score: {confidence_score1} The given image is not skin type of image please upload some other image related to it."
            return result

@app.route('/benign')
def benign():
    return render_template('Benign.html')

@app.route('/malignant')
def malignant():
    return render_template('Malignant.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
