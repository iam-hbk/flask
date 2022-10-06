import math
from flask import Flask, jsonify, request
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


def predict(file):
    # Load the model
    model = load_model("keras_model.h5")

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(file).convert("RGB")
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    with open("labels.txt") as f:
        class_names = f.read().splitlines()
        class_names = [i.split(" ")[1] for i in class_names]
        
    
    confidence_score = prediction[0][index]

    return {"Class": class_names[index], "Confidence": math. floor(confidence_score * 100)}


@app.route('/test',methods = ['POST'])
def test():
    file = request.files['file'].stream
    return jsonify(predict(file))

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
