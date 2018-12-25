from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import base64
import os
import io
import re
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from src.preprocessing import Preprocessing
from src.custom_layers import PoolHelper, LRN2D

# Initialize the Flask application
app = Flask(__name__)


def create_file_list(path, extension='.png'):
    # Useful function
    file_list = []
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(extension):
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list


def natural_sort(s):
    """
        Sort the given list in the way that humans expect.
    """
    numbers = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(numbers, s)]


def get_model():
    model = load_model(
        './lib/models/length3.h5', custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    print('Model loaded')

    return model


def prepare_images(image):
    preprocessing = Preprocessing()
    preprocessing.clear_images()
    preprocessing.crop(image)
    preprocessing.segment()


# route http posts to this method
@app.route('/api/predict', methods=['POST'])
def test():
    img_rows, img_cols = 64, 64
    req = request
    # convert string of image data to uint8
    nparr = np.fromstring(req.data, np.uint8)
    # decode image
    full = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # do some fancy processing here....
    prepare_images(full)
    files = create_file_list('./data/digits')
    files.sort(key=natural_sort)

    for file in files:
        # read the image
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        if K.image_data_format() == 'channels_first':
            img = img.reshape(1, 1, img_rows, img_cols)
        else:
            img = img.reshape(1, img_rows, img_cols, 1)

        # Standardization the image
        mean_px = img.mean().astype(np.float32)
        std_px = img.std().astype(np.float32)
        img = (img - mean_px)/(std_px)

        length_predicaion = model.predict(img)
        length = np.argmax(length_predicaion, axis=1)

        # build a response dict to send back to client
        response = {'Message Server': 'image received. size={}x{}' 'Length= {}'.format(
            img.shape[1], img.shape[0], length)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == '__main__':
    global model
    model = get_model()
    app.run(port=5000)
