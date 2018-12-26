"""
    autour: Eraser (ตะวัน)
"""
# Standard librray imports
import re
import os

# Third-part imports
import jsonpickle
import cv2
import numpy as np
from flask import Flask, request, Response
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

# Local imports
from src.preprocessing import Preprocessing
from src.custom_layers import PoolHelper, LRN2D

# Initialize the Flask application
app = Flask(__name__)


def create_file_list(path, extension='.png'):
    """
        Creates a files list containing all the files
        in the specified directory with the provided file extension

        Attributes:
            path: The path to the directory containing the required files
            extension: The files extensions (default: .png)

        Return:
            file_list: Python list containing all the files
    """
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

        Attributes:
            s: The path to the directory containing the required files

        Return:
            file_list: Python list containing all the files
    """
    numbers = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(numbers, s)]


def split_filename(path):
    """
        Split the provided file paths into (Region corresponding to row order in the original PDF,
        the file name, and the file extension)

        Attributes:
            path: The path to the directory containing the required files.

        Return:
            region: The number corsponding to row order in the original PDF.
            name: The file name (A number).
            extenstion: File extenstion (Default .png)
    """
    filename = os.path.basename(path)
    name, extension = os.path.splitext(filename)
    region = name.split('.')[0]

    return region, name, extension


def get_model():
    """
        Load all the models (length, single, double) weights and architecture
        from models directory ('./lib/models)

        Attributes:
            None.

        Return:
            l_model: The length model architecture and weights.
            l_graph: The length model graph.
            s_model: The single model architecture and weights.
            s_graph: The single model graph.
            d_model: The double model architecture and weights.
            d_graph: The double model graph.
    """
    l_model = load_model(
        './lib/models/length3.h5', custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    l_graph = tf.get_default_graph()

    s_model = load_model('./lib/models/single_mix3.h5',
                         custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    s_graph = tf.get_default_graph()

    d_model = load_model('./lib/models/double2.h5',
                         custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    d_graph = tf.get_default_graph()
    print('----------------------------Models are loaded successfully ----------------------------')

    return l_model, l_graph, s_model, s_graph, d_model, d_graph


def prepare_images(image):
    """
        Performers all the preprocessing (Text detection, deskewing, border deletion,
        region cropping, and segmentation) (See, preporcessing.py)

        Attributes:
            Image: The full image that received through the POST request.

        Return:
            None.
    """
    preprocessing = Preprocessing()
    preprocessing.clear_images()
    preprocessing.crop(image)
    preprocessing.segment()


# Route http posts to this method
@app.route('/api/predict', methods=['POST'])
def predict():
    """
        Performers all the preprocessing (Text detection, deskewing, border deletion,
        region cropping, and segmentation) (See, preporcessing.py)

        Attributes:
            None.

        Return:
            Response: In a JSON format containing all the predications for the PDF.
    """
    response = {"success": False}
    response["predictions"] = []
    img_rows, img_cols = 64, 64

    # Convert string of image data to uint8
    np_array = np.fromstring(request.data, np.uint8)
    # Decode image
    full = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    # Image preprocessing (see, preprocessing.py)
    prepare_images(full)

    # Create a list containing all the digits
    files = create_file_list('./data/digits')
    # Sort the images inside the list
    files.sort(key=natural_sort)

    for file in files:

        # Split the file name into region, file name, extention
        region, _, _ = split_filename(file)

        # Read the image
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        # Checking Kreas backend
        if K.image_data_format() == 'channels_first':
            img = img.reshape(1, 1, img_rows, img_cols)
        else:
            img = img.reshape(1, img_rows, img_cols, 1)

        # Standardization the image
        mean = img.mean().astype(np.float32)
        std = img.std().astype(np.float32)
        img = (img - mean)/(std)

        # Preform predictions for length model
        with length_graph.as_default():
            length_predicaion = length_model.predict(img)
            length = np.argmax(length_predicaion, axis=1)

        # Preform predictions for single model
        with length_graph.as_default():
            single_predicaion = single_model.predict(img)
            singles = np.argmax(single_predicaion, axis=1)

        # Preform predictions for double model
        with double_graph.as_default():
            double_predicaion = double_model.predict(img)
            doubles = np.argmax(double_predicaion, axis=1)

        # Build the response dict to send back to client
        for _, len_pred in enumerate(length):
            if len_pred == 0:
                for _, single in enumerate(singles):
                    res = {'digit': 'region= {} ' ' length= {}' ' single= {}'.format(
                        region, len_pred, single)}
                    response["predictions"].append(res)
            else:
                for _, double in enumerate(doubles):
                    if double < 10:
                        res = {'digit': 'region= %s ' ' length= %d' ' double= %.02d' % (
                            region, len_pred, double)}
                        response["predictions"].append(res)
                    else:
                        res = {'digit': 'region= {} ' ' length= {}' ' double= {}'.format(
                            region, len_pred, double)}
                        response["predictions"].append(res)

    response["success"] = True

    # Encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# Start flask app
if __name__ == '__main__':
    length_model, length_graph, single_model, single_graph, double_model, double_graph = get_model()
    app.run(port=5000)
