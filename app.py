"""
    Autour: Eraser (ตะวัน)
"""

# Standard librray imports
import re
import os

# Third-party imports
import jsonpickle
import cv2
import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

# Local imports
from src.preprocessing import Preprocessing
from src.custom_layers import PoolHelper, LRN2D

# Initialize the Flask application
APP = Flask(__name__)
CORS(APP, supports_credentials=True)


def create_file_list(path, extension='.png'):
    """Creates a files list containing all the files
        in the specified directory with the provided file extension

        Args:
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
    """Sort the given list in the way that humans expect.

        Args:
            s: The path to the directory containing the required files

        Return:
            file_list: Python list containing all the files
    """
    numbers = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(numbers, s)]


def split_filename(path):
    """Split the provided file paths into (Region corresponding to row order in the original PDF,
        the file name, and the file extension)

        Args:
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
    """Load all the models (length, single, double) weights and architecture
        from models directory ('./lib/models)

        Args:
            None.

        Return:
            l_model: The length model architecture and weights.
            s_model: The single model architecture and weights.
            d_model: The double model architecture and weights.
            graph: The models graph.
    """
    l_model = load_model(
        './lib/models/length3.h5', custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    s_model = load_model('./lib/models/single_mix3.h5',
                         custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    d_model = load_model('./lib/models/double2.h5',
                         custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    graph = tf.get_default_graph()
    print('----------------------------Models are loaded successfully ----------------------------')

    return l_model, s_model, d_model, graph


def prepare_images(image):
    """Performers all the preprocessing (Text detection, deskewing, border deletion,
        region cropping, and segmentation) (See, preporcessing.py)

        Args:
            Image: The full image that received through the POST request.

        Return:
            None.
    """
    preprocessing = Preprocessing()
    preprocessing.clear_images()
    preprocessing.crop(image)
    preprocessing.segment()


# Route http posts to this method

@APP.route('/api/predict', methods=['POST'])
def predict():
    """Performers all the preprocessing (Text detection, deskewing, border deletion,
        region cropping, and segmentation) (See, preporcessing.py)

        Args:
            None.

        Return:
            Response: In a JSON format containing all the predications for the PDF.
    """
    # Create new dictionary to hold the response
    response = {"status": "fail"}
    response["predictions"] = []
    # Define the image size (This model trained using a 64x64 px images)
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
        # Preform predictions for length model, single model, and double model
        with GRAPH.as_default():
            # Predictions for length model
            length_predicaion = LENGTH_MODEL.predict(img)
            length = np.argmax(length_predicaion, axis=1)
            # Predictions for single model
            single_predicaion = SINGLE_MODEL.predict(img)
            singles = np.argmax(single_predicaion, axis=1)
            # Predictions for double model
            double_predicaion = DOUBLE_MODEL.predict(img)
            doubles = np.argmax(double_predicaion, axis=1)
        # Build the response dictionary to send back to client
        for _, len_pred in enumerate(length):
            if len_pred == 0:
                for _, single in enumerate(singles):
                    res = {"row": '{}'.format(region),
                           "pairs": '{}'.format(len_pred),
                           "single": '{}'.format(single)}
                    response["predictions"].append(res)
            else:
                for _, double in enumerate(doubles):
                    if double < 10:
                        res = {"row": '{}'.format(region),
                               "pairs": '{}'.format(len_pred),
                               "double": '%.02d' % (double)}
                        response["predictions"].append(res)
                    else:
                        res = {"row": '{}'.format(region),
                               "pairs": '{}'.format(len_pred),
                               "single": '{}'.format(double)}
                        response["predictions"].append(res)
    # Set the success flag into true
    response["status"] = "success"
    # Encode response using jsonpickle
    predication_response = jsonpickle.encode(response)

    return Response(response=predication_response, status=200, mimetype="application/json")


@APP.errorhandler(404)
def error404(error):
    """Error handler for 404.
        Args:
            error: The error code get passed automatically from the app.
        Return:
            response: A JSON response containing an error message and the error code.
    """
    error_message = '''Erro code {} This enterd route does not exists.
    This server has only one route / api/predict'''.format(error)
    error_response = jsonpickle.encode(error_message)
    return Response(response=error_response, status=404, mimetype="application/json")


@APP.errorhandler(405)
def error405(error):
    """Error handler for 405.
        Args:
            error: The error code get passed automatically from the app.
        Return:
            response: A JSON response containing an error message and the error code.
    """
    error_message = '''Erro code {} The requested HTTP method is not
    allowed for this server.'''.format(error)
    error_response = jsonpickle.encode(error_message)
    return Response(response=error_response, status=405, mimetype="application/json")


@APP.errorhandler(500)
def error500(error):
    """Error handler for 500.
        Args:
            error: The error code get passed automatically from the app.
        Return:
            response: A JSON response containing an error message and the error code.
    """
    error_message = '''Erro code {} Image not found or it's too small.
    Please selet a valid image.'''.format(error)
    error_response = jsonpickle.encode(error_message)
    return Response(response=error_response, status=500, mimetype="application/json")


# Start flask APP
if __name__ == '__main__':
    LENGTH_MODEL, SINGLE_MODEL, DOUBLE_MODEL, GRAPH = get_model()
    APP.run(host="192.168.1.25", port=5010)
