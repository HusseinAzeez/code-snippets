"""
    Autour: Eraser (ตะวัน)
"""

# Standard library imports
import time

# Third-party imports
import jsonpickle
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response
from flask_cors import CORS
from keras import backend as K
from keras.models import load_model

# Local imports
from src.features.preprocessing import Preprocessing
from src.models.custom_layers import PoolHelper, LRN2D
from src.utils.utils import download_pdf, convert_pdf, split_filename
from src.utils.utils import create_file_list, natural_sort

# Initialize the Flask application
APP = Flask(__name__)
CORS(APP, supports_credentials=True)


def get_model():
    """Load all the models (length, single, double) weights and architecture
        from models directory ('./models/saved_models)

        Args:
            None.

        Return:
            l_model: The length model architecture and weights.
            s_model: The single model architecture and weights.
            d_model: The double model architecture and weights.
            graph: The models graph.
    """
    l_model = load_model('./models/length_mix.h5',
                         custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    s_model = load_model('./models/single_mix.h5',
                         custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    d_model = load_model('./models/double_mix.h5',
                         custom_objects={'PoolHelper': PoolHelper(), 'LRN2D': LRN2D()})
    graph = tf.get_default_graph()
    print('----------------------------Models are loaded successfully ----------------------------')

    return l_model, s_model, d_model, graph


def prepare_image(image):
    """Performers all the preprocessing (Text detection, deskewing, border deletion,
        region cropping, and segmentation) (See, preporcessing.py)

        Args:
            image: The full image that converted from the downloaded PDF.

        Return:
            None.
    """
    preprocessing = Preprocessing()
    preprocessing.clear_images()
    preprocessing.crop(image)
    preprocessing.segment()


@APP.route('/api/predict', methods=['POST'])
def predict():
    """Performers all the preprocessing (Text detection, deskewing, border deletion,
        region cropping, and segmentation) (See, preporcessing.py)

        Args:
            None.

        Return:
            Response: In a JSON format containing all the predications for the PDF.
    """
    START_TIME = time.time()
    # Create new dictionary to hold the response
    response = {"status": "fail"}
    response["predictions"] = []
    # Define the image size (This model trained using a 64x64 px images)
    img_rows, img_cols = 64, 64
    # Downlaod the PDF from the linked send with the request
    download_pdf(request.data)
    # Convert the PDF into image
    convert_pdf('./data/raw/full.pdf')
    # Read the converted image
    full = cv2.imread('./data/raw/full.tiff', cv2.IMREAD_GRAYSCALE)
    # Image preprocessing (see, preprocessing.py)
    prepare_image(full)
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
    print("--- {} seconds ---".format(time.time() - START_TIME))
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


if __name__ == '__main__':
    LENGTH_MODEL, SINGLE_MODEL, DOUBLE_MODEL, GRAPH = get_model()
    APP.run(host="192.168.1.25", port=5010)
