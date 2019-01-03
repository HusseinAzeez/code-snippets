"""
    Autour: Eraser (ตะวัน)
"""

# Local imports
from src.single_model import SingleModel
from src.length_model import LengthModel
from src.double_model import DoubleModel


def train_length(path):
    """Trains, evaluates, and save the length classifier.

        Args:
            path: A valid path to the full dataset.

        Return:
            None.
    """
    length = LengthModel()
    x_train, y_train, x_val, y_val, x_test, y_test, input_shape = length.load_data(
        path)
    model = length.create_model(input_shape)
    length.train_evaluate_model(
        model, x_train, y_train, x_val, y_val, x_test, y_test)


def train_single(path):
    """Trains, evaluates, and save the single digit (0 pair) classifier.

        Args:
            path: A valid path to the full dataset.

        Return:
            None.
    """
    single = SingleModel()
    x_train, y_train, x_val, y_val, x_test, y_test, input_shape = single.load_data(
        path)
    model = single.create_model(input_shape)
    single.train_evaluate_model(
        model, x_train, y_train, x_val, y_val, x_test, y_test)


def train_double(path):
    """Trains, evaluates, and save the double digit (1 pair) classifier.

        Args:
            path: A valid path to the full dataset.

        Return:
            None.
    """
    double = DoubleModel()
    x_train, y_train, x_val, y_val, x_test, y_test, input_shape = double.load_data(
        path)
    model = double.create_model(input_shape)
    double.train_evaluate_model(
        model, x_train, y_train, x_val, y_val, x_test, y_test)


def select_model(model_number, dataset_path):
    """Reads the user inputs (dataset path, required model) for training.

        Args:
            model_number: The number of the required model for training.
            dataset_path: A valid path to the full dataset.

        Return:
            None.
    """
    if model_number == '0':
        train_length(dataset_path)
    elif model_number == '1':
        train_single(dataset_path)
    elif model_number == '2':
        train_double(dataset_path)
    else:
        print('Invalid Number')


if __name__ == '__main__':
    USER_INPUT_PATH = input(
        '\n---------------- Please insert a valid dataset path ----------------\n path=: ')
    USER_INPUT_MODEL = input('''---------------- Please select a model to train ----------------\n
        0: To train the length model\n
        1: To train the single digit model\n
        2: To trian the double digit model\n model: ''')
    select_model(model_number=USER_INPUT_MODEL, dataset_path=USER_INPUT_PATH)
