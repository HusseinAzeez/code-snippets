"""
    autour: Eraser (ตะวัน)
"""

# Third-party imports
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D
from keras.layers import ZeroPadding2D, Dropout, Flatten, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Local imports
from src.custom_layers import PoolHelper, LRN2D


class DoubleModel():
    """
        Double digit model
    """

    def __init__(self):
        """ Double digit constructor
                Attributes:
                    batch_size: Number of batch
                    epochs: Number of cycles the model will take for training
                    num_classes: Number of classes in the dataset
                    img_rows: The image rows number
                    img_cols: The image columns number
        """
        self.batch_size = 256
        self.epochs = 25
        self.num_classes = 100
        self.img_rows = 64
        self.img_cols = 64

    def load_data(self, path):
        """
            Load dataset, split into training, validation, and testing sets

            Attributes:
                path: A path to the dataset
            Return:
                x_train: training set (features)
                y_train: training set (label)
                x_val: validation set (features)
                y_val: validation set (label)
                x_test: testing set (features)
                y_test: testing set (label)
        """
        # Read the entire dataset
        dataset = pd.read_csv(path)
        print(dataset.iloc[:, 0].value_counts())
        # Print out the shape of the dataset
        print('Dataset Shape: ', dataset.shape)

        # Assign x to the features and y to the labels
        features = dataset.iloc[:, 1:].values
        label = dataset.iloc[:, 0].values

        # Split the dataset into 4 sets
        x_train, x_test, y_train, y_test = train_test_split(
            features, label, test_size=0.1, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=0)

        print(x_train.shape, y_train.shape, x_val.shape,
              y_val.shape, x_test.shape, y_test.shape)

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(
                x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_val = x_val.reshape(
                x_val.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(
                x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(
                x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_val = x_val.reshape(
                x_val.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(
                x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)

        # Standardization training set
        mean_px = x_train.mean().astype(np.float32)
        std_px = x_train.std().astype(np.float32)
        x_train = (x_train - mean_px)/(std_px)

        # Standardization validation set
        mean_px = x_val.mean().astype(np.float32)
        std_px = x_val.std().astype(np.float32)
        x_val = (x_val - mean_px)/(std_px)

        # Standardization testing set
        mean_px = x_test.mean().astype(np.float32)
        std_px = x_test.std().astype(np.float32)
        x_test = (x_test - mean_px)/(std_px)

        # Print out the number of training and testing samples
        print(x_train.shape[0], 'train samples')
        print(x_val.shape[0], 'validation samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, self.num_classes)
        y_val = to_categorical(y_val, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        # Return the datasets
        return x_train, y_train, x_val, y_val, x_test, y_test, input_shape

    def create_model(self, input_shape):

        img_shape = Input(shape=input_shape)

        conv1_7x7_s3 = Convolution2D(72, kernel_size=(
            7, 7), strides=3, padding='same', activation='relu', name='conv1/7x7_s3')(img_shape)

        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s3)

        pool1_helper = PoolHelper()(conv1_zero_pad)

        pool1_3x3_s2 = MaxPooling2D(pool_size=(
            3, 3), strides=2, padding='same', name='pool1/3x3_s2')(pool1_helper)

        pool1_norm1 = LRN2D(name='pool1/norm1')(pool1_3x3_s2)

        conv2_3x3_s1 = Convolution2D(192, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     name='conv2/3x3_s1')(pool1_norm1)

        conv3_3x3_s1 = Convolution2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     name='conv3/3x3_s1')(conv2_3x3_s1)

        pool2_3x3_s1 = MaxPooling2D(pool_size=(
            3, 3), strides=2, padding='same', name='pool2/3x3_s2')(conv3_3x3_s1)

        loss1_flat = Flatten()(pool2_3x3_s1)

        loss1_fc = Dense(1024, activation='relu', name='loss1/fc')(loss1_flat)

        loss1_drop_fc = Dropout(0.5)(loss1_fc)

        loss1_classifier = Dense(100, name='loss1/classifier')(loss1_drop_fc)

        loss1_classifier_act = Activation(
            'softmax', name='prob')(loss1_classifier)

        model = Model(inputs=img_shape, outputs=loss1_classifier_act)

        return model

    def train_evaluate_model(self, model, x_train, y_train, x_val, y_val, x_test, y_test):
        # Callbacks
        # Tensorboard callback
        tensor_board = TensorBoard(
            log_dir='../lib/Graph', histogram_freq=0, write_graph=True, write_images=True)

        # Earily stopping callback to prevent the model from overfitting.
        early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        # Model checkpoint save the best weight only
        model_checkpoint = ModelCheckpoint(
            filepath='../lib/models/checkpoints/best_double_model.h5', monitor='val_loss', save_best_only=True)

        # Setup SGD parameters
        sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9)

        # Complie the model with SGD
        model.compile(optimizer=sgd, loss='categorical_crossentropy',
                      metrics=['acc'])

        # Train the model
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(x_val, y_val),
                  callbacks=[tensor_board, early_stopping, model_checkpoint])

        # Test the model
        score = model.evaluate(x_test, y_test, verbose=0)
        # Save the model as h5 format
        model.save('./models/double2.h5')
        print("Saved model to disk")
        # Print the prediction for the test set
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
