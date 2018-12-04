import keras
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import backend as K
from custom_layers import PoolHelper, LRN2D
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflowjs as tfjs
import numpy as np
import pandas as pd
import cv2


def load_data():

    # Number of classes
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 64, 64

    # Read the entire dataset
    dataset = pd.read_csv('./datasets/full_single_mix.csv')

    # Print out the shape of the dataset
    print('Dataset Shape: ', dataset.shape)

    # Assign x to the features and y to the labels
    x = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    # Split the dataset into 4 sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=0)

    print(x_train.shape, y_train.shape, x_val.shape,
          y_val.shape, x_test.shape, y_test.shape)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Return the datasets
    return x_train, y_train, x_val, y_val, x_test, y_test


def create_model(weights_path=None):

    input = Input(shape=(1, 64, 64))

    conv1_7x7_s2 = Convolution2D(64, kernel_size=(
        7, 7), strides=3, padding='same', activation='relu', name='conv1/7x7_s2')(input)

    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

    pool1_helper = PoolHelper()(conv1_zero_pad)

    pool1_3x3_s1 = MaxPooling2D(pool_size=(
        3, 3), strides=2, padding='same', name='pool1/3x3_s1')(pool1_helper)

    pool1_norm1 = LRN2D(name='pool1/norm1')(pool1_3x3_s1)

    conv2_3x3_reduce = Convolution2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     name='conv2/3x3_reduce')(pool1_norm1)

    conv2_3x3 = Convolution2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                              name='conv2/3x3')(conv2_3x3_reduce)

    pool2_3x3_s1 = MaxPooling2D(pool_size=(
        3, 3), strides=2, padding='same', name='pool2/3x3_s1')(conv2_3x3)

    loss1_flat = Flatten()(pool2_3x3_s1)

    loss1_fc = Dense(128, activation='relu', name='loss1/fc')(loss1_flat)

    loss1_drop_fc = Dropout(0.5)(loss1_fc)

    loss1_classifier = Dense(10, name='loss1/classifier')(loss1_drop_fc)

    loss1_classifier_act = Activation('softmax', name='prob')(loss1_classifier)

    leNet = Model(inputs=input, outputs=loss1_classifier_act)

    return leNet


def train_evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test):
    batch_size = 256
    epochs = 50

    # Callbacks
    # Tensorboard callback
    tensor_board = TensorBoard(
        log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # Earily stopping callback to prevent the model from overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

    # Setup SGD parameters
    sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9)

    # Complie the model with SGD
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['acc'])

    # Train the model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[tensor_board, early_stopping])

    # Test the model
    score = model.evaluate(x_test, y_test, verbose=0)

    # Print the prediction for the test set
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = create_model()
    train_evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test)

    # Save the model as h5 format
    model.save('./models/single_mix.h5')

    # Convert the Keras model to TensoflowJs model
    tfjs.converters.save_keras_model(model, "./models/model_single_mix_js")

    print("Saved model to disk")
