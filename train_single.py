import keras
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import backend as K
from custom_layers import PoolHelper, LRN
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs
import numpy as np
import pandas as pd
import cv2


def create_leNet(weights_path=None):
    input = Input(shape=(1, 64, 64))

    conv1_7x7_s2 = Convolution2D(64, 7, 7, subsample=(
        3, 3), border_mode='same', activation='relu', name='conv1/7x7_s2')(input)

    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

    pool1_helper = PoolHelper()(conv1_zero_pad)

    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), border_mode='valid', name='pool1/3x3_s2')(pool1_helper)

    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

    conv2_3x3_reduce = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                     name='conv2/3x3_reduce')(pool1_norm1)

    conv2_3x3 = Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                              name='conv2/3x3')(conv2_3x3_reduce)

    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), border_mode='valid', name='pool1/3x3_s2')(conv2_3x3)

    loss1_flat = Flatten()(pool1_3x3_s2)

    loss1_fc = Dense(128, activation='relu', name='loss1/fc')(loss1_flat)

    loss1_drop_fc = Dropout(0.5)(loss1_fc)

    loss1_classifier = Dense(10, name='loss1/classifier')(loss1_drop_fc)

    loss1_classifier_act = Activation('softmax', name='prob')(loss1_classifier)

    leNet = Model(input=input, output=loss1_classifier_act)

    return leNet


if __name__ == "__main__":
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 64, 64

    dataset = pd.read_csv('./full_single.csv')
    print('Dataset Shape: ', dataset.shape)
    x = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    print(x_train.shape, x_test.shape, y_train, y_test.shape)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Standardization
    mean_px = x_train.mean().astype(np.float32)
    std_px = x_train.std().astype(np.float32)
    x_train = (x_train - mean_px)/(std_px)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    tensor_board = TensorBoard(
        log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model = create_leNet()
    sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9)
    print(model.summary())
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensor_board])
    # Test the model
    score = model.evaluate(x_test, y_test, verbose=0)
    tfjs.converters.save_keras_model(model, "./model4_js")
    # out = model.predict(img)  # note: the model has three outputs
    print(np.argmax(score[2]))
