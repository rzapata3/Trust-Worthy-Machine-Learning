import tensorflow as tf
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPool2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from tensorflow.python.ops.init_ops_v2 import Initializer



def load_mnist_data():
    # load MNIST Data
    (trainX, trainY), (testX, testY) = datasets.mnist.load_data()

    # rescale the images from [0,255] to the [0.0, 1.0] range.
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    
    # one hot encode categorical variables.
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY


def process_pixels(train, test):
    train_norm = train.astype("float32")
    test_norm = test.astype('float32')
    
    # norm vals from 0-1
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    
    return train_norm, test_norm


def create_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), 
                            activation='relu',
                            kernel_initializer='he_uniform',
                            input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, 
                           activation='relu', 
                           kernel_initializer='he_uniform'))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def randomize(x, y):
    permutation = np.random.permutation(y,shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation, :]
    
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    
    return x_batch, y_batch

    
def weight_variable(shape):
    initer = tf.truncated_normal_initializer(stddev=0.01)
    
    return tf.get_variable("W",
                           dtype = tf.float32,
                           shape = shape,
                           initializer = initer)
    
def bias_variable(shape):
    initial = tf.constant(0.,
                          shape = shape,
                          dtype = tf.float32)
    return tf.get_variable("b",
                           dtype = tf.float32,
                           initializer = initial)
    
    
batch_size = 64
input_dim = 28
units = 64
output_size = 10  # labels are from 0 to 9

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model
    
    
    
    



def evaluate_cnn_model(x_input, y_input, n_folds = 5):
    scores, histories = list(), list()
    
    # organize 5 fold cross validation
    kfold = KFold(n_folds,
                  shuffle=True,
                  random_state=1)
    
    # enumerate splits
    for trainx_ix, test_ix in kfold.split(x_input):
        cnn_model = create_cnn()
        trainX, trainY, testX, testY = x_input[trainx_ix], y_input[trainx_ix], x_input[test_ix], y_input[test_ix]
        
        # fit model
        history = cnn_model.fit(trainX, 
                            trainY,
                            epochs=10,
                            batch_size=32,
                            validation_data=(testX, testY),
                            verbose=0)
        
        # evaluate model
        _, acc = cnn_model.evaluate(testX, testY, verbose=0)
        #print('> %.3f' % (acc * 100.0))
        acc = acc * 100.0
        print(f"CNN Accuracy: {acc:.2f}%")
        
        # save the acc score
        scores.append(acc)
        histories.append(history)
        
        return scores, histories

def evaluate_rnn_model():
    
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample, sample_label = x_train[0], y_train[0]
    
    model = build_model(allow_cudnn_kernel=True)

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy
                  (from_logits=True),
                  optimizer="sgd",
                  metrics=["accuracy"],)


    history = model.fit(x_train, 
                       y_train, 
                       validation_data=(x_test, y_test), 
                       batch_size=batch_size, 
                       epochs=1)
    acc = history.history["accuracy"][0] * 100
    print(f"RNN Accuracy: {acc:.2f}%")
    
    return history
    
  
if __name__ == "__main__":
    trainX, trainY, testX, testY = load_mnist_data()
    trainX, testX = process_pixels(trainX, testX)
    acc = evaluate_rnn_model()
    scores, histories = evaluate_cnn_model(trainX, trainY)