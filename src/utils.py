import numpy as np
import pandas as pd
from layers import forward_prop

def load_train_data(path='data/train.csv', dev_size=1000):
    data = pd.read_csv(path).to_numpy()
    np.random.shuffle(data)
    m, n = data.shape

    data_dev = data[0:dev_size].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n] / 255.

    data_train = data[dev_size:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.

    return X_train, Y_train, X_dev, Y_dev

def load_test_data(path='data/test.csv'):
    data = pd.read_csv(path).to_numpy()
    data = data.T
    X_test = data / 255.
    return X_test

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def get_predictions(A2):
    # Looks down on each column and return row index with max value
    return np.argmax(A2, axis=0) 

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def evaluate_model(W1, b1, W2, b2, X_val, Y_val):
    _, _, _, A2 = forward_prop(X_val, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    print("Predictions:", predictions[:10])
    print("Y_test     :", Y_val[:10])
    print("Shape of predictions:", predictions.shape)
    print("Shape of Y_test     :", Y_val.shape)
    return get_accuracy(predictions, Y_val)
