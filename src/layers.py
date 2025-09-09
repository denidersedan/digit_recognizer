import numpy as np

def init_params(hidden_size=10):
    W1 = np.random.rand(hidden_size, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.rand(10, hidden_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def relu_deriv(Z):
    return Z > 0

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0)

def forward_prop(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, A2, W1, W2, X, one_hot_Y, m, lambd):
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T) + (lambd/m) * W2
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * relu_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T) + (lambd/m) * W1
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2
