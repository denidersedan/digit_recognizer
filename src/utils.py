import numpy as np
import pandas as pd
from layers import forward_prop
import matplotlib.pyplot as plt

def load_train_data(path="data/train.csv", val_ratio=0.1, test_ratio = 0.1, seed=123, return_indices=False):
    # load once as numpy for speed
    df = pd.read_csv(path)
    data = df.to_numpy()       # shape (m_total, n_cols)  labels in col 0

    m_total = data.shape[0]
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    # reproducible shuffling
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m_total)

    # split indices
    n_val = int(np.floor(m_total * val_ratio))
    n_test = int(np.floor(m_total * test_ratio))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    # extract labels and features (features scaled to [0,1])
    labels = data[:, 0].astype(int)               # shape (m_total,)
    features = data[:, 1:].astype(np.float32) / 255.0  # shape (m_total, 784)

    # select splits and transpose to (features, examples)
    X_train = features[train_idx].T               # shape (784, m_train)
    Y_train = labels[train_idx]                   # shape (m_train,)
    X_val = features[val_idx].T                   # shape (784, m_val)
    Y_val = labels[val_idx]                       # shape (m_val,)
    X_test = features[test_idx].T
    Y_test = labels[test_idx]

    if return_indices:
        return X_train, Y_train, X_val, Y_val, (train_idx, val_idx)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_test_data(path='data/test.csv'):
    df = pd.read_csv(path)
    data = df.to_numpy()
    features = data.astype(np.float32) / 255.0
    X_predict = features.T
    return X_predict

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def get_predictions(A2):
    # Looks down on each column and return row index with max value
    return np.argmax(A2, axis=0) 

def get_accuracy(A2, Y):
    return np.mean(get_predictions(A2) == Y)

def evaluate_model(W1, b1, W2, b2, X_val, Y_val):
    _, _, _, A2, _ = forward_prop(X_val, W1, b1, W2, b2, training=False)
    predictions = get_predictions(A2)
    return np.mean(predictions == Y_val)

def check_balanced(Y):
    classes, counts = np.unique(Y, return_counts=True)
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Class distribution")
    plt.show()

    
def run():
    _, Y_train, _, Y_val, _, _ = load_train_data()
    check_balanced(Y_train)
    check_balanced(Y_val)
