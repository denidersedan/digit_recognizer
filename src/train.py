import numpy as np
from layers import init_params, forward_prop, backward_prop, update_params
from utils import one_hot, get_predictions, get_accuracy, load_train_data, evaluate_model

def find_best_hidden_size(X_train, Y_train, X_val, Y_val, alpha, iterations, lambd, hidden_sizes):
    best_size = None
    best_accuracy = 0
    results = []

    for size in hidden_sizes:
        print(f"\nTraining with hidden layer size = {size}")
        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations, alpha, lambd, hidden_size=size)
        acc = evaluate_model(W1, b1, W2, b2, X_val, Y_val)
        results.append((size, acc))
        print(f"Validation Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_size = size

    print(f"\nBest hidden size: {best_size} with accuracy: {best_accuracy:.4f}")
    return best_size, results


def find_best_lambda(X_train, Y_train, X_val, Y_val, alpha, iterations, lambdas, hidden_size):
    best_lambda = None
    best_accuracy = 0
    results = []

    for lambd in lambdas:
        print(f"Training with lambda = {lambd}")
        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations, alpha, lambd, hidden_size)
        acc = evaluate_model(W1, b1, W2, b2, X_val, Y_val)
        results.append((lambd, acc))
        print(f"Validation Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_lambda = lambd

    print(f"\n Best lambda: {best_lambda} with accuracy: {best_accuracy:.4f}")
    return best_lambda, results

def gradient_descent(X, Y, iterations, alpha, lambd=0, hidden_size=10):
    m = X.shape[1]
    W1, b1, W2, b2 = init_params(hidden_size)
    one_hot_Y = one_hot(Y)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W1, W2, X, one_hot_Y, m, lambd)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            acc = get_accuracy(get_predictions(A2), Y)
            print(f"Iteration {i}: Accuracy {acc:.4f}")

    return W1, b1, W2, b2

def run():
    X_train, Y_train, X_dev, Y_dev = load_train_data()

    dev_accuracy = evaluate_model(W1, b1, W2, b2, X_dev, Y_dev)
    print("Dev Accuracy:", dev_accuracy)
    
    hidden_sizes = [64, 110, 128]
    best_size, results = find_best_hidden_size(X_train, Y_train, X_dev, Y_dev, alpha=0.1, iterations=500, lambd=0, hidden_sizes=hidden_sizes)
    print("\nBest hidden layer size: ", best_size)
    
    lambdas = [0.01, 0.1, 1, 2]
    best_lambda, results = find_best_lambda(X_train, Y_train, X_dev, Y_dev, 0.1, 501, lambdas, best_size)
    print("\nBest lambda: ", best_lambda)
    
    X_full = np.concatenate([X_train, X_dev], axis=1)
    Y_full = np.concatenate([Y_train, Y_dev], axis=0)
    
    W1, b1, W2, b2 = gradient_descent(X_full, Y_full, alpha=0.1, iterations=500, lambd=best_lambda, hidden_size=best_size)

    np.savez("models/params.npz", W1=W1, b1=b1, W2=W2, b2=b2)