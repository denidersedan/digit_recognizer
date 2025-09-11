import numpy as np
import itertools
from layers import init_params, forward_prop, backward_prop, update_params
from utils import one_hot, get_accuracy, load_train_data, evaluate_model

def init_rmsprop_state(W1, b1, W2, b2):
    state = {
        "vW1": np.zeros_like(W1), "vb1": np.zeros_like(b1),
        "vW2": np.zeros_like(W2), "vb2": np.zeros_like(b2),
        "t": 0
    }
    return state

def rmsprop_update(params, grads, state, lr=1e-3, beta=0.9, eps=1e-8):
    for name, vname in (("W1", "vW1"), ("b1", "vb1"), ("W2", "vW2"), ("b2", "vb2")):
        g = grads["d" + name]
        state[vname] = beta * state[vname] + (1 - beta) * (g * g)
        params[name] -= lr * g / (np.sqrt(state[vname]) + eps)
    state["t"] += 1
    return params, state

def gradient_descent(X, Y, iterations, alpha, lambd=0, hidden_size=10, X_val=None, Y_val=None,
                    early_stopping=True, patience=10, min_delta=1e-4,
                    val_check_interval=10, verbose=False, rmsprop_beta=0.9, rmsprop_eps=1e-8):
    m = X.shape[1]
    W1, b1, W2, b2 = init_params(hidden_size)
    one_hot_Y = one_hot(Y)
    
    state = init_rmsprop_state(W1, b1, W2, b2)
    
    best_weights = None
    best_val_acc = 0.0
    wait = 0

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W1, W2, X, one_hot_Y, m, lambd)
        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        params, state = rmsprop_update(params, grads, state, lr=alpha, beta=rmsprop_beta, eps=rmsprop_eps)
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        
        if best_weights is None:
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        
        if i % 50 == 0:
            train_acc = get_accuracy(A2, Y)
            if X_val is not None and not verbose:
                val_acc = evaluate_model(W1, b1, W2, b2, X_val, Y_val)
                print(f"Iter {i}: Train {train_acc:.4f}, Val {val_acc:.4f}")
            else:
                print(f"Iter {i}: Train {train_acc:.4f}")

        # validation check for early stopping
        if early_stopping and (X_val is not None) and (i % val_check_interval == 0):
            if verbose:
                val_acc = evaluate_model(W1, b1, W2, b2, X_val, Y_val)
                print(f"Iteration {i}: Validation accuracy {val_acc:.4f}")
            
            # improvement?
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at iteration {i} (no improvement for {patience} checks).")
                    print(f"Best validation accuracy: {best_val_acc:.4f}")
                    # return best weights seen on validation
                    return best_weights, best_val_acc

    # end loop
    if early_stopping and (X_val is not None):
        # if we never early-stopped, return best_weights (might be final if improved last)
        return best_weights, best_val_acc
    return (W1, b1, W2, b2), None

def random_search(X_train, Y_train, X_val, Y_val,
                  alphas, hidden_sizes, lambdas,
                  iterations=200, n_trials=20, seed=0,
                  early_stopping=True, patience=6, val_check_interval=10,
                  verbose=True):
    
    rng = np.random.default_rng(seed)
    all_combos = list(itertools.product(alphas, hidden_sizes, lambdas))
    total = len(all_combos)
    n_trials = min(n_trials, total)

    # sample unique combos
    combos = rng.choice(np.arange(total), size=n_trials, replace=False)
    results = []
    best = {"params": None, "acc": -1}

    if verbose:
        print(f"Random-search: sampling {n_trials}/{total} combos (iterations={iterations})")

    for idx in combos:
        # we shouldn't take updated parameters here because then as parameters improve
        # results will get biased
        alpha, hidden_size, lambd = all_combos[int(idx)]
        if verbose:
            print(f"\nTrial: alpha={alpha}, hidden_size={hidden_size}, lambda={lambd}")

        # training only on X_train and evaluating on X_val
        trained, val_acc = gradient_descent(
            X_train, Y_train,
            iterations=iterations, alpha=alpha, lambd=lambd, hidden_size=hidden_size,
            X_val=X_val, Y_val=Y_val,
            early_stopping=early_stopping, patience=patience, val_check_interval=val_check_interval,
            verbose=verbose
        )

        results.append(((alpha, hidden_size, lambd), val_acc))
        if verbose:
            print(f" Current validation Accuracy: {val_acc:.4f}")

        if val_acc > best["acc"]:
            best["acc"] = val_acc
            best["params"] = {"alpha": alpha, "hidden_size": hidden_size, "lambd": lambd}
            W1, b1, W2, b2 = trained
            np.savez("models/random_search_params.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    
    if verbose:
        print(f"\n Best combo: {best['params']} with accuracy: {best['acc']:.4f}")
    return best, results

def run():
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = load_train_data()

    # random search
    best, _ = random_search(
        X_train, Y_train, X_dev, Y_dev,
        alphas=[1e-3, 3e-3, 1e-2, 3e-2],
        hidden_sizes=[64, 128, 256],
        lambdas=[0, 0.001, 0.01, 0.1, 1],
        iterations=500,
        n_trials=5,
        patience=10
    )
    best_params = best["params"]

    # retrain full
    X_full = np.concatenate([X_train, X_dev], axis=1)
    Y_full = np.concatenate([Y_train, Y_dev], axis=0)

    weights, _ = gradient_descent(
        X_full, Y_full,
        iterations=1000,
        alpha=best_params["alpha"],
        lambd=best_params["lambd"],
        hidden_size=best_params["hidden_size"],
        early_stopping=False,
        rmsprop_beta=0.9,
        rmsprop_eps=1e-8,
        verbose=True
    )

    W1, b1, W2, b2 = weights

    np.savez("models/params.npz", W1=W1, b1=b1, W2=W2, b2=b2, **best_params)

    final_acc = evaluate_model(W1, b1, W2, b2, X_test, Y_test)
    print("Final model accuracy:", final_acc)
    print("\nBest hyperparameters:", best_params)
