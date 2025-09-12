import numpy as np
import itertools
from layers import init_params, forward_prop, backward_prop, update_params
from utils import one_hot, get_accuracy, load_train_data, evaluate_model

def init_adam_state(W1, b1, W2, b2):
    return {
        "mW1": np.zeros_like(W1), "vW1": np.zeros_like(W1),
        "mb1": np.zeros_like(b1), "vb1": np.zeros_like(b1),
        "mW2": np.zeros_like(W2), "vW2": np.zeros_like(W2),
        "mb2": np.zeros_like(b2), "vb2": np.zeros_like(b2),
        "t": 0
    }

def adam_update_inplace(W1, b1, W2, b2, dW1, db1, dW2, db2, state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    # first moments
    state["mW1"] = beta1*state["mW1"] + (1-beta1)*dW1
    state["mb1"] = beta1*state["mb1"] + (1-beta1)*db1
    state["mW2"] = beta1*state["mW2"] + (1-beta1)*dW2
    state["mb2"] = beta1*state["mb2"] + (1-beta1)*db2
    # second moments
    state["vW1"] = beta2*state["vW1"] + (1-beta2)*(dW1*dW1)
    state["vb1"] = beta2*state["vb1"] + (1-beta2)*(db1*db1)
    state["vW2"] = beta2*state["vW2"] + (1-beta2)*(dW2*dW2)
    state["vb2"] = beta2*state["vb2"] + (1-beta2)*(db2*db2)

    # bias-corrected
    mw1 = state["mW1"] / (1 - beta1**t)
    mb1 = state["mb1"] / (1 - beta1**t)
    mw2 = state["mW2"] / (1 - beta1**t)
    mb2 = state["mb2"] / (1 - beta1**t)

    vw1 = state["vW1"] / (1 - beta2**t)
    vb1 = state["vb1"] / (1 - beta2**t)
    vw2 = state["vW2"] / (1 - beta2**t)
    vb2 = state["vb2"] / (1 - beta2**t)

    W1 -= lr * mw1 / (np.sqrt(vw1) + eps)
    W2 -= lr * mw2 / (np.sqrt(vw2) + eps)
    b1 -= lr * mb1 / (np.sqrt(vb1) + eps)
    b2 -= lr * mb2 / (np.sqrt(vb2) + eps)

    return W1, b1, W2, b2, state


def get_minibatches(X, Y, batch_size=128, shuffle=True):
    m = X.shape[1]
    indices = np.arange(m)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        batch_idx = indices[start:end]
        yield X[:, batch_idx], Y[batch_idx]

def gradient_descent(X, Y, epochs, alpha, lambd=0, hidden_size=10, X_val=None, Y_val=None,
                    early_stopping=True, patience=10, min_delta=1e-4,
                    val_check_interval=10, verbose=False, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8):
    W1, b1, W2, b2 = init_params(hidden_size)
    
    state = init_adam_state(W1, b1, W2, b2)
    
    best_weights = None
    best_val_acc = 0.0
    wait = 0

    for epoch in range(epochs):
        for X_batch, Y_batch in get_minibatches(X, Y, batch_size=128):
            m_batch = X_batch.shape[1]
            one_hot_Y_batch = one_hot(Y_batch)
            Z1, A1, Z2, A2, mask1 = forward_prop(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W1, W2, X_batch, one_hot_Y_batch, m_batch, mask1, lambd)
            W1, b1, W2, b2, state = adam_update_inplace(W1, b1, W2, b2, dW1, db1, dW2, db2, state, lr=alpha, beta1=adam_beta1, beta2=adam_beta2, eps=adam_eps)

        if best_weights is None:
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        
        _, _, _, A2_all, _ = forward_prop(X, W1, b1, W2, b2, keep_prob=1.0, training=False)
        train_acc = get_accuracy(A2_all, Y)
        val_acc = None
        if X_val is not None:
            _, _, _, A2_val, _ = forward_prop(X_val, W1, b1, W2, b2, keep_prob=1.0, training=False)
            val_acc = get_accuracy(A2_val, Y_val)
        
        if (epoch % 50 == 0) and not verbose:
            print(f"Epoch {epoch}: Train acc {train_acc}, Val acc {val_acc}")
        if (epoch % val_check_interval == 0) and verbose:
            print(f"Epoch {epoch}: Train acc {train_acc}, Val acc {val_acc}")
        
        # validation check for early stopping
        if early_stopping and (X_val is not None) and (epoch % val_check_interval == 0):
            # improvement?
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} checks).")
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
                  epochs=200, n_trials=20, seed=0,
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
        print(f"Random-search: sampling {n_trials}/{total} combos (iterations={epochs})")

    for idx in combos:
        # we shouldn't take updated parameters here because then as parameters improve
        # results will get biased
        alpha, hidden_size, lambd = all_combos[int(idx)]
        if verbose:
            print(f"\nTrial: alpha={alpha}, hidden_size={hidden_size}, lambda={lambd}")

        # training only on X_train and evaluating on X_val
        trained, val_acc = gradient_descent(
            X_train, Y_train,
            epochs=epochs, alpha=alpha, lambd=lambd, hidden_size=hidden_size,
            X_val=X_val, Y_val=Y_val,
            early_stopping=early_stopping, patience=patience, val_check_interval=val_check_interval,
            verbose=verbose
        )

        results.append(((alpha, hidden_size, lambd), val_acc))

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

    # # random search
    # best, _ = random_search(
    #     X_train, Y_train, X_dev, Y_dev,
    #     alphas=[1e-3, 3e-3, 1e-2],
    #     hidden_sizes=[64, 128, 256],
    #     lambdas=[0, 0.001, 0.01],
    #     epochs=500,
    #     n_trials=10,
    #     patience=10
    # )
    # best_params = best["params"]

    # retrain full
    X_full = np.concatenate([X_train, X_dev], axis=1)
    Y_full = np.concatenate([Y_train, Y_dev], axis=0)

    weights, _ = gradient_descent(
        X_full, Y_full,
        X_val=X_dev, Y_val=Y_dev,
        epochs=500,
        alpha=0.001,#best_params["alpha"],
        lambd=0.01,#best_params["lambd"],
        hidden_size=256#best_params["hidden_size"],
    )

    W1, b1, W2, b2 = weights

    np.savez("models/params.npz", W1=W1, b1=b1, W2=W2, b2=b2)#, **best_params)

    final_acc = evaluate_model(W1, b1, W2, b2, X_test, Y_test)
    print("Final model accuracy:", final_acc)
    #print("\nBest hyperparameters:", best_params)
