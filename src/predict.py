import numpy as np
from layers import forward_prop
from utils import get_predictions, load_test_data

def run():
    # Load model
    params = np.load("models/params.npz")
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    # Load dev data just to test
    X_test = load_test_data()
    _, _, _, A2, _ = forward_prop(X_test, W1, b1, W2, b2, training=False)
    
    # Write to a submission.csv file answers
    preds = get_predictions(A2)
    
    with open("submission.csv", "w") as f:
        f.write("ImageId,Label\n")
        for i, p in enumerate(preds, start=1):
            f.write(f"{i},{p}\n")
    
