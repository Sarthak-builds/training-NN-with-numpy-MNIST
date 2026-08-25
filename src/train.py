import numpy as np
from preprocess import x_train_final, x_test_final, y_train_final, y_test_final
from model import gradient_descent, get_predictions, get_accuracy, forward_prop

W1, b1, W2, b2 = gradient_descent(x_train_final, y_train_final, 500, 0.1)

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

test_predictions = predict(x_test_final, W1, b1, W2, b2)
test_accuracy = get_accuracy(test_predictions, y_test_final)
print(f"\nTest Set Accuracy: {test_accuracy:.4f}")