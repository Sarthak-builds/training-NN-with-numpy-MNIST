# LOG

## preprocess.py
- `train_data, test_data` — loads MNIST CSVs from the local data/ folder
- `x_train, y_train` — extracts pixel features (normalized 0–1) and labels from train_data
- `train_test_split` — splits data into 90% train / 10% validation sets
- `x_train_final, x_test_final` — transposed to shape (784, samples) for matrix ops
- `one_hot(Y)` — converts integer labels into one-hot encoded matrix of shape (10, samples)

## model.py
- `init_params()` — randomly initializes weights W1, W2 and biases b1, b2
- `ReLU(Z)` — activation function; clips negatives to zero
- `softmax(Z)` — converts output layer scores to probabilities summing to 1
- `forward_prop(W1, b1, W2, b2, X)` — runs input through the 2-layer network, returns Z1, A1, Z2, A2
- `deriv_ReLU(Z)` — gradient of ReLU; returns 1 where Z > 0, else 0
- `back_prop(Z1, A1, Z2, A2, Y, X, W2)` — computes gradients dW1, dW2, db1, db2 via backpropagation
- `update_params(...)` — applies gradient descent step to all weights and biases
- `get_predictions(A2)` — returns index of max probability (predicted digit) for each sample
- `get_accuracy(predictions, Y)` — computes fraction of correct predictions
- `gradient_descent(X, Y, iterations, alpha)` — full training loop; prints epoch and accuracy every 50 steps

## train.py
- imports preprocessed data splits and model functions
- runs `gradient_descent` for 500 iterations with learning rate 0.1
- `predict(X, W1, b1, W2, b2)` — runs forward pass and returns predicted labels for a dataset
- evaluates and prints final accuracy on the held-out test set
