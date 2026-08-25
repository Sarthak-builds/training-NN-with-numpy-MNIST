import torch
from matplotlib import pyplot as plt
weights = 0.4
bias = 0.01
X = torch.arange(0,1,0.02).unsqueeze(dim=1)
y = weights * X + bias

# train-test split and visualisation
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data =  X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, predictions= None):
  plt.figure(figsize=(10,7))
  plt.scatter(train_data, train_labels, c="b", s=4, label= "Training data")
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4,labels="Predictions")
    plt.legend(prop={"size":14})

plot_predictions()

