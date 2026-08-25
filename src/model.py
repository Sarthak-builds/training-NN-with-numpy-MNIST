import numpy as np
import pandas as pd
from preprocess import one_hot

def init_params():
    W1 = np.random.randn(10,784) - 0.5
    b1 = np.random.randn(10,1) - 0.5
    W2 = np.random.randn(10,10) - 0.5
    b2 = np.random.randn(10,1) - 0.5
    return W1,b1,b2,W2


def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z), axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X)+b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def deriv_ReLU(Z):
    return Z > 0    

def back_prop(Z1, A1, Z2, A2, Y, X, W2):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 /   m * dZ2.dot(A1.T)
    db2 = 1/ m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)

    dW1 = 1 /   m * dZ1.dot(X.T)
    db1 = 1/ m * np.sum(dZ1, axis=1, keepdims=True)   
    return dW1, dW2, db1, db2


def update_params(dW1, dW2, db1, db2, w1, b1, w2, b2, alpha=0.01):
    W1 = w1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = w2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, W2, b1, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, b2, W2 = init_params()
    for i in range(iterations):
         Z1,A1,Z2,A2 = forward_prop( W1, b1, W2, b2 , X)
         dW1, dW2, db1, db2 = back_prop(Z1,A1,Z2,A2,Y,X, W2)
         W1, W2, b1, b2 = update_params(dW1, dW2, db1, db2, W1, b1, W2, b2, alpha)
         if i % 50 ==0:
              print(" Epoch: ", i)
              print(" Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2          