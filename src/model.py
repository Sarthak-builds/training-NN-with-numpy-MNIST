import numpy as np
import pandas as pd

def init_params():
    W1 = np.random.randn(10,784) - 0.5
    b1 = np.random.randn(10,1) - 0.5
    W2 = np.random.randn(10,10) - 0.5
    b2 = np.random.randn(10,1) - 0.5
    return W1,b1,b2,W2