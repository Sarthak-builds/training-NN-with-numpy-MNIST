import numpy as np

# activation func for hidden layers will be relu
def relu(x):
    return np.maximum(0,x)

def relu_deriv(output):
    return output >= 0

    