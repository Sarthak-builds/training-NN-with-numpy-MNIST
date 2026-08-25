import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print(train_data.shape)
print(test_data.shape)
print(train_data.head())
print(test_data.head())

x_train = train_data.drop('label', axis=1).values / 255.0
y_train = train_data['label'].values

x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
    x_train, y_train, test_size=0.1, random_state=884736743
)

print(x_train_final.shape, y_train_final.shape)
print(x_test_final.shape, y_test_final.shape)

x_train_final = x_train_final.T
x_test_final = x_test_final.T

def one_hot(Y):
     one_hot_Y = np.zeros((Y.size, Y.max()+1))
     one_hot_Y[np.arange(Y.size), Y] = 1
     one_hot_Y= one_hot_Y.T
     return one_hot_Y