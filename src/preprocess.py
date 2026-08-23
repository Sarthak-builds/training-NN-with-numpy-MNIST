import numpy as np
import pandas as pd

train_data = pd.read_csv('/kaggle/input/competitions/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/competitions/digit-recognizer/test.csv')

print(train_data.shape)
print(test_data.shape)
print(train_data.head())
print(test_data.head())

from sklearn.model_selection import train_test_split

x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
    x_train, y_train, test_size=0.1, random_state=884736743
)

print(x_train_final.shape, y_train_final.shape)
print(x_test_final.shape, y_test_final.shape)

