import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('Datasets/eng.csv')
# print(df)

X = np.array(list(range(1, 10)))

# Change the shape of X array from 1D to 2D
X = X.reshape((9, 1))

Y = df.loc[:, ['0']]
Y = np.array(Y)

x_train = X[0:10:2]
print('X_train:')
print(x_train)
print()
x_test = X[1:10:2]
print('X_test:')
print(x_test)
print()
y_train = Y[0:10:2]
print('Y_train:')
print(y_train)
print()
y_test = Y[1:10:2]
print('Y_test:')
print(y_test)

reg = RandomForestRegressor(
    n_estimators=60, max_depth=30, n_jobs=-1, warm_start=True)
reg.fit(x_train, y_train)

Y_pred = reg.predict(np.array([[10]]))
print('Y_pred:')
print(Y_pred)