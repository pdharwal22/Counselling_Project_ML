import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score

from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('Datasets/eng.csv')
print(df)

X = np.array(list(range(1, 10)))
# print('X:')
# print(X)

# Change the shape of X array from 1D to 2D
X = X.reshape((9, 1))
# print('New X:')
# print(X)
# print(X.shape)

Y = df.loc[:, ['0']]
Y = np.array(Y)
# Y = Y.flatten()
print('Y:')
print(Y)
# print(Y.shape)

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

training_accuracy = reg.score(x_train, y_train)
test_accuracy = reg.score(x_test, y_test)
rmse_train = np.sqrt(mean_squared_error(reg.predict(x_train), y_train))
rmse_test = np.sqrt(mean_squared_error(reg.predict(x_test), y_test))
print("Training Accuracy = %0.3f, Test Accuracy = %0.3f, RMSE (train) = %0.3f, RMSE (test) = %0.3f" %
      (training_accuracy, test_accuracy, rmse_train, rmse_test))

y_true = y_test
y_pred = reg.predict(x_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
plt.title('Random Forest')
plt.xlabel('Actual')
plt.ylabel('Predicted')
# plt.savefig('Random_Forest.png')

print("exp variance err", explained_variance_score(y_true, y_pred))
print("max error", max_error(y_true, y_pred))
print("mae", mean_absolute_error(y_true, y_pred))
print("mse", mean_squared_error(y_true, y_pred))
print("mean sq log error", mean_squared_log_error(y_true, y_pred))
print("median absolute error", median_absolute_error(y_true, y_pred))
print("r2", r2_score(y_true, y_pred))   # 0.07385542168674553

Y_pred = reg.predict(np.array([[10]]))
print('Y_pred:')
print(Y_pred)   # 92.15

# filename = 'random_forest.sav'
# joblib.dump(reg, filename)