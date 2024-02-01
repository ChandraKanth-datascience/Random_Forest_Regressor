# -*- coding: utf-8 -*-
"""
Created on Thu Feb 1 15:30:00 2024

@author: Chandra
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("E:\\Data sets\\Position_Salaries.csv")  # Double backslashes in the path
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Predicting a new result
# You need to reshape the input since predict expects a 2D array
y_pred = regressor.predict(np.array([[6.5]]))

# Visualising the Random Forest Regression results
X_grid = np.arange(min(X), max(X), 0.01)  # create a smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
