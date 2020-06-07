from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

print("Loading CSV Data")
selected_data = pd.read_csv("data/selected_data_10.csv", index_col=0)
sef_data = pd.read_csv("data/SEF_filled_mean.csv", dtype={"cty": str, "name": str, "_merge": str, "_merge1": str, "_merge2": str}, index_col=0)

x = selected_data[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
y = sef_data['v131']

print("Linear Regression")
lm = linear_model.LinearRegression()
lm.fit(x, y)
y_pred = lm.predict(x)

# data_points = plt.scatter(x=x, y=y, color='blue', label='Data')
regression = plt.plot(x, y_pred, color='black', label='Regression Model')
plt.legend()
plt.show()