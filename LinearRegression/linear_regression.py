import numpy as np 
import pandas as pd
## SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
import math

#read .csv into a DataFrame
dataset = pd.read_csv("house_prices.csv")
print(dataset)
size=dataset['sqft_living']
price=dataset['price']

#convert dataframes to arrays
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)
print(x)

#we use Linear Regression + fit(),  training
model = LinearRegression()
model.fit(x, y)

#MSE and R value
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value:", model.score(x,y))

#we can get the b values after the model fit
#this is the b0
print(model.coef_[0])
#this is b1 in our model
print(model.intercept_[0])

#visualize the dataset with the fitted model
plt.scatter(x, y, color= 'green')
plt.plot(x, model.predict(x), color = 'black')
plt.title ("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

#Predicting the prices
print("Prediction by the model:" , model.predict([[2000]]))
