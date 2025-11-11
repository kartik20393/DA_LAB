import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score,confusion_matrix
df = pd.read_csv('data.csv')
X_simple = sm.add_constant(df[['sqft_living']]) 
y_simple = df['price'] 
model_simple = sm.OLS(y_simple, X_simple).fit()
print("Summary of the simple linear regression")
print(model_simple.summary())
X_multi = sm.add_constant(df[['sqft_living', 'bedrooms', 'bathrooms']]) 
y_multi = df['price'] 
model_multi = sm.OLS(y_multi, X_multi).fit()
print("Summary of the Multiple Linear Regression")
print(model_multi.summary())
predictions_simple = model_simple.predict(X_simple)
print("Simple Prediction")
print(predictions_simple)
X_multi = sm.add_constant(df[['sqft_living', 'bedrooms', 'bathrooms']]) 
y_multi = df['price'] 
model_multi = sm.OLS(y_multi, X_multi).fit() 
predictions_multi = model_multi.predict(X_multi)  
print("Multiple Predictions")
print(predictions_multi)

