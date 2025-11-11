import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score,confusion_matrix
df = pd.read_csv('data.csv')
print(df.columns)
print(df.head())
plt.scatter(df['sqft_living'], df['price']) 
plt.title('Scatter Diagram: sqft_living vs. price') 
plt.xlabel('sqft_living') 
plt.ylabel('price') 
plt.show()
correlation_coefficient = df['sqft_living'].corr(df['price']) 
print(f'Correlation Coefficient (sqft_living vs. price): {correlation_coefficient}')



