import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score,confusion_matrix
df = pd.read_csv('data.csv')
X_logistic = df[['sqft_living', 'waterfront']] 
y_logistic = (df['price'] > df['price'].median()).astype(int)   
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_logistic, y_logistic, 
test_size=0.2, random_state=42)
print("Assuming 'waterfront' is a binary variable indicating waterfront or not")
print(X_logistic) 
print(y_logistic)
logreg = LogisticRegression()
print("Logistic Regression")
logreg.fit(X_train_log, y_train_log)
y_pred_log = logreg.predict(X_test_log)
print("Predictions")
print(y_pred_log)
accuracy_log = accuracy_score(y_test_log, y_pred_log) 
conf_matrix_log = confusion_matrix(y_test_log, y_pred_log)
print("Model Evaluation")
print(f'Accuracy (Logistic Regression): {accuracy_log}') 
print(f'Confusion Matrix (Logistic Regression): \n{conf_matrix_log}')
