import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
df = pd.read_csv('train.csv')
titanic_data = df.dropna(subset=['Age'])
plt.hist(df['Age'], bins=30, density=True, alpha=0.5, color='b',label='Age Distribution')
mu, std = norm.fit(titanic_data['Age'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()
