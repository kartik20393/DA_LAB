import pandas as pd
df = pd.read_csv('train.csv')
probability_event = df['Survived'].value_counts() / len(df['Survived'])
print(probability_event)
