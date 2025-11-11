from scipy.stats import ttest_ind 
import pandas as pd
df = pd.read_csv('StudentsPerformance.csv')
male_scores = df[df['gender'] == 'male']['math score'] 
print("Separate data for male students:")
print(male_scores)
female_scores = df[df['gender'] == 'female']['math score'] 
print("Separate data for female students")
print(female_scores)
t_statistic, p_value = ttest_ind(male_scores, female_scores)
print(f'T-Statistic: {t_statistic}') 
print(f'P-Value: {p_value}')
alpha = 0.05 
if p_value < alpha:
    print("There is a significant difference in math scores between male and female students.") 
else:
    print("There is no significant difference in math scores between male and female students.")

