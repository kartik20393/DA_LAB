import pandas as pd
from scipy.stats import f_oneway
df = pd.read_csv('StudentsPerformance.csv')
ethnicity_groups = df['ethnicity'].unique() 
print("Math scores for each ethnicity_groups")
ethnicity_data = {ethnicity: df[df['ethnicity'] == ethnicity]['math score'] for ethnicity in ethnicity_groups}
print(ethnicity_data)
f_statistic, p_value_anova = f_oneway(*ethnicity_data.values())
print(f'F-Statistic: {f_statistic}') 
print(f'P-Value (ANOVA): {p_value_anova}')
alpha = 0.05
if p_value_anova < alpha:
    print("There is a significant difference in math scores among different ethnicities.") 
else:
    print("There is no significant difference in math scores among different ethnicities.") 
