import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.neighbors

df_gdp = pd.read_csv('gdp.csv')
df_satisfaction = pd.read_csv('life_satisfaction.csv') 

df_gdp_clean = df_gdp.drop(df_gdp.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,65]], axis=1)
df_satisfaction_clean = df_satisfaction.drop(df_satisfaction.columns[[0,2,3,4,5,6,7,8,9,10,11,12,13,15,16]], axis=1)

df_gdp_clean = df_gdp_clean.drop_duplicates()
df_gdp_clean = df_gdp_clean.rename(columns={"2020": 'GDP per capita'})
df_satisfaction_clean = df_satisfaction_clean.drop_duplicates()
df_satisfaction_clean = df_satisfaction_clean.rename(columns={"Country": 'Country Name'})

df_country_index = df_gdp_clean.merge(df_satisfaction_clean)

# print(df_country_index.sort_values(by='GDP per capita', ascending=False))

# x = np.array(df_country_index['GDP per capita'])
# y = np.array(df_country_index['Value'])

# plt.scatter(x,y)

# plt.show()

x = df_country_index['GDP per capita'].values
x = x.reshape(len(x), 1)

y = df_country_index['Value'].values
y = y.reshape(len(y), 1)

# model = sklearn.linear_model.LinearRegression()
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
x_new = [[22587]]
print(model.predict(x_new))