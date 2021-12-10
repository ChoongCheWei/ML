import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("df_total.csv")

# Understand the dimensions, features and check for missing values
print(df.isna().sum())
print(df.shape)
print(list(df.columns))

X = df[['cases_recovered', 'cases_active', 'checkin', 'close_contact',
        'rtk-ag', 'pcr', 'daily', 'daily_booster', 'cumul_full', 'cumul_booster']]
y = df['cases_new']

lr = LinearRegression()
lr.fit(X, y)

print(lr.intercept_)
print(lr.coef_)
