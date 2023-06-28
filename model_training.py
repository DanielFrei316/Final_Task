import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict, KFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression
from bs4 import BeautifulSoup
import re
import numpy as np
from IPython.display import display
import openpyxl
from datetime import datetime
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from madlan_data_prep import prepare_data

df = pd.read_excel('output_all_students_Train_v10.xlsx')
clean_df = prepare_data(df)


predictive_df=clean_df.copy()

#columns_to_drop = ['hasElevator', 'hasBars', 'hasAirCondition', 'handicapFriendly']
#predictive_df = predictive_df.drop(columns=columns_to_drop)

X = predictive_df.drop('price', axis=1)
y = predictive_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Categorical columns for one-hot encoding
categorical_columns = ["City", "type","city_area", "condition", "furniture", "entranceDate"]

# Numerical columns for standardization
numerical_columns = ['Area','room_number','floor']

# Binary columns
binary_columns = ['hasParking',
                  'hasBalcony', 'hasMamad', 'hasStorage']

# Preprocessing steps for categorical, numerical, and binary columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns),
        ('bin', 'passthrough', binary_columns)
    ])

# Create the pipeline with preprocessing and the elastic net model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('elasticnet', ElasticNet(l1_ratio=0.9, alpha=0.1))
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
mse_scores = -cv_scores
rmse_scores = np.sqrt(mse_scores)  # Calculate RMSE
mae_scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
r2_scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='r2')

n_train = len(X_train)
k_train = X_train.shape[1]  # Number of predictors (columns)
adj_r2_scores = 1 - ((1 - r2_scores) * (n_train - 1)) / (n_train - k_train - 1)

# Print the mean performance metrics
print("Mean Squared Error (MSE):", mse_scores.mean())
print("Root Mean Squared Error (RMSE):", rmse_scores.mean())  # Add this line
print("Mean Absolute Error (MAE):", mae_scores.mean())
print("R-squared (R^2):", r2_scores.mean())
print("Adjusted R-squared:", adj_r2_scores.mean())

pipeline.fit(X_train, y_train)

import joblib
joblib.dump(pipeline, 'trained_model.pkl')
