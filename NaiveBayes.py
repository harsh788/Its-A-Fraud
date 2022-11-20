# -*- coding: utf-8 -*-
"""LR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KMAnr7t6_82g1Fp8oGPs-2fs-rOp3EVX
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

main_df = pd.read_parquet("preprocessed_df.parquet")
X_test = pd.read_csv("../test.csv")

print(X_test.shape)

X_train = main_df.drop("isFraud", axis = 1)
Y_train = main_df["isFraud"]

# One hot encoding the categorical columns
categorical = X_test.select_dtypes(include = 'object').columns
# Filling missing values in categorical columns with mode value
X_test[categorical] = X_test[categorical].fillna(X_test[categorical].mode().iloc[0])
X_test = pd.get_dummies(X_test, columns = categorical)
X_test.shape
print(X_test.shape)

# Only selecting columns which are present in preprocessed data
X_test = pd.DataFrame(X_test, columns=X_train.columns)
print(X_test.shape)

# Filling the null values with mean
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical = X_test.select_dtypes(include = numerics).columns
X_test[numerical] = X_test[numerical].fillna(X_test[numerical].mean())

# Checking if any null value still exists
for i in X_test.columns:
  if X_test[i].isnull().sum()>0:
    print(i)

# Standardising the data
cols = X_test.select_dtypes(include=np.number).columns  
for i in cols:
    X_test[i] = (X_test[i] - X_test[i].mean())/X_test[i].std()

# Performing hyperparameter tuning
from sklearn.model_selection import RepeatedStratifiedKFold
cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=999)
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
params_NB = {'var_smoothing': np.logspace(0,-9, num=10)}
gs_NB = GridSearchCV(estimator=GaussianNB(), param_grid=params_NB, cv=cv_method, verbose=1, scoring='accuracy')
Data_transformed = PowerTransformer().fit_transform(X_train)

gs_NB.fit(Data_transformed, Y_train)

print(gs_NB.best_params_)

# Calculating the output column 
# y_pred = classifier.predict(X_test)
classifier = GaussianNB(var_smoothing=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

y_pred = pd.DataFrame(y_pred, columns=["isFraud"])
y_pred.to_csv("../csv/Y_predict_NB.csv", index=True, index_label=["Id"])