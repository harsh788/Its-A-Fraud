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

# Training the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=200)
classifier.fit(X_train, Y_train)

# One hot encoding the categorical columns
categorical = X_test.select_dtypes(include = 'object').columns
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

# Calculating the output column 
y_pred = classifier.predict(X_test)

y_pred = pd.DataFrame(y_pred, columns=["isFraud"])
y_pred.to_csv("../csv/Y_predict_LR.csv", index=True, index_label=["Id"])