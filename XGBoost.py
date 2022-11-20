import pandas as pd
import numpy as np
main_df = pd.read_parquet("preprocessed_df.parquet")

X_train = main_df.drop("isFraud", axis = 1)
Y_train = main_df["isFraud"]

X_test = pd.read_csv("../test.csv")
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
# Training the model
# import xgboost as xgb
# from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

param_grid = {
    'n_estimators':[3000], 
    'max_depth':[14], 
    'learning_rate':[0.2], 
    'subsample':[0.8],
    'colsample_bytree':[0.4],  
    'eval_metric':['auc'],
    'tree_method':['hist'],
    'missing':[-1],
    }

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgbModel = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, verbose=1, cv=5, n_jobs=-1)
xgbModel.fit(X_train, Y_train)

print(xgbModel.best_params_)
print(xgbModel.best_score_)
Y_test = xgbModel.predict(X_test);
Y_test = pd.DataFrame(Y_test)
Y_test.to_csv("./XGBPred.csv");
# xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
# xgb_model.fit(X_train, Y_train)

# y_pred = xgb_model.predict(X_test)

# print(confusion_matrix(y, y_pred))