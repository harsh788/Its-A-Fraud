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


import tensorflow as tf

input_shape = [X_train.shape[1]]

model = tf.keras.Sequential([
 
    tf.keras.layers.Dense(units=64, activation='relu',
                          input_shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.summary()

model.compile(optimizer='adam', 
               
# MAE error is good for
# numerical predictions
loss='mae') 

losses = model.fit(X_train, Y_train,
 
#    validation_data=(X_val, y_val),

# it will use 'batch_size' number
# of examples per example
batch_size=256,
epochs=15,  # total epoch
)

pred = model.predict(X_test)
pred = pd.DataFrame(pred)
pred = pred.round(0)
pred[0] = pred[0].astype(int)

pred.to_csv("./NN.csv", index=True, index_label=["Id"])