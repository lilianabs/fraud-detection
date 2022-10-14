import logging
from random import random
import pandas as pd

import mlflow
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")

df = pd.read_csv("../data/raw/fraudTrain.csv")
df = df.drop('Unnamed: 0', axis=1)

# Feature engineering
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df.trans_date_trans_time.dt.hour

# Normal hours are between 05:00 and 21:00 and abnormal otherwise
df["is_normal_hour"] = 0
df.loc[df.hour < 5, "is_normal_hour"] = 1
df.loc[df.hour > 21, "is_normal_hour"] = 1

features = ["amt", 'is_normal_hour']

X = df[features]
y = df.is_fraud

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      stratify=y,
                                                      random_state=1,
                                                      train_size=0.8)

model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)

predictions = model.predict(X_valid)

print(accuracy_score(y_valid, predictions, normalize=True))
print(f1_score(y_valid, predictions))
