import logging
from random import random
import pandas as pd

import mlflow
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

logging.basicConfig(
    filename='../logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")

try:
    logging.info("Loading data")
    df = pd.read_csv("../data/raw/fraudTrain.csv")
    df = df.drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    logging.error("CSV not found")
    df = pd.DataFrame()
    
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

with mlflow.start_run():
    
    mlflow.set_tag("experiment_type", "inital experiments")
    
    params = {
        "n_estimators": 10,
        "max_depth": 5
    }
    
    mlflow.log_params(params)
    
    model = RandomForestClassifier(
        **params,
        random_state=1
        )
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    acc = accuracy_score(y_valid, predictions, normalize=True)
    f1_sc = f1_score(y_valid, predictions)
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1_sc
    }
    
    mlflow.log_metrics(metrics)
