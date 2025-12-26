import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_model(df):
    # split by time
    split_index = int(len(df) * 0.8)

    train_df = df[:split_index]
    test_df = df[split_index:]

    target_cols = 'target_home_team_win'
    feature_cols = [col for col in df.columns if col != target_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]

    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]

    print(f"Training on {len(X_train)} games. Testing on {len(X_test)} games.")

    # initialize and train the ensemble
    ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=2000, C=1.0)),
        ('rf', RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            random_state=42
        )),
        ('xgb', XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        ))
    ],
    weights=[2, 1, 1],
    voting='soft')

    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)
    final_accuracy = accuracy_score(y_test, predictions)
    print("Ensemble Accuracy:", final_accuracy)

    # model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
    # model.fit(X_train, y_train)

    # model = DecisionTreeClassifier()
    # model.fit(X_train, y_train)

    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # model = XGBClassifier()
    # model.fit(X_train, y_train)

    # evaluate the model
    # predictions = model.predict(X_test)

    # # accuracy score
    # accuracy = accuracy_score(y_test, predictions)

    # print(f"Model Accuracy: {accuracy:.4f}")

    # classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    return ensemble, final_accuracy