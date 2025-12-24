import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
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

    # initialize and train the model
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)

    # model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
    # model.fit(X_train, y_train)

    # model = DecisionTreeClassifier()
    # model.fit(X_train, y_train)

    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # model = XGBClassifier()
    # model.fit(X_train, y_train)

    # evaluate the model
    predictions = model.predict(X_test)

    # accuracy score
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    # classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))


    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # "Skepticism" level
        'penalty': ['l2'],                    # L2 is usually best for sports
        'solver': ['liblinear', 'lbfgs'],     # Different math engines
        'class_weight': [None, 'balanced']    # Handle home/away win imbalance
    }

    # 2. Use TimeSeriesSplit (CRITICAL for sports)
    # This ensures we don't train on 2024 data to predict 2012 games.
    tscv = TimeSeriesSplit(n_splits=5)

    # 3. Run the Search
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000), 
        param_grid, 
        cv=tscv, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    
    print("Tuning Logistic Regression parameters...")
    grid.fit(X_train, y_train)
    
    print(f"  Best Accuracy: {grid.best_score_:.4f}")
    print(f"  Best Settings: {grid.best_params_}")

    model = grid.best_estimator_
    
    # Get the weights
    coefs = pd.DataFrame({
        'Feature': feature_cols,
        'Weight': model.coef_[0]
    }).sort_values(by='Weight', ascending=False)
    
    print("\n--- Model Logic (Coefficients) ---")
    print(coefs)

    return model, accuracy