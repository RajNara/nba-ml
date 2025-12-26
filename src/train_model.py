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

    # initialize and train the model
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)

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

    # plot_calibration_curve(ensemble, X_test, y_test)
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


    # param_grid = {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # "Skepticism" level
    #     'penalty': ['l2'],                    # L2 is usually best for sports
    #     'solver': ['liblinear', 'lbfgs'],     # Different math engines
    #     'class_weight': [None, 'balanced']    # Handle home/away win imbalance
    # }

    # # 2. Use TimeSeriesSplit (CRITICAL for sports)
    # # This ensures we don't train on 2024 data to predict 2012 games.
    # tscv = TimeSeriesSplit(n_splits=5)

    # # 3. Run the Search
    # grid = GridSearchCV(
    #     LogisticRegression(max_iter=2000), 
    #     param_grid, 
    #     cv=tscv, 
    #     scoring='accuracy', 
    #     n_jobs=-1,
    #     verbose=1
    # )
    
    # print("Tuning Logistic Regression parameters...")
    # grid.fit(X_train, y_train)
    
    # print(f"  Best Accuracy: {grid.best_score_:.4f}")
    # print(f"  Best Settings: {grid.best_params_}")

    # model = grid.best_estimator_
    
    # # Get the weights
    # coefs = pd.DataFrame({
    #     'Feature': feature_cols,
    #     'Weight': model.coef_[0]
    # }).sort_values(by='Weight', ascending=False)
    
    # print("\n--- Model Logic (Coefficients) ---")
    # print(coefs)

    return model, final_accuracy


def plot_calibration_curve(model, X_test, y_test):
    # 1. Get Probabilities for the Positive Class (Home Win)
    # Note: VotingClassifier needs voting='soft' to support predict_proba
    probs = model.predict_proba(X_test)[:, 1]
    
    # 2. Calculate the Curve
    # n_bins=10: Split predictions into 10 buckets (0-10%, 10-20%... 90-100%)
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Your Ensemble')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability (Confidence)')
    plt.ylabel('Fraction of Positives (Actual Win Rate)')
    plt.title('Reliability Diagram: Can we trust the confidence?')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Print the Bucket Data
    print("\n--- Calibration Data ---")
    for p_pred, p_true in zip(prob_pred, prob_true):
        print(f"Predicted: {p_pred:.2f} | Actual: {p_true:.2f} | Diff: {p_true - p_pred:.2f}")
