import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

    # evaluate the model
    predictions = model.predict(X_test)

    # accuracy score
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    # classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    return model, accuracy