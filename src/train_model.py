import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

    # evaluate the model
    predictions = model.predict(X_test)

    # accuracy score
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    # classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # 1. Extract and Sort Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances) # Sort ascending for horizontal bar chart

    # 2. Create the Plot
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    
    # 3. Add Labels
    # We map the sorted indices back to the feature names
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    # 4. Save and Show
    plt.tight_layout()
    plt.savefig('feature_importance.png') # Saves the image to your folder
    plt.show()

    # 5. Print the numeric list (Optional)
    print("\n--- Feature Importance ---")
    for i in reversed(indices):
        print(f"{feature_cols[i]}: {importances[i]:.4f}")

    return model, accuracy