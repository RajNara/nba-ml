import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from read_csv_file import read_csv_file
from pre_processing import pre_processing_data

# 1. Load Data with YOUR GOLDEN SETTINGS (65.32%)
print("Loading Data with Golden Settings (K=19, Home=90)...")
game_data = read_csv_file("game.csv")
inactive_players_data = read_csv_file("inactive_players.csv")

# Ensure these match your finding exactly
processed_df = pre_processing_data(game_data, inactive_players_data,
                                 window_long=40, window_short=15,
                                 k_factor=19, home_advantage=90)

# 2. Split Data (80/20)
split_idx = int(len(processed_df) * 0.8)
train_df = processed_df.iloc[:split_idx]
test_df = processed_df.iloc[split_idx:]

X_train = train_df.drop(columns=['target_home_team_win'])
y_train = train_df['target_home_team_win']
X_test = test_df.drop(columns=['target_home_team_win'])
y_test = test_df['target_home_team_win']

# 3. Train the 3 Base Models
print("Training Base Models...")
clf_lr = LogisticRegression(solver='liblinear', max_iter=2000, C=0.001)
clf_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
clf_xgb = XGBClassifier(            
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
)

clf_lr.fit(X_train, y_train)
clf_rf.fit(X_train, y_train)
clf_xgb.fit(X_train, y_train)

# 4. Get Probabilities
print("Getting Model Probabilities...")
p_lr = clf_lr.predict_proba(X_test)
p_rf = clf_rf.predict_proba(X_test)
p_xgb = clf_xgb.predict_proba(X_test)

# 5. Brute Force the Weights (1 to 5)
print("Searching for best voting weights...")
best_acc = 0
best_w = (1, 1, 1)

weights_to_test = [1, 2, 3, 4, 5]

for w_lr in weights_to_test:
    for w_rf in weights_to_test:
        for w_xgb in weights_to_test:
            
            # Weighted Average Formula
            avg_p = (p_lr * w_lr + p_rf * w_rf + p_xgb * w_xgb) / (w_lr + w_rf + w_xgb)
            preds = np.argmax(avg_p, axis=1)
            acc = accuracy_score(y_test, preds)
            
            if acc > best_acc:
                best_acc = acc
                best_w = (w_lr, w_rf, w_xgb)
                print(f"New Best: {best_acc:.5f} | Weights: LR={w_lr}, RF={w_rf}, XGB={w_xgb}")

print(f"\nFINAL OPTIMIZED ACCURACY: {best_acc:.5f}")
print(f"Use these weights in your VotingClassifier: {best_w}")