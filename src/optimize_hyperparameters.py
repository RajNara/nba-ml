import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Import your modules
from read_csv_file import read_csv_file
from pre_processing import pre_processing_data

def optimize():
    print("Loading Raw Data...")
    game_data = read_csv_file("game.csv")
    inactive_players_data = read_csv_file("inactive_players.csv")

    param_grid = {
        'k_factors': [19],        # Test slightly lower/higher stability
        'home_advantages': [85, 90, 95, 100, 105, 110], # Fine-tune the court advantage
        'windows_long': [40, 41, 42],             # Is 41 magic, or is 40 better?
        'windows_short': [8, 10, 12, 15, 18, 20]          # Check the "Form" window closely
    }

    best_score = 0
    best_params = {}

    # Total iterations = 4 * 3 * 3 * 3 = 108 loops. 
    # At ~10s per loop, this takes ~18 minutes.
    
    print("Starting Optimization Loop...")

    for k in param_grid['k_factors']:
        for home_adv in param_grid['home_advantages']:
            for w_long in param_grid['windows_long']:
                for w_short in param_grid['windows_short']:
                    
                    print(f"Testing: K={k}, Home={home_adv}, Long={w_long}, Short={w_short}...", end="")

                    # 1. Generate Data with these specific settings
                    processed_df = pre_processing_data(
                        game_data.copy(), 
                        inactive_players_data.copy(), 
                        window_long=w_long, 
                        window_short=w_short,
                        k_factor=k, 
                        home_advantage=home_adv
                    )

                    # 2. Fast Evaluation (Walk-Forward)
                    # Use last 20% of data as validation
                    split_idx = int(len(processed_df) * 0.8)
                    train_df = processed_df.iloc[:split_idx]
                    test_df = processed_df.iloc[split_idx:]
                    
                    target = 'target_home_team_win'
                    features = [c for c in processed_df.columns if c != target]

                    # Use Simple LR for speed - it correlates highly with Ensemble performance
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
                    ensemble.fit(train_df[features], train_df[target])
                    preds = ensemble.predict(test_df[features])
                    
                    acc = accuracy_score(test_df[target], preds)
                    print(f" Acc: {acc:.4f}")

                    # 3. Track Winner
                    if acc > best_score:
                        best_score = acc
                        best_params = {
                            'k': k, 'home': home_adv, 
                            'long': w_long, 'short': w_short
                        }
                        print(f"  --> NEW BEST FOUND! {best_score:.4f}")

    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Best Accuracy: {best_score:.4f}")
    print(f"Optimal Parameters: {best_params}")

if __name__ == "__main__":
    optimize()