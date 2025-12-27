import numpy as np
import pandas as pd

class EloCalculator:
    def __init__(self, k_factor=20, home_advantage=100, reversion_factor=0.75):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.reversion_factor = reversion_factor # How much rating keeps between seasons
        self.team_ratings = {}
        self.default_rating = 1500

    def get_rating(self, team):
        return self.team_ratings.get(team, self.default_rating)

    def update_ratings(self, home_team, away_team, margin, is_home_win):
        # 1. Get Current Ratings
        home_r = self.get_rating(home_team)
        away_r = self.get_rating(away_team)

        # 2. Calculate Expected Win Probability (with Home Advantage)
        # Formula: 1 / (1 + 10^((Opponent - You) / 400))
        dr = (home_r + self.home_advantage) - away_r
        expected_home = 1 / (1 + 10 ** (-dr / 400))

        # 3. Calculate Actual Result (1 for Win, 0 for Loss)
        actual_home = 1 if is_home_win else 0

        # 4. Margin of Victory Multiplier (The "Pro" Tweak)
        # Using a natural log multiplier to reward big wins but prevent huge outliers
        # Example: 1 pt win = 1.6x, 20 pt win = 4.5x
        mov_multiplier = np.log(abs(margin) + 1) * 2.2 / ((home_r - away_r) * 0.001 + 2.2)
        
        # Safety check for blowouts in reverse (rare edge case)
        if mov_multiplier < 0: mov_multiplier = 1.0 

        # 5. Calculate Point Change
        k_effective = self.k_factor * mov_multiplier
        point_delta = k_effective * (actual_home - expected_home)

        # 6. Update Teams
        self.team_ratings[home_team] = home_r + point_delta
        self.team_ratings[away_team] = away_r - point_delta

    def handle_season_reset(self):
        """
        Soft Reset: Pull everyone 25% closer to 1500 at season start.
        This prevents a team that was good 5 years ago from keeping a high rating forever.
        """
        for team in self.team_ratings:
            self.team_ratings[team] = (self.team_ratings[team] * self.reversion_factor) + (1500 * (1 - self.reversion_factor))

def add_elo_ratings(df):
    """
    Iterates through the dataframe chronologically and attaches Elo ratings.
    """
    tracker = EloCalculator(k_factor=25, home_advantage=75) # Tuned constants
    
    # Storage for the new columns
    home_elos = []
    away_elos = []

    # Identify when seasons change to trigger resets
    # Assumes DF is sorted by Date
    current_season = df['season_id'].iloc[0]

    print("Calculating Elo Ratings...")
    
    for index, row in df.iterrows():
        # Check for season change
        if row['season_id'] != current_season:
            tracker.handle_season_reset()
            current_season = row['season_id']

        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        
        # 1. RECORD PRE-GAME RATINGS (Important: Do this BEFORE updating)
        h_rating = tracker.get_rating(home_team)
        a_rating = tracker.get_rating(away_team)
        
        home_elos.append(h_rating)
        away_elos.append(a_rating)

        # 2. UPDATE RATINGS based on game result
        margin = row['pts_home'] - row['pts_away']
        is_home_win = row['wl_home'] == 'W'
        
        tracker.update_ratings(home_team, away_team, margin, is_home_win)

    # Attach columns to DF
    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    
    # Create the Diff Feature (including Home Court)
    df['diff_elo'] = (df['home_elo'] + tracker.home_advantage) - df['away_elo']
    
    return df