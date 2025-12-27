import json
import os
import pandas as pd
import numpy as np
from query_nba_api import fetch_nba_player_stats
from elo_calc import add_elo_ratings

# Number of recent games to use when computing rolling statistics
ROLLING_WINDOW_LONG = 41
ROLLING_WINDOW_SHORT = 10

def pre_processing_data(game_data_df, inactive_players_df):
    """
    Main preprocessing pipeline.
    Filters to regular season games since 2000
    Sorts rows by date
    Computes target label and rest days
    Builds per-team rolling statistics and normalizes features
    Returns a dataframe containing selected features + target.
    """
    # Keep only regular season games (case-insensitive match)
    game_data_df = game_data_df[game_data_df['season_type'].str.contains('regular season', case=False)]
    game_data_df = game_data_df[game_data_df['game_date'] > '2000-01-01']
    game_data_df = game_data_df.sort_values(by='game_date').reset_index(drop=True)
    game_data_df['target_home_team_win'] = (game_data_df['wl_home'] == 'W').astype(int)

    # Add columns for rest days for both teams
    game_data_df = calculate_rest_days(game_data_df)
    game_data_df = star_players_injured(game_data_df, inactive_players_df)
    game_data_df = add_elo_ratings(game_data_df)

    game_data_df['home_load_mgmt'] = ((game_data_df['home_stars_out'] > 0) & (game_data_df['home_rest_days'] == 1)).astype(int)
    game_data_df['away_load_mgmt'] = ((game_data_df['away_stars_out'] > 0) & (game_data_df['away_rest_days'] == 1)).astype(int)

    # initialize per-team history store and lists to collect per-row features
    team_history = {}
    features_list = []

    # Iterate rows chronologically and compute rolling stats from prior games
    for index, row in game_data_df.iterrows():
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        current_season = row['season_id']

        # Ensure each team has an entry in the history dict
        if home_team not in team_history:
            team_history[home_team] = []
        if away_team not in team_history:
            team_history[away_team] = []

        # Compute rolling stats for this season up to (but not including) this game
        home_team_win_rate_long, home_margin_long = get_rolling_season_stats(team_history[home_team], current_season, ROLLING_WINDOW_LONG)
        away_team_win_rate_long, away_margin_long = get_rolling_season_stats(team_history[away_team], current_season, ROLLING_WINDOW_LONG)

        home_team_win_rate_short, home_margin_short = get_rolling_season_stats(team_history[home_team], current_season, ROLLING_WINDOW_SHORT)
        away_team_win_rate_short, away_margin_short = get_rolling_season_stats(team_history[away_team], current_season, ROLLING_WINDOW_SHORT)

        features_list.append({
            'home_win_rate_long': home_team_win_rate_long,
            'away_win_rate_long': away_team_win_rate_long,
            'home_avg_margin_long': home_margin_long,
            'away_avg_margin_long': away_margin_long,
            'home_win_rate_short': home_team_win_rate_short,
            'away_win_rate_short': away_team_win_rate_short,
            'home_avg_margin_short': home_margin_short,
            'away_avg_margin_short': away_margin_short
        })

        margin_home = row['pts_home'] - row['pts_away']
        margin_away = row['pts_away'] - row['pts_home']

        # Determine outcome of the current game for each team
        is_home_winner = 1 if row['wl_home'] == 'W' else 0
        is_away_winner = 1 if row['wl_away'] == 'W' else 0

        # Append this game's result to each team's history for future rows
        team_history[home_team].append({
            'season_id': current_season,
            'win': is_home_winner,
            'margin': margin_home
        })

        team_history[away_team].append({
            'season_id': current_season,
            'win': is_away_winner,
            'margin': margin_away
        })

    features_df = pd.DataFrame(features_list)
    game_data_df = pd.concat([game_data_df, features_df], axis=1)

    game_data_df['diff_win_rate_long'] = game_data_df['home_win_rate_long'] - game_data_df['away_win_rate_long']
    game_data_df['diff_win_rate_short'] = game_data_df['home_win_rate_short'] - game_data_df['away_win_rate_short']

    game_data_df['diff_avg_margin_long'] = game_data_df['home_avg_margin_long'] - game_data_df['away_avg_margin_long']
    game_data_df['diff_avg_margin_short'] = game_data_df['home_avg_margin_short'] - game_data_df['away_avg_margin_short']

    game_data_df['home_trend'] = game_data_df['home_avg_margin_short'] - game_data_df['home_avg_margin_long']
    game_data_df['away_trend'] = game_data_df['away_avg_margin_short'] - game_data_df['away_avg_margin_long']

    game_data_df['diff_rest'] = game_data_df['home_rest_days'] - game_data_df['away_rest_days']
    game_data_df['diff_stars'] = game_data_df['home_stars_out'] - game_data_df['away_stars_out']

    # Select final feature columns and the target
    feature_cols = [
        'diff_elo',
        'diff_win_rate_long',
        'diff_win_rate_short',
        'diff_avg_margin_long',
        'diff_avg_margin_short',
        'home_trend',
        'away_trend',
        'diff_rest',
        'diff_stars',
        'home_load_mgmt',
        'away_load_mgmt'
    ]

    game_data_df = game_data_df[feature_cols + ['target_home_team_win']]
    return game_data_df


def get_rolling_season_stats(history, season_id, rolling_window):
    """
    Return (win_pct, avg_points) for the last ROLLING_WINDOW games
    within the same season from the provided history list. If no games are
    found for the season, return reasonable defaults (0.5 win rate, 100 pts).
    """
    # Filter history to the current season only
    season_history = [game for game in history if game['season_id'] == season_id]

    # Take the most recent ROLLING_WINDOW games
    recent_games = season_history[-rolling_window:]

    # Default values for teams with no prior games this season
    if len(recent_games) == 0:
        return 0.5, 100.0
    else:
        win_pct = np.mean([game['win'] for game in recent_games])
        avg_points = np.mean([game['margin'] for game in recent_games])
        return win_pct, avg_points


def normalize_features(df):
    """
    Normalize team average points relative to season-wide averages.
    Produces `home_avg_points_normalized` and `away_avg_pts_normalized` then
    drops the raw `home_avg_pts`/`away_avg_pts` columns.
    """
    # compute season-level mean for the two average-points columns
    season_average = df.groupby('season_id')[["home_avg_pts", "away_avg_pts"]].transform('mean')

    # Normalize by season mean to remove season-to-season scoring inflation
    df['home_avg_points_normalized'] = df['home_avg_pts'] / season_average['home_avg_pts']
    df['away_avg_points_normalized'] = df['away_avg_pts'] / season_average['away_avg_pts']

    df['diff_avg_points_normalized'] = df['home_avg_points_normalized'] - df['away_avg_points_normalized']

    # Remove raw average point columns (we keep the normalized versions)
    df = df.drop(columns=['home_avg_pts', 'away_avg_pts'])
    return df


def calculate_rest_days(df):
    """
    Compute days off before each game for home and away teams.

    Approach:
    - Convert `game_date` to datetime
    - Build a combined schedule for all teams (home + away rows)
    - Compute days difference between consecutive games per team
    - Merge the computed rest values back onto the original dataframe
    """
    # ensure datetime dtype for date computations
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Build two schedules (home and away) and unify the team id column name
    home_schedule = df[['game_date', 'team_id_home']].rename(columns={'team_id_home': 'team_id'})
    away_schedule = df[['game_date', 'team_id_away']].rename(columns={'team_id_away': 'team_id'})

    # Combine and sort so diffs are computed in chronological order per team
    full_schedule = pd.concat([home_schedule, away_schedule])
    full_schedule = full_schedule.sort_values(by=['team_id', 'game_date'])

    # days since previous game for each team (NaN -> fill with 7 days)
    full_schedule['days_since_last_game'] = full_schedule.groupby('team_id')['game_date'].diff().dt.days
    full_schedule['days_since_last_game'] = full_schedule['days_since_last_game'].fillna(7)

    # cap rest days at 7 (treat longer breaks the same as a week)
    full_schedule['days_since_last_game'] = full_schedule['days_since_last_game'].clip(upper=7)

    # Merge back to get home team rest days for each game
    df = pd.merge(df, full_schedule,
                    left_on=['team_id_home', 'game_date'],
                    right_on=['team_id', 'game_date'],
                    how='left').rename(columns={'days_since_last_game': 'home_rest_days'})
    
    # Merge again to get away team rest days
    df = pd.merge(df, full_schedule,
                    left_on=['team_id_away', 'game_date'],
                    right_on=['team_id', 'game_date'],
                    how='left').rename(columns={'days_since_last_game': 'away_rest_days'})
    
    return df


def star_players_injured(df, injured_players_df):
    """Count the number of star players out for each team in each game."""
    stats_path = '../data/raw/nba_player_stats.json'
    
    if not os.path.exists(stats_path):
        print('Fetching NBA player stats as json was not found...')
        fetch_nba_player_stats()

    print('Loading NBA player stats from json...')
    with open(stats_path, 'r') as file:
        star_players_map = json.load(file)

    def check_if_player_is_star(row):
        try:
            player_id = int(row['player_id'])
        except (ValueError, TypeError):
            return 0
        season_id = str(row['season_id'])
        return 1 if season_id in star_players_map and player_id in star_players_map[season_id] else 0
        
    game_season_map = df[['game_id', 'season_id']].drop_duplicates()

    # Inner merge: Only keep injuries that match a game in our dataset
    injured_w_season = injured_players_df.merge(game_season_map, on='game_id', how='inner')

    # Apply the star check and filter for stars only
    injured_w_season['is_star'] = injured_w_season.apply(check_if_player_is_star, axis=1)
    stars_only = injured_w_season[injured_w_season['is_star'] == 1]
    
    # Count how many stars are missing per team, per game
    stars_out_counts = stars_only.groupby(['game_id', 'team_id']).size().reset_index(name='stars_missing')

    # Merge home team star counts
    df = df.merge(stars_out_counts, 
                  left_on=['game_id', 'team_id_home'], 
                  right_on=['game_id', 'team_id'], 
                  how='left',
                  suffixes=('', '_remove_me')).rename(columns={'stars_missing': 'home_stars_out'})
    
    # Drop temporary columns from merge
    df = df.drop(columns=[c for c in df.columns if c == 'team_id' or c.endswith('_remove_me')], errors='ignore')
    
    # Merge away team star counts
    df = df.merge(stars_out_counts, 
                  left_on=['game_id', 'team_id_away'], 
                  right_on=['game_id', 'team_id'], 
                  how='left',
                  suffixes=('', '_remove_me')).rename(columns={'stars_missing': 'away_stars_out'})
    
    # Drop temporary columns from merge
    df = df.drop(columns=[c for c in df.columns if c == 'team_id' or c.endswith('_remove_me')], errors='ignore')

    # Fill NaNs with 0 (no stars out)
    df['home_stars_out'] = df['home_stars_out'].fillna(0)
    df['away_stars_out'] = df['away_stars_out'].fillna(0)

    return df
