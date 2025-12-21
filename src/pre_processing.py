import pandas as pd
import numpy as np

ROLLING_WINDOW = 10

def pre_processing_data(df):
    # Clean dataframe
    df = df[df['season_type'].str.contains('regular season', case=False)]
    df = df[df['game_date'] > '2000-01-01']
    df = df.sort_values(by='game_date').reset_index(drop=True)

    df['target_home_team_win'] = (df['wl_home'] == 'W').astype(int)

    # initialize tracking per team
    team_history = {}
    home_win_rate = []
    away_win_rate = []
    home_avg_pts = []
    away_avg_pts = []

    for index, row in df.iterrows():
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        current_season = row['season_id']

        if home_team not in team_history: team_history[home_team] = []
        if away_team not in team_history: team_history[away_team] = []

        home_team_win_rate, home_team_avg_pts = get_rolling_season_stats(team_history[home_team], current_season)
        away_team_win_rate, away_team_avg_pts = get_rolling_season_stats(team_history[away_team], current_season)

        home_win_rate.append(home_team_win_rate)
        away_win_rate.append(away_team_win_rate)
        home_avg_pts.append(home_team_avg_pts)
        away_avg_pts.append(away_team_avg_pts)

        is_home_winner = 1 if row['wl_home'] == 'W' else 0
        is_away_winner = 1 if row['wl_away'] == 'W' else 0

        team_history[home_team].append({
            'season_id': current_season,
            'win': is_home_winner,
            'points': row['pts_home'],
            'opponent': away_team
        })

        team_history[away_team].append({
            'season_id': current_season,
            'win': is_away_winner,
            'points': row['pts_away'],
            'opponent': home_team
        })

    df['home_win_rate'] = home_win_rate
    df['away_win_rate'] = away_win_rate
    df['home_avg_pts'] = home_avg_pts
    df['away_avg_pts'] = away_avg_pts

    feature_cols = ['home_win_rate', 'away_win_rate', 'home_avg_pts', 'away_avg_pts']

    df = df[feature_cols + ['target_home_team_win']]
    return df

def get_rolling_season_stats(history, season_id):
    season_history = [game for game in history if game['season_id'] == season_id]

    recent_games = season_history[-ROLLING_WINDOW:]

    if len(recent_games) == 0:
        return 0.5, 100.0
    else:
        win_pct = np.mean([game['win'] for game in recent_games])
        avg_points = np.mean([game['points'] for game in recent_games])
        return win_pct, avg_points