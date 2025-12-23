import pandas as pd
import numpy as np

# Number of recent games to use when computing rolling statistics
ROLLING_WINDOW = 10

def pre_processing_data(df):
    """
    Main preprocessing pipeline.
    Filters to regular season games since 2000
    Sorts rows by date
    Computes target label and rest days
    Builds per-team rolling statistics and normalizes features
    Returns a dataframe containing selected features + target.
    """
    # Keep only regular season games (case-insensitive match)
    df = df[df['season_type'].str.contains('regular season', case=False)]

    # Remove games before 2000
    df = df[df['game_date'] > '2000-01-01']

    # Ensure chronological order for rolling computations
    df = df.sort_values(by='game_date').reset_index(drop=True)

    # Binary target: 1 if home team won, 0 otherwise
    df['target_home_team_win'] = (df['wl_home'] == 'W').astype(int)

    # Add columns for rest days for both teams
    df = calculate_rest_days(df)

    # initialize per-team history store and lists to collect per-row features
    team_history = {}
    home_win_rate = []
    away_win_rate = []
    home_avg_pts = []
    away_avg_pts = []

    # Iterate rows chronologically and compute rolling stats from prior games
    for index, row in df.iterrows():
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        current_season = row['season_id']

        # Ensure each team has an entry in the history dict
        if home_team not in team_history: team_history[home_team] = []
        if away_team not in team_history: team_history[away_team] = []

        # Compute rolling stats for this season up to (but not including) this game
        home_team_win_rate, home_team_avg_pts = get_rolling_season_stats(team_history[home_team], current_season)
        away_team_win_rate, away_team_avg_pts = get_rolling_season_stats(team_history[away_team], current_season)

        # Collect computed features for adding to dataframe later
        home_win_rate.append(home_team_win_rate)
        away_win_rate.append(away_team_win_rate)
        home_avg_pts.append(home_team_avg_pts)
        away_avg_pts.append(away_team_avg_pts)

        # Determine outcome of the current game for each team
        is_home_winner = 1 if row['wl_home'] == 'W' else 0
        is_away_winner = 1 if row['wl_away'] == 'W' else 0

        # Append this game's result to each team's history for future rows
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

    # Attach the computed rolling features back to the dataframe
    df['home_win_rate'] = home_win_rate
    df['away_win_rate'] = away_win_rate
    df['home_avg_pts'] = home_avg_pts
    df['away_avg_pts'] = away_avg_pts

    # Normalize average points relative to season averages and drop raw avg columns
    df = normalize_features(df)

    # Select final feature columns and the target
    feature_cols = [
        'home_win_rate',
        'away_win_rate',
        'home_avg_points_normalized',
        'away_avg_pts_normalized',
        'home_rest_days',
        'away_rest_days'
    ]

    df = df[feature_cols + ['target_home_team_win']]
    return df


def get_rolling_season_stats(history, season_id):
    """
    Return (win_pct, avg_points) for the last ROLLING_WINDOW games
    within the same season from the provided history list. If no games are
    found for the season, return reasonable defaults (0.5 win rate, 100 pts).
    """
    # Filter history to the current season only
    season_history = [game for game in history if game['season_id'] == season_id]

    # Take the most recent ROLLING_WINDOW games
    recent_games = season_history[-ROLLING_WINDOW:]

    # Default values for teams with no prior games this season
    if len(recent_games) == 0:
        return 0.5, 100.0
    else:
        win_pct = np.mean([game['win'] for game in recent_games])
        avg_points = np.mean([game['points'] for game in recent_games])
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
    df['away_avg_pts_normalized'] = df['away_avg_pts'] / season_average['away_avg_pts']

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