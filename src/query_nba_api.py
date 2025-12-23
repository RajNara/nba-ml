import json
import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

def fetch_nba_player_stats(start_year=2000, end_year=2024):
    top_players_map = {}

    print("Fetching NBA player stats from", start_year, "to", end_year)

    for year in range(start_year, end_year + 1):
        season_str = f"{year}-{str(year+1)[-2:]}"
        print("Fetching stats for season:", season_str)

        try:
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season_str,
                per_mode_detailed='PerGame',
                season_type_all_star='Regular Season'
            ).get_dict()

            headers = player_stats['resultSets'][0]['headers']
            rows = player_stats['resultSets'][0]['rowSet']

            df = pd.DataFrame(rows, columns=headers)

            df = df[df['MIN'] > 28.0]
            df = df.sort_values(by='NBA_FANTASY_PTS', ascending=False)
            top_players = df.head(50)

            top_players_map[f"{season_str}"] = top_players['PLAYER_ID'].tolist()
            time.sleep(3)  # Add a delay to avoid hitting rate limits
        except Exception as e:
            print("Error fetching stats for season", season_str, ":", e)


    with open('nba_player_stats.json', 'w') as f:
        json.dump(top_players_map, f)