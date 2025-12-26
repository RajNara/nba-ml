from read_csv_file import read_csv_file
from pre_processing import pre_processing_data
from train_model import train_model

def main():
    # loading data
    print("Loading game data...")
    game_data = read_csv_file("game.csv")
    inactive_players_data = read_csv_file("inactive_players.csv")

    # preprocess data
    print("Preprocessing data...")
    data = pre_processing_data(game_data, inactive_players_data)

    # train
    print("Training model...")
    train_model(data)

    print("Done!")

if __name__ == "__main__":
    main()