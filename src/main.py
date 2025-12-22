from read_csv_file import read_csv_file
from pre_processing import pre_processing_data
from train_model import train_model

def main():
    # loading data
    print("Loading data...")
    data = read_csv_file()

    # preprocess data
    print("Preprocessing data...")
    data = pre_processing_data(data)

    # train
    print("Training model...")
    model, accuracy = train_model(data)

    print("Done!")

if __name__ == "__main__":
    main()