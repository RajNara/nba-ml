from read_csv_file import read_csv_file
from pre_processing import pre_processing_data

def main():
    df = read_csv_file()
    pre_processing_data(df)

if __name__ == "__main__":
    main()