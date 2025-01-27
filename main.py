from src.load_data import get_data
from src.eda import explore_data, preprocess_data


def main():
    df = get_data()

    explore_data(df)

    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessing complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    


if __name__ == "__main__":
    main()