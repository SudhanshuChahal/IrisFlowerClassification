from src.load_data import get_data
from src.eda import explore_data, preprocess_data
from src.train_evaluate import train_evaluate_model


def main():
    df = get_data()

    explore_data(df)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    train_evaluate_model(X_train, X_test, y_train, y_test)
    


if __name__ == "__main__":
    main()