import pandas as pd

def get_data():
    df = pd.read_csv("data/iris.csv")
    return df