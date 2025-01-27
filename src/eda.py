import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def explore_data(df):
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.isnull().sum())

def preprocess_data(df):
    X = df.iloc[:, :-1]
    y = df['variety']
    y = y.map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
