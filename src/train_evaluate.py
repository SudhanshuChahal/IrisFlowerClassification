from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



def train_evaluate_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("\nclassification_report:")
    print(classification_report(y_test, y_pred))
    print("\nconfusion_matrix:")   
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "iris_model.pkl")
    print("\nModel saved successfully as 'iris_model.pkl'.")

    return model
