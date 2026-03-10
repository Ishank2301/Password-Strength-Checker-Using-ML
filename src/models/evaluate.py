from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
