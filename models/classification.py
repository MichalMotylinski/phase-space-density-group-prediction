import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def random_forest(data, labels, transform_type="6d"):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    predictions = rfc.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision = precision_score(y_test, predictions, average="macro")
    test_recall = recall_score(y_test, predictions, average="macro")
    test_f1 = f1_score(y_test, predictions, average="macro")

    return ["Random Forest", data.columns.to_list(), test_accuracy, test_precision, test_recall, test_f1]

