import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


def random_forest(x_train, y_train):
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=40, n_jobs=-1)
    rfc.fit(x_train, y_train)

    return rfc, "Random Forest"


def data_split(data, labels, test_size):
    return train_test_split(data, labels, test_size=test_size, stratify=labels)


def get_scores(model, features, x_test, y_test):
    predictions = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision = precision_score(y_test, predictions, average="macro")
    test_recall = recall_score(y_test, predictions, average="macro")
    test_f1 = f1_score(y_test, predictions, average="macro")

    print(classification_report(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    return ["Random Forest", features, test_accuracy, test_precision, test_recall, test_f1]


def perm_importance(estimator, x_train, x_test, y_test, fig_name="", n_repeats=10, random_state=0):
    perms = permutation_importance(estimator, x_test, y_test, n_repeats=n_repeats, random_state=random_state)

    sorted_idx = perms.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(x_train.columns[sorted_idx], perms.importances[sorted_idx].mean(axis=1).T)
    ax.set_title("Permutation Importances")
    plt.show()

    arr = []
    """for i in perms.importances_mean.argsort()[::-1]:
        arr.append(cols[i])"""
    return arr
