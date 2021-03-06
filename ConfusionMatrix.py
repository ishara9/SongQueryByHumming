from sklearn import metrics


def create_confusion_matrix(predicted, actual, values):
    confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=values)
    print(confusion_matrix)
    return confusion_matrix


def get_classification_report(predicted, actual, values):
    classification_report = metrics.classification_report(actual, predicted, labels=values)
    print(classification_report)
    return classification_report


def prototype():
    # Predicted values
    y_pred = ["a", "b", "c", "a", "b"]
    # Actual values
    y_act = ["a", "b", "c", "c", "a"]
    # Printing the confusion matrix
    # The columns will show the instances predicted for each label,
    # and the rows will show the actual number of instances for each label.
    print(metrics.confusion_matrix(y_act, y_pred, labels=["a", "b", "c"]))
    # Printing the precision and recall, among other metrics
    print(metrics.classification_report(y_act, y_pred, labels=["a", "b", "c"]))
