from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def Confusion_Matrix(y_true, y_pred, model):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion matrix ({model})')
    plt.show()

    TP = cm[1, 1]  # True Positive: reality is 1, predicted is 1.
    TN = cm[0, 0]  # True Negative: reality is 0, predicted is 0.
    FP = cm[0, 1]  # False Positive: reality is 0, predicted is 1.
    FN = cm[1, 0]  # False Negative: reality is 1, predicted is 0.

    # Calculating metrics
    accuracy_dt = (TP + TN) / (TP + TN + FP + FN)
    recall_dt = TP / (TP + FN)
    precision_dt = TP / (TP + FP)

    print(f"Accuracy ({model}): {accuracy_dt}")
    print(f"Recall ({model}): {recall_dt}")
    print(f"Precision ({model}): {precision_dt}")
