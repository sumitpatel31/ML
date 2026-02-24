import predictions
from predictions import predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def metric():
    try:
        test, pred = predict()

        if test is None or pred is None:
            print("Prediction failed. Metrics cannot be calculated.")
            return None

        acc = round(accuracy_score(test, pred), 2)
        pr  = round(precision_score(test, pred), 2)
        rc  = round(recall_score(test, pred), 2)
        f1  = round(f1_score(test, pred), 2)

        matrix = [acc, pr, rc, f1]
        print(matrix)

    except Exception as e:
        print("Metric error:", e)
        return None

metric()