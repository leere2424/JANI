import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve

def roc_graph(model, y_train, y_train_pred, y_test, y_test_pred):
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_pred)
    fpr_te, tpr_te, _ = roc_curve(y_test, y_test_pred)

    auc_tr = auc(fpr_tr, tpr_tr)
    auc_te = auc(fpr_te, tpr_te)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr_tr, tpr_tr, label=f"Train ROC (AUC={auc_tr:.3f})")
    plt.plot(fpr_te, tpr_te, label=f"Test ROC (AUC={auc_te:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title(f'{model} ROC Curve (Train vs Test)')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.grid(True)

    plt.show()
