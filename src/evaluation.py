import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix

# 모델 예측 및 스코어 반환
def predict_and_score(model, X, y):

    # 예측
    y_pred = model.predict(X)

    # 확률값 (가능한 경우)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = None

    # 점수 계산
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    if y_proba is not None:
        roc = roc_auc_score(y, y_proba)
    else:
        roc = None

    print(f"\n{model.__class__.__name__} Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC  : {roc:.4f}")

    return {
        "model": model.__class__.__name__,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }


# train vs test ROC Graph
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


# model compare ROC Graph
def compare_roc_graph(models, y_test, results):
    for model in models:

        model_name = model.__class__.__name__
        y_test_proba = results[model_name]['y_proba']

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auc = roc_auc_score(y_test, y_test_proba)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.title('Compare ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


# 특성 중요도 시각화
def feature_importance_plot(model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    # 중요도 비율 계산
    importance_df["importance_ratio"] = (
            importance_df["importance"] / importance_df["importance"].sum()
    )

    # 상위 15개 변수 선택
    top_n = 15
    plot_df = (
        importance_df
        .sort_values("importance_ratio", ascending=False)
        .head(top_n)
        .sort_values("importance_ratio")
    )

    # XGBoost 주요 변수 중요도 시각화
    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=plot_df,
        x="importance_ratio",
        y="feature",
        color="steelblue"
    )

    plt.title("XGBoost Feature Importance (Top 15)", fontsize=14)
    plt.xlabel("Importance Ratio")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()


# 모델 예측 및 스코어 반환
def predict_score_plot(model, best_params, X, y):

    # 예측
    y_pred = model.predict(X)

    # 확률값 (가능한 경우)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = None

    # 점수 계산
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    if y_proba is not None:
        roc = roc_auc_score(y, y_proba)
    else:
        roc = None

    print(f"\n{model.__class__.__name__} Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if roc is not None:
        print(f"ROC AUC  : {roc:.4f}")

    print("\n" + "=" * 50)
    print(f"HyperOpt가 찾은 최적 파라미터:")
    for key, value in best_params.items():
        print(f"- {key}: {value}")
    print("=" * 50)

    # 혼동 행렬 시각화
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model.__class__.__name__} Final Confusion Matrix (HyperOpt)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return {
        "model": model.__class__.__name__,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }