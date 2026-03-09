from catboost import Pool, CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# CatBoost 모델 정의 및 학습 함수
def train_cat(X_tr, y_tr, X_val, y_val):

    # 0. 범주형 컬럼 이용해서 Pool 객체 만듦.
    cat_features = ['location', 'subscription_type', 'payment_plan', 'payment_method', 'customer_service_inquiries']
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    # 1. 모델 정의
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )

    # 2. 모델 학습
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=False,
    )

    return model


# XGBoost 모델 정의 및 학습 함수
def train_xgb(X_tr, y_tr, X_val, y_val):

    # 1. 모델 정의
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )

    # 2. 모델 학습
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False
    )

    return model


# LightGBM 모델 정의 및 학습 함수
def train_lgbm(X_tr, y_tr, X_val, y_val):

    # 1. 모델 정의
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_round=50,
        random_state=42,
        verbose=-1,
    )

    # 2. 모델 학습
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=10, verbose=False)],
    )

    return model


# RandomForest 모델 정의 및 학습 함수
def train_rf(X_tr, y_tr):

    # 1. 모델 정의
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=6,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        oob_score=True
    )

    # 2. 모델 학습
    model.fit(
        X_tr, y_tr,
    )

    return model


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

# 모델 비교 함수
def compare_models(X_tr, X_tr_prep, X_val, X_val_prep, y_tr, y_val):
    models = []

    models.append(train_cat(X_tr, y_tr, X_val, y_val))
    models.append(train_xgb(X_tr_prep, y_tr, X_val_prep, y_val))
    models.append(train_lgbm(X_tr_prep, y_tr, X_val_prep, y_val))
    models.append(train_rf(X_tr_prep, y_tr))

    return models


