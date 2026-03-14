import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import f1_score


# XGBoost 최적 하이퍼 파라미터 찾기
def find_best_xgb_params(X_tr_prep, X_val_prep, y_tr, y_val,
                         n_estimators_list, learning_rate_list, max_depth_list, subsample_list, colsample_bytree_list, max_evals=50):

    # XGBoost 튜닝용 목적 함수 정의
    def objective_xgb(params):
        model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1,
            **params
        )

        model.fit(X_tr_prep, y_tr)
        y_pred = model.predict(X_val_prep)
        score = f1_score(y_val, y_pred, average='macro')

        return {'loss': -score, 'status': STATUS_OK}

    # HyperOpt용 탐색 공간 설정
    search_space = {
        'n_estimators': hp.choice('n_estimators', n_estimators_list),
        'learning_rate': hp.choice('learning_rate', learning_rate_list),
        'max_depth': hp.choice('max_depth', max_depth_list),
        'subsample': hp.choice('subsample', subsample_list),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytree_list)
    }

    trials = Trials()
    best_idx = fmin(
        fn=objective_xgb,
        space=search_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
        rstate=np.random.default_rng(seed=42)
    )

    best_params = {
        'n_estimators': n_estimators_list[best_idx['n_estimators']],
        'learning_rate': learning_rate_list[best_idx['learning_rate']],
        'max_depth': max_depth_list[best_idx['max_depth']],
        'subsample': subsample_list[best_idx['subsample']],
        'colsample_bytree': colsample_bytree_list[best_idx['colsample_bytree']]
    }

    return best_params


# LightGBM 최적 하이퍼 파라미터 탐색
def find_best_lgbm_params(X_tr_prep, X_val_prep, y_tr, y_val,
                          n_estimators_list, learning_rate_list, max_depth_list, num_leaves_list, subsample_list, colsample_bytree_list, max_evals=50):
    # LightGBM 튜닝용 목적 함수 정의
    def objective_lgbm(params):
        model = LGBMClassifier(
            random_state=42,
            eval_metric='logloss',
            verbose=-1,
            n_jobs=-1,
            **params
        )

        model.fit(X_tr_prep, y_tr)
        y_pred = model.predict(X_val_prep)
        score = f1_score(y_val, y_pred, average='macro')

        return {'loss': -score, 'status': STATUS_OK}

    # HyperOpt용 LightGBM 탐색 공간 설정
    search_space_lgbm = {
        'n_estimators': hp.choice('lgbm_n_estimators', n_estimators_list),
        'learning_rate': hp.choice('lgbm_learning_rate', learning_rate_list),
        'max_depth': hp.choice('lgbm_max_depth', max_depth_list),
        'num_leaves': hp.choice('lgbm_num_leaves', num_leaves_list),
        'subsample': hp.choice('lgbm_subsample', subsample_list),
        'colsample_bytree': hp.choice('lgbm_colsample_bytree', colsample_bytree_list)
    }

    # HyperOpt로 LightGBM 최적 파라미터 탐색
    trials = Trials()

    best_idx = fmin(
        fn=objective_lgbm,
        space=search_space_lgbm,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
        rstate=np.random.default_rng(seed=42)
    )

    # 탐색 결과를 실제 LightGBM 파라미터 값으로 변환
    best_params = {
    'n_estimators': n_estimators_list[best_idx['lgbm_n_estimators']],
    'learning_rate': learning_rate_list[best_idx['lgbm_learning_rate']],
    'max_depth': max_depth_list[best_idx['lgbm_max_depth']],
    'num_leaves': num_leaves_list[best_idx['lgbm_num_leaves']],
    'subsample': subsample_list[best_idx['lgbm_subsample']],
    'colsample_bytree': colsample_bytree_list[best_idx['lgbm_colsample_bytree']]
    }

    return best_params


# CatBoost 최적 하이퍼 파라미터 찾기
def find_best_cat_params(X_tr_prep, X_val_prep, y_tr, y_val, iterations_list, learning_rate_list, depth_list, l2_leaf_reg_list, max_evals=50):

    # CatBoost 튜닝용 목적 함수 정의
    def objective_cat(params):
        model = CatBoostClassifier(
            random_state=42,
            verbose=0,
            **params
        )

        model.fit(X_tr_prep, y_tr)
        y_pred = model.predict(X_val_prep)
        score = f1_score(y_val, y_pred, average='macro')

        return {'loss': -score, 'status': STATUS_OK}

    # HyperOpt용 CatBoost 탐색 공간 설정
    search_space_cat = {
        'iterations': hp.choice('cat_iterations', iterations_list),
        'learning_rate': hp.choice('cat_learning_rate', learning_rate_list),
        'depth': hp.choice('cat_depth', depth_list),
        'l2_leaf_reg': hp.choice('cat_l2_leaf_reg', l2_leaf_reg_list)
    }

    # HyperOpt로 CatBoost 최적 파라미터 탐색
    trials_cat = Trials()
    best_idx_cat = fmin(
        fn=objective_cat,
        space=search_space_cat,
        algo=tpe.suggest,
        trials=trials_cat,
        max_evals=max_evals,
        rstate=np.random.default_rng(seed=42)
    )

    # 탐색 결과를 실제 CatBoost 파라미터 값으로 변환
    best_params = {
        'iterations': iterations_list[best_idx_cat['cat_iterations']],
        'learning_rate': learning_rate_list[best_idx_cat['cat_learning_rate']],
        'depth': depth_list[best_idx_cat['cat_depth']],
        'l2_leaf_reg': l2_leaf_reg_list[best_idx_cat['cat_l2_leaf_reg']]
    }

    return best_params


