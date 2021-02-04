import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import make_scorer, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from sklearn.pipeline import make_union
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate

from category_encoders import OrdinalEncoder, OneHotEncoder, CountEncoder

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from pathlib import Path
import config

# pd.set_option('display.max_rows', 120)
# plt.style.use("dark_background")
# plt.rcParams['figure.figsize'] = (20, 15)

# os.chdir(Path.cwd()/'home_works'/'1. Auto_ML'/'drivendata_flu_shot')

print(Path.cwd())

X = pd.read_csv("./input/training_set_features.csv", index_col='respondent_id')
y = pd.read_csv("./input/training_set_labels.csv", index_col='respondent_id')
submission_format = pd.read_csv("./input/submission_format.csv", index_col='respondent_id')
test_set = pd.read_csv("./input/test_set_features.csv", index_col='respondent_id')


#  Encoders

ordinal_label_encoder_pipe = Pipeline([

    ('label-encoder', OrdinalEncoder(cols=config.categorical)),
    # ('quasy_constant_remover', VarianceThreshold(.99 * (1 - .99))),
    # ('imputer', SimpleImputer(strategy='most_frequent')),
    # ('scaler', StandardScaler()),
    # ('regressor', BayesianRidge())
])


# split for  catboost datasets
X.replace(to_replace=config.ordinal_to_replace, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# %%

X_hyper_encoder = ordinal_label_encoder_pipe.fit(X=X)
X_hyper_values = X_hyper_encoder.transform(X).astype(dtype=str)
X_hyper = pd.DataFrame(data=X_hyper_values, index=X.index, columns=X.columns)

test_set.replace(to_replace=config.ordinal_to_replace, inplace=True)
test_set = pd.DataFrame(data=X_hyper_encoder.transform(test_set),
                        index=test_set.index,
                        columns=test_set.columns).astype(int, errors='ignore')

test_set = test_set.astype(str)


def hyperopt_ctb_scoreCV_manual(params):
    global X_hyper
    global y_hyper
    global global_best_model

    for key in catboost_space:
        print(f'       {key} {params[key]}')

    clf = CatBoostClassifier(**params)
    skf = StratifiedKFold(n_splits=3)

    cross_val_result = {'estimator': [], 'test_score': []}
    for i, (train_ind, val_ind) in enumerate(skf.split(X_hyper, y_hyper)):
        train_set = Pool(data=X_hyper.loc[train_ind], label=y_hyper[train_ind], cat_features=X_hyper.columns)
        val_set = Pool(data=X_hyper.loc[val_ind], label=y_hyper[val_ind], cat_features=X_hyper.columns)

        clf.fit(X=train_set, eval_set=val_set, use_best_model=True)
        cross_val_result['estimator'].append(clf)
        cross_val_result['test_score'].append(clf.get_best_score()['validation']['AUC'])

    # current_score = clf.get_best_score()['validation']['AUC']
    current_score = np.mean(cross_val_result['test_score'])

    if current_score > global_best_model['AUC']:
        global_best_model['AUC'] = current_score
        global_best_model['model'] = cross_val_result['estimator']
        print(f'new best AUC = {current_score}')

    result = {
        'loss': -current_score,
        'status': STATUS_OK,

        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # 'model': clf,
        # -- attachments are handled differently
        'attachments':
            {'attachments': 'attachments'}
    }

    return result


catboost_space = {

    # 'grow_policy': hp.choice('grow_policy', ['Lossguide', 'Depthwise']), # 'SymmetricTree',  #  'Depthwise',
    # 'auto_class_weights': hp.choice('auto_class_weights', ['SqrtBalanced','Balanced']),
    # 'langevin': True,  # CPU only
    # 'learning_rate': hp.quniform('learning_rate', 1e-3, 3e-2, 1e-3),
    'depth': hp.choice('depth', np.arange(5, 14, dtype=int)),
    'l2_leaf_reg': hp.quniform('l2_leaf_reg', .5, 300, .5),
    'max_leaves': hp.choice('max_leaves', np.arange(5, 64, dtype=int)),
    'border_count': hp.choice('border_count', np.arange(64, 256, dtype=int)),

    'random_strength': hp.quniform('random_strength', 1e-3, 8e-1, 1e-3),
    'bagging_temperature': hp.quniform('bagging_temperature', 1e-3, 8e-1, 1e-3),
}

# Hyperopt main loop

# global_hyperopt_counter = 0
# hyper_best_params = {}


if __name__ == "__main__":

    models = {}
    for col in y.columns:
        print('********* Hyperopt main loop', col, '*****************')

        global_best_model = {'AUC': -1, 'model': None}

        y_hyper = y[col]

        hyperopt_local_params = config.params.copy()
        hyperopt_local_params.update(catboost_space)

        best = fmin(fn=hyperopt_ctb_scoreCV_manual,
                    space=hyperopt_local_params,
                    algo=tpe.suggest,
                    max_evals=30,
                    verbose=True,
                    )

        models[col] = global_best_model


    print("submission")

    submission_df = pd.DataFrame(index=test_set.index)
    for i, (col, best_model_dict) in enumerate(models.items()):
        test_pool = Pool(test_set, cat_features=test_set.columns)

        predictions_df = pd.DataFrame(index=test_set.index)
        for model in best_model_dict['model']:
            predictions_df[i] = model.predict_proba(test_pool)[:, 1]

        submission_df[col] = predictions_df.mean(axis=1)


    submission_df.to_csv("./flu_submission_ycloud.csv")

    print('*********   FINISH  ************')



