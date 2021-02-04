import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
import featuretools as ft

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

pd.set_option('display.max_rows', 120)
plt.style.use("dark_background")
plt.rcParams['figure.figsize'] = (20, 15)

print(Path.cwd())

X = pd.read_csv("./input/training_set_features.csv", index_col='respondent_id')
y = pd.read_csv("./input/training_set_labels.csv", index_col='respondent_id')
submission_format = pd.read_csv("./input/submission_format.csv", index_col='respondent_id')
test_set = pd.read_csv("./input/test_set_features.csv", index_col='respondent_id')

X['test'] = False
test_set['test'] = True

X.reset_index(inplace=True, )
test_set.reset_index(inplace=True, )
X_joined = X.append(test_set, ignore_index=True, sort=False)

# split for ordinal values manually, i.e. ages groups
X_joined.replace(to_replace=config.ordinal_to_replace, inplace=True)

numeric = []
for col in config.ordinal:
    num_col = f'{col}_num'
    numeric.append(num_col)
    X_joined[num_col] = X_joined[col]

# Add dataframe to entityset
categorical_ft = dict([col, ft.variable_types.Boolean] for col in config.categorical.keys())
ordinal_ft = dict([col, ft.variable_types.Ordinal] for col in config.ordinal)
numeric_ft = dict([col, ft.variable_types.Numeric] for col in numeric)
variable_dtypes = {**categorical_ft, **ordinal_ft, **numeric_ft}

# Create an entity set
es = ft.EntitySet(id='flu')
es = es.entity_from_dataframe(entity_id='flu',
                              dataframe=X_joined,
                              index='respondent_id',
                              variable_types=variable_dtypes,
                              )

agg_primitives = ['count', 'median', 'entropy']
trans_primitives = ['add_numeric']

# Run deep feature synthesis
dfs_feat, dfs_defs = ft.dfs(entityset=es,
                            target_entity='flu',
                            trans_primitives=trans_primitives,
                            agg_primitives=agg_primitives,
                            max_features=1000,
                            chunk_size=4000,
                            verbose=True,
                            max_depth=2,
                            n_jobs=1)

test_set = dfs_feat.loc[dfs_feat.test].copy()
test_set.drop(columns=['test'], inplace=True)
test_set = test_set.astype(config.categorical)

X_hyper = dfs_feat.loc[~dfs_feat.test].copy()
X_hyper = X_hyper.astype(config.categorical)
X_hyper.drop(columns=['test'], inplace=True)

del X, dfs_feat, dfs_defs


def hyperopt_ctb_scoreCV_manual(params):
    global X_hyper
    global y_hyper
    global global_best_model

    for key in catboost_space:
        print(f'       {key} {params[key]}')

    clf = CatBoostClassifier(**params)
    skf = StratifiedKFold(n_splits=5)

    cross_val_result = {'estimator': [], 'test_score': []}
    for i, (train_ind, val_ind) in enumerate(skf.split(X_hyper, y_hyper)):
        train_set = Pool(data=X_hyper.loc[train_ind], label=y_hyper[train_ind], cat_features=config.categorical.keys())
        val_set = Pool(data=X_hyper.loc[val_ind], label=y_hyper[val_ind], cat_features=config.categorical.keys())

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
    'loss_function': hp.choice('loss_function', ['Logloss', 'CrossEntropy']),
    'depth': hp.choice('depth', np.arange(5, 14, dtype=int)),
    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 300, 1),
    'max_leaves': hp.choice('max_leaves', np.arange(5, 64, dtype=int)),
    'border_count': hp.choice('border_count', np.arange(64, 256, dtype=int)),

    'random_strength': hp.quniform('random_strength', 1e-3, 8e-1, 1e-3),
    'bagging_temperature': hp.quniform('bagging_temperature', 1e-3, 8e-1, 1e-3),
}

# Hyperopt main loop
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
                max_evals=2,
                verbose=True,
                )

    models[col] = global_best_model

print("submission")

submission_df = pd.DataFrame(index=test_set.index)
for i, (col, best_model_dict) in enumerate(models.items()):
    test_pool = Pool(test_set, cat_features=config.categorical.keys())

    predictions_df = pd.DataFrame(index=test_set.index)
    for model in best_model_dict['model']:
        predictions_df[i] = model.predict_proba(test_pool)[:, 1]

    submission_df[col] = predictions_df.mean(axis=1)

print("submission_df.shape", submission_df.shape)
print(submission_df.head())

submission_df.to_csv("./flu_submission_ycloud.csv")

print('*********   FINISH  ************')


