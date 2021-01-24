import pandas as pd
from pathlib import Path
import sys

sys.path.append('../')
import config


def make_corpus_without_labels(df):
    # make body without labels
    # same result for train and test
    df.replace(to_replace=config.ordinal_to_replace, inplace=True)
    result = pd.DataFrame(index=df.index)
    for i, col in enumerate(df):
        result[col] = f'{i}_' + df[col].astype(str)

    return result


def make_test_corpus(df, starspace_path=Path.cwd(), fname='test_corpus.txt'):
    result = make_corpus_without_labels(df)
    result = pd.DataFrame(data=[" ".join(item) for item in result.values.astype(str)],
                          index=df.index)

    result.to_csv(f'{starspace_path/fname}')

    return fname


def make_train_starspace_corpus(X_starspace_df, y_starspace_df, starspace_path, to_file=False):

    if not to_file:
        return ['h1n1_vaccine', 'seasonal_vaccine', 'both_labels']

    # make corpus data  for starspace
    X = make_corpus_without_labels(X_starspace_df)

    # add labels h1n1, seasonal
    results = {}
    both_labels_df = X.copy()
    for col in y_starspace_df:
        X.drop(labels=['h1n1_vaccine', 'seasonal_vaccine'],
               inplace=True, axis=1, errors='ignore')
        X[col] = '__label__' + y_starspace_df[col].astype(str)
        both_labels_df[col] = X[col]
        results[col] = X

    results['both_labels'] = both_labels_df

    for fname, df in results.items():
        df.to_csv(f'{starspace_path/fname}.txt', sep=' ', header=False, index=False)

    return results.keys()

