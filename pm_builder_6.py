""" Build pricing model with random forest."""
import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from tc_main import TopCoder

TOPCODER = TopCoder()
FILT_CHA_INFO = TOPCODER.get_filtered_challenge_basic_info()
RANDOM_INDEX = pd.Series(FILT_CHA_INFO.index).sample(n=568)

RANDOM_INDEX.to_json('temp_random_index.json')

def grow_forest():
    """ Grow random forest."""
    n_estimators = 100
    rf_clf = RandomForestClassifier(n_jobs=-1, warm_start=True)
    X_test, y_test = [], []

    for i in range(1, 163):
        X_df = pd.read_json(f'pricing_model_6/training_data/ridx_process_X_{i}.json').set_index(['level_0', 'level_1'])
        y_df = pd.read_json(f'pricing_model_6/training_data/ridx_process_y_{i}.json').set_index(['level_0', 'level_1'])

        X = X_df.loc[~X_df.index.get_level_values(0).isin(RANDOM_INDEX) & ~X_df.index.get_level_values(1).isin(RANDOM_INDEX)].copy().to_numpy()
        y = y_df.loc[~y_df.index.get_level_values(0).isin(RANDOM_INDEX) & ~y_df.index.get_level_values(1).isin(RANDOM_INDEX)].copy().to_numpy().ravel()

        X_test.append(X_df.loc[X_df.index.get_level_values(0).isin(RANDOM_INDEX) | X_df.index.get_level_values(1).isin(RANDOM_INDEX)].copy())
        y_test.append(y_df.loc[y_df.index.get_level_values(0).isin(RANDOM_INDEX) | y_df.index.get_level_values(1).isin(RANDOM_INDEX)].copy())

        print(f'Training round #{i}: X shape {X.shape} | y shape {y.shape} | n_estimator {rf_clf.n_estimators}: {n_estimators}', end='\r')

        rf_clf.set_params(n_estimators=n_estimators)
        rf_clf.fit(X, y)
        n_estimators += 10

    X_test, y_test = pd.concat(X_test), pd.concat(y_test)
    print(f'\nX_test shape: {X_test.shape} | y_test shape: {y_test.shape}')

    clf_score = rf_clf.score(X_test.to_numpy(), y_test.to_numpy().ravel())
    print(f'Classifier score: {clf_score}')
    y_pred = rf_clf.predict(X_test.to_numpy())
    y_prob = rf_clf.predict_proba(X_test.to_numpy())

    y_test['y_pred'] = y_pred
    y_test.reset_index().to_json('pricing_model_6/temp_y_pred.json')

    y_prob_df = pd.DataFrame(y_prob, index=y_test.index)
    y_prob_df.to_json('pricing_model_6/temp_y_prob.json')

    with open('pricing_model_6/rf_clf', 'wb') as fwrite:
        pickle.dump(rf_clf, fwrite)

if __name__ == '__main__':
    start = datetime.now()
    grow_forest()
    end = datetime.now()
    print(end - start)
