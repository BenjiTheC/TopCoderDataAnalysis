""" Functions to preprocess challenge pari data
    Including:
    - split challenge Id into 10 part randomly (10-fold CV), 568 challenges each part
    - get the training X and y (5112 * 5111 * 0.5)
    - undersample / oversample the imblanced dataset

    NOTE: I'm using shorthanded var name here, which should not be recommended...
          But I know what they mean, so anyway..
"""

import os
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from tc_main import TopCoder

PP_PATH = os.path.join(os.curdir, 'pricing_model_6', 'preprocess_data')
PP_DATA = {
    'splt_cha': os.path.join(PP_PATH, 'split_challenges.json'),
}

XY_PATH = {
    'X': os.path.join(os.curdir, 'pricing_model_6', 'round1', 'X_{}.json'),
    'y': os.path.join(os.curdir, 'pricing_model_6', 'round1', 'y_{}.json')
}
RESULT_PATH = os.path.join(os.curdir, 'pricing_model_6', 'round1_res')

TC = TopCoder()
FILT_CHA_INFO = TC.get_filtered_challenge_basic_info()

def split_challenges():
    """ Split challenges into 10 equal part randomly, with proportionally divided challenges by subtrack.
        It's randomly splited but consistant with a fixed random_state param.
    """
    cha_id_sr = pd.Series(FILT_CHA_INFO.index)
    split_cha_id = [splt_ids.to_list() for splt_ids in np.array_split(cha_id_sr.sample(frac=1, random_state=0), 10)]
    with open(PP_DATA['splt_cha'], 'w') as fwrite:
        json.dump(split_cha_id, fwrite, indent=4)

def get_train_test_Xy(X: pd.DataFrame, y: pd.DataFrame, chunk_idx: int):
    """ Get train X, test X, train y, test y for given chunk of challenge ids."""
    with open(PP_DATA['splt_cha']) as fread:
        split_cha_id = json.load(fread)

    test_cha_id = split_cha_id[chunk_idx]
    l0_in_test = y.index.get_level_values(0).isin(test_cha_id) # using y index considering it's much smaller than X
    l1_in_test = y.index.get_level_values(1).isin(test_cha_id) # so it may be quicker (NOT REALLY...)

    # return X_train, X_test, y_train, y_test
    return (
        X.loc[~l0_in_test & ~l1_in_test], # X_train
        X.loc[(l0_in_test | l1_in_test) & ~(l0_in_test & l1_in_test)], # X_test
        y.loc[~l0_in_test & ~l1_in_test], # y_train
        y.loc[(l0_in_test | l1_in_test) & ~(l0_in_test & l1_in_test)], # y_test
    )

def main():
    """ Main entrance."""
    print('Reading X...')
    X = pd.concat([pd.read_json(XY_PATH['X'].format(i), orient='records') for i in range(1, 163)]).set_index(['l0', 'l1'])
    print('Reading y...')
    y = pd.concat([pd.read_json(XY_PATH['y'].format(i), orient='records') for i in range(1, 163)]).set_index(['l0', 'l1'])

    print('\nTraining Inner sampler RFC')
    for i in range(10):
        print(f'Training 10-Fold CV #{i}', end='\r')
        X_train, X_test, y_train, y_test = get_train_test_Xy(X, y, i)

        balanced_rfc = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
        balanced_rfc.fit(X_train.to_numpy(), y_train.to_numpy().ravel())

        pd.DataFrame(balanced_rfc.predict_proba(X_test.to_numpy()), index=y_test.index).reset_index().to_json(os.path.join(RESULT_PATH, 'brf', f'y_prob_{i}.json'), orient='records')
        pd.Series(balanced_rfc.feature_importances_).to_json(os.path.join(RESULT_PATH, 'brf', f'feature_importance_{i}.json'))

    print('\nTraining RandomUnderSampler')
    for i in range(10):
        print(f'Training 10-Fold CV #{i}', end='\r')
        X_train, X_test, y_train, y_test = get_train_test_Xy(X, y, i)

        rfc = RandomForestClassifier(n_estimators=100, random_state=0)
        rus = RandomUnderSampler(random_state=0)

        X_resample, y_resample = rus.fit_resample(X_train.to_numpy(), y_train.to_numpy().ravel())
        rfc.fit(X_resample, y_resample)

        pd.DataFrame(rfc.predict_proba(X_test.to_numpy()), index=y_test.index).reset_index().to_json(os.path.join(RESULT_PATH, 'rus', f'y_prob_{i}.json'), orient='records')
        pd.Series(rfc.feature_importances_).to_json(os.path.join(RESULT_PATH, 'rus', f'feature_importance_{i}.json'))


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(end - start)
