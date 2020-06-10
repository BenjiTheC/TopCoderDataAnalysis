""" Building Pricing Model 3
    This model is built with K-Nearest Neighboor algorithm
    X: document vectors calculated from pricing_model_0 appending meta data of challenges
    y: actual total prize
    10-fold cross validation
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from tc_main import TopCoder

TOPCODER = TopCoder()
ACTUAL_PRIZE = TOPCODER.challenge_basic_info.total_prize[TOPCODER.challenge_basic_info.total_prize != 0]

def get_path_by_track_and_dimension(track, dimension):
    """ Get document vector path by track and doc vec dimension."""
    return os.path.join(os.curdir, 'pricing_model_0', f'{track}_track', 'document_vec', f'document_vec_{dimension}D.json')

def get_path_handpick_challenge(no_overlap, with_phrase):
    """ Get document vector path from pricing model 4"""
    return os.path.join(os.curdir, 'pricing_model_4', 'document_vec', f'document_vec_{str(no_overlap)[0]}{str(with_phrase)[0]}_600D.json')

def get_challenge_meta_data():
    """ Return challenge meta data in pandas DataFrame."""
    cha_basic_info = TOPCODER.challenge_basic_info
    challenge_duration = (cha_basic_info.submission_end_date - cha_basic_info.registration_start_date).apply(lambda td: td.days)

    meta_data = pd.concat(
        [
            # cha_basic_info.reindex(['subtrack_category'], axis=1).apply({'subtrack_category': lambda c: c.cat.codes}),
            # cha_basic_info.reindex(['number_of_platforms'], axis=1),
            # cha_basic_info.reindex(['number_of_technologies'], axis=1),
            challenge_duration
        ],
        axis=1
    )

    return meta_data

def KNN_10Fold_training(doc_vec_path):
    """ Train KNN with 10-fold cross validation."""
    with open(doc_vec_path) as fread:
        doc_vec = {int(cha_id): np.array(vec) for cha_id, vec in json.load(fread).items() if int(cha_id) in ACTUAL_PRIZE.index}

    challenge_meta_data = get_challenge_meta_data()

    # Append meta at the end of document vector to form a new vector
    X = np.stack([np.concatenate([vec, challenge_meta_data.loc[cha_id].to_numpy()]) for cha_id, vec in doc_vec.items()])
    y = np.array([TOPCODER.challenge_basic_info.total_prize[cha_id] for cha_id in doc_vec.keys()]) # float is not suitable for KNN

    print(X.shape, y.shape)

    challenge_id_vec = np.array(list(doc_vec.keys()))

    label_encoder = LabelEncoder() # encoder for conversion of the float prize to integer index
    label_encoder.fit(y)

    k_fold = KFold(n_splits=10)
    result = []
    # result_cha_id = []
    for idx, (train_idx, test_idx) in enumerate(k_fold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # result_cha_id.extend(challenge_id_vec[test_idx])

        knn_classifier = KNeighborsClassifier(n_neighbors=10, weights='distance')
        knn_classifier.fit(X_train, label_encoder.transform(y_train))

        y_predict = label_encoder.inverse_transform(knn_classifier.predict(X_test)) # the result of prediction is index of prize

        mre = np.absolute(y_test - y_predict) / y_test
        result.append(mre.mean())
        print(f'round {idx} mmre: {mre.mean()}')

    avg_mean_mre = sum(result) / len(result)
    print(f'\nAverage of Mean MRE: {avg_mean_mre:.3f}\n')

    return avg_mean_mre

def main():
    """ Main entrance."""
    pm3_accuraccy = defaultdict(dict)
    for track in ('develop', ):#('all', 'develop', 'design'):
        print('=' * 15, f' Training {track} track ', '=' * 15)
        for dimension in range(100, 1100, 100):
            print('=' * 10, f'Training {dimension} dimension', '=' * 10)
            doc_vec_path = get_path_by_track_and_dimension(track, dimension)
            mean_mre = KNN_10Fold_training(doc_vec_path)
            pm3_accuraccy[track][dimension] = mean_mre

    with open(os.path.join(os.curdir, 'pricing_model_3', 'knn_pricing_model_measure_4.json'), 'w') as fwrite:
        json.dump(pm3_accuraccy, fwrite, indent=4)

def main_pm4():
    """ Main entrance for pricing model 4"""
    pm4_accuraccy = defaultdict(dict)
    for no_overlap in (False, True):
        for with_phrase in (False, True):
            print('=' * 10, f'KNN training | NO_OVERLAP = {no_overlap} | WITH_PHRASE = {with_phrase}', '=' * 10)
            doc_vec_path = get_path_handpick_challenge(no_overlap, with_phrase)
            mean_mre = KNN_10Fold_training(doc_vec_path)
            pm4_accuraccy[f'no_overlap_{no_overlap}'][f'with_phrase_{with_phrase}'] = mean_mre

    
    with open(os.path.join(os.curdir, 'pricing_model_4', 'measures', 'knn_pricing_model_measure.json'), 'w') as fwrite:
        json.dump(pm4_accuraccy, fwrite, indent=4)

if __name__ == '__main__':
    # main_pm4()
    main()
