""" Pricing model 5:
    Train KNN model with hand pick challenge by the price range
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
DOC_VEC_SIZE = 100 # choose this dimensionality based on empirical reason
DOC_VEC_PATH = os.path.join(os.curdir, 'pricing_model_0', 'develop_track', 'document_vec', f'document_vec_{DOC_VEC_SIZE}D.json')

HANDPICKED_CHALLENGES = TOPCODER.get_handpick_dev_cha_id()

def get_challenge_meta_data():
    """ Return challenge meta data of handpicked challenges."""
    cbi_df = TOPCODER.challenge_basic_info.loc[TOPCODER.challenge_basic_info.index.isin(HANDPICKED_CHALLENGES)]

    challenge_duration = (cbi_df.submission_end_date - cbi_df.registration_start_date).apply(lambda td: td.days)

    meta_data = pd.concat(
        [
            cbi_df.reindex(['subtrack'], axis=1).astype('category').apply(lambda c: c.cat.codes),
            cbi_df.reindex(['number_of_platforms'], axis=1),
            cbi_df.reindex(['number_of_technologies'], axis=1),
            challenge_duration
        ],
        axis=1
    )

    return meta_data

def KNN_10Fold_trainning():
    """ Train KNN with 10-fold cross validation."""
    with open(DOC_VEC_PATH) as f:
        doc_vec = {int(cha_id): np.array(vec) for cha_id, vec in json.load(f).items() if int(cha_id) in HANDPICKED_CHALLENGES}

    challenge_ids_arr = np.array(list(doc_vec.keys())) # for indexing the predicted data in 10-fold CV
    challenge_meta_data = get_challenge_meta_data()

    # Append meta at the end of document vector to form a new vector
    X = np.stack([np.concatenate([vec, challenge_meta_data.loc[cha_id].to_numpy()]) for cha_id, vec in doc_vec.items()])
    y = np.array([TOPCODER.challenge_basic_info.total_prize[cha_id] for cha_id in doc_vec.keys()]) # float is not suitable for KNN

    print(X.shape, y.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    k_fold = KFold(n_splits=10)

    mmre_result = []
    prediction_result = []
    for idx, (train_idx, test_idx) in enumerate(k_fold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_challenge_ids = challenge_ids_arr[test_idx]

        knn_classifier = KNeighborsClassifier(n_neighbors=10, weights='distance')
        knn_classifier.fit(X_train, label_encoder.transform(y_train))

        y_predict = label_encoder.inverse_transform(knn_classifier.predict(X_test))

        mre = np.absolute(y_test - y_predict) / y_test
        mmre_result.append(mre.mean())

        prediction_result.append(
            pd.DataFrame(
                {'actual_total_prize': y_test, 'estimated_total_prize': y_predict},
                index=test_challenge_ids
            )
        )

        print(f'round {idx} mmre: {mre.mean()}')

    avg_mmre = sum(mmre_result) / len(mmre_result)

    prediction_df = pd.concat(prediction_result)
    prediction_df['MRE'] = (prediction_df['actual_total_prize'] - prediction_df['estimated_total_prize']).abs() / prediction_df['actual_total_prize']

    print('avg_mmre: {} | mmre from prediction df: {}'.format(avg_mmre, prediction_df['MRE'].mean()))

    return avg_mmre, prediction_df

def main():
    """ Main entrance."""
    print('=' * 15, 'Trainning...', '=' * 15)
    avg_mmre, prediction_df = KNN_10Fold_trainning()

    # with open(os.path.join(os.curdir, 'pricing_model_5', 'knn_measure_4.json'), 'w') as fwrite:
        # prediction_df.reset_index().to_json(fwrite, orient='records', indent=4, index=True)

if __name__ == "__main__":
    main()
