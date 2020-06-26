""" Build pricing model with random forest."""
import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from tc_main import TopCoder
from pair_cha_imbl_learning import PP_DATA

TOPCODER = TopCoder()
FILT_CHA_INFO = TOPCODER.get_filtered_challenge_basic_info()

def prz_estimation_from_prob(y_prob_path, target_ids: list):
    """ Estimate challenge prize from top most confident predictions."""
    prob_df = pd.read_json(y_prob_path, orient='records').set_index(['l0', 'l1'])

    prz_estimation = []
    for cha_id in target_ids:
        cha_pair = prob_df.loc[(prob_df.index.get_level_values(0) == cha_id) | (prob_df.index.get_level_values(1) == cha_id)].copy()
        cha_pair.index = cha_pair.index.map(lambda cha_ids: cha_ids[0] if cha_ids[0] != cha_id else cha_ids[1])

        top_one_cha_ids = cha_pair['1'].sort_values(ascending=False).head(51).index
        prz_estimation.append({
            'challenge_id': cha_id,
            'actual': FILT_CHA_INFO.total_prize[cha_id],
            'median': FILT_CHA_INFO.total_prize[FILT_CHA_INFO.index.isin(top_one_cha_ids)].median(),
            'mean': FILT_CHA_INFO.total_prize[FILT_CHA_INFO.index.isin(top_one_cha_ids)].mean()
        })

    return prz_estimation

def estimate_and_measure_prize(res_folder):
    """ Estimate the prize from prediction result."""
    with open(PP_DATA['splt_cha']) as f:
        split_cha_id = json.load(f)

    whole_prz_estimation = []
    for idx, target_ids in enumerate(split_cha_id):
        whole_prz_estimation.extend(prz_estimation_from_prob(os.path.join(res_folder, f'y_prob_{idx}.json'), target_ids))

    prz_estimation_df = pd.DataFrame.from_records(whole_prz_estimation)
    prz_estimation_df.to_json(os.path.join(res_folder, 'prz_estimation.json'))
    mmre_median = ((prz_estimation_df['actual'] - prz_estimation_df['median']).abs() / prz_estimation_df['actual']).mean()
    mmre_mean = ((prz_estimation_df['actual'] - prz_estimation_df['mean']).abs() / prz_estimation_df['actual']).mean()

    print(f'MMRE median: {mmre_median}\n MMRE mean: {mmre_mean}')

if __name__ == '__main__':
    start = datetime.now()
    estimate_and_measure_prize(os.path.join(os.curdir, 'pricing_model_6', 'round1_res', 'rus'))
    end = datetime.now()
    print(end - start)
