""" This script is for preparing the training data for machine learning modeling.

    Here is the methodology:
    - Pair two challenges up as a group. Use the difference of their metadata/doc_vec to form training vector
    - Increase the weight of metadata, decrease weight of doc_vec
"""

import os
import json
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from tc_main import TopCoder
from tc_pricing_models import cosine_similarity

DATA_PATH = os.path.join(os.curdir, 'pricing_model_6', 'training_data')

TOPCODER = TopCoder()
FILTERED_CHALLENGE_INFO = TOPCODER.get_filtered_challenge_basic_info()
CHALLENGE_ID_COMBINATION = lambda: itertools.combinations(FILTERED_CHALLENGE_INFO.index, 2)

SUBTRACK_COMB = [sorted(subtrack_comb) for subtrack_comb in itertools.combinations_with_replacement(FILTERED_CHALLENGE_INFO.subtrack.unique(), 2)]
TECH_COMB = \
    [sorted(tech_comb) for tech_comb in itertools.combinations_with_replacement(TOPCODER.get_tech_popularity().head(30).tech_name, 2)] +\
    TOPCODER.get_tech_popularity().head(30).tech_name.to_list()

NUM_OF_COMB = int(len(FILTERED_CHALLENGE_INFO) * (len(FILTERED_CHALLENGE_INFO) - 1) * 0.5)

def calculate_cosine_similarity(doc_vec_path):
    """ Calculate the cosine similarity of every pair of documents."""
    with open(doc_vec_path) as f:
        challenge_vec = {int(cha_id): np.array(vec) for cha_id, vec in json.load(f).items() if int(cha_id) in FILTERED_CHALLENGE_INFO.index}

    cosine_similarity_dok = {(cha_id_a, cha_id_b): cosine_similarity(challenge_vec[cha_id_a], challenge_vec[cha_id_b]) for cha_id_a, cha_id_b in CHALLENGE_ID_COMBINATION()}
    cosine_similarity_df = pd.DataFrame.from_dict(cosine_similarity_dok, orient='index')
    cosine_similarity_df.index = pd.MultiIndex.from_tuples(cosine_similarity_df.index)
    cosine_similarity_df.index.names, cosine_similarity_df.columns = ['l0', 'l1'], ['consine_similarity']

    print(f'length of cos sim df: {len(cosine_similarity_df)}')

    with open(os.path.join(DATA_PATH, 'cosine_similarity.json'), 'w') as fwrite:
        cosine_similarity_df.reset_index().to_json(fwrite, orient='records')

def calculate_metadata_difference():
    """ Calculate the difference of metadata for every pair of challenges."""
    reindex_df = FILTERED_CHALLENGE_INFO.reindex(['number_of_platforms', 'number_of_technologies', 'challenge_duration', 'total_prize'], axis=1)
    file_size = 100000
    
    metadata_diff_lst = []
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        multi_idx = pd.Series({'l0': cha_id_a, 'l1': cha_id_b})
        metadata_diff_row = multi_idx.append((reindex_df.loc[cha_id_a] - reindex_df.loc[cha_id_b]).abs()).to_frame().T
        metadata_diff_lst.append(metadata_diff_row)

        if (idx + 1) % file_size == 0:
            metadata_diff_df = pd.concat(metadata_diff_lst, ignore_index=True)
            metadata_diff_df.columns = ['l0', 'l1', 'pltf_diff', 'techn_diff', 'dura_diff', 'prz_diff']

            print(f'No.{idx + 1 - file_size} - No.{idx} comb. {idx + 1}/{NUM_OF_COMB}', end='\r')

            with open(os.path.join(DATA_PATH, f'meta_data_diff_{(idx + 1) // file_size}.json'), 'w') as fwrite:
                metadata_diff_df.to_json(fwrite, orient='records')

            metadata_diff_lst = []

    if metadata_diff_lst != []:
        metadata_diff_df = pd.concat(metadata_diff_lst, ignore_index=True)
        metadata_diff_df.columns = ['l0', 'l1', 'pltf_diff', 'techn_diff', 'dura_diff', 'prz_diff']

        print(f'\nSaving one last file: {len(metadata_diff_df)} records')

        suffix = NUM_OF_COMB // file_size + 1
        with open(os.path.join(DATA_PATH, f'meta_data_diff_{suffix}.json'), 'w') as fwrite:
            metadata_diff_df.to_json(fwrite, orient='records')

def get_subtrack_combination():
    """ Record the subtrack combination of challenges."""
    subtrack_series = FILTERED_CHALLENGE_INFO['subtrack']
    file_size = 100000

    subtrack_comb_lst = []
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        subtrack_comb_lst.append({
            'l0': cha_id_a,
            'l1': cha_id_b,
            'comb_idx': SUBTRACK_COMB.index(sorted([subtrack_series[cha_id_a], subtrack_series[cha_id_b]]))
        })

        if (idx + 1) % file_size == 0:
            print(f'No.{idx + 1 - file_size} - No.{idx} comb. {idx + 1}/{NUM_OF_COMB}', end='\r')
            with open(os.path.join(DATA_PATH, f'subtrack_comb_{(idx + 1) // file_size}.json'), 'w') as fwrite:
                json.dump(subtrack_comb_lst, fwrite)
            subtrack_comb_lst = []

    if subtrack_comb_lst != []:
        print(f'\nSaving one last file: {len(subtrack_comb_lst)} records')
        suffix = NUM_OF_COMB // file_size + 1
        with open(os.path.join(DATA_PATH, f'subtrack_comb_{suffix}.json'), 'w') as fwrite:
            json.dump(subtrack_comb_lst, fwrite)

def get_tech_combination():
    """ Record the technologies combination"""
    with open(os.path.join(os.curdir, 'data', 'tech_by_challenge.json')) as f:
        tech_by_cha_rough = {cha['challenge_id']: cha['tech_lst'] for cha in json.load(f) if cha['challenge_id'] in FILTERED_CHALLENGE_INFO.index}

    top30_popular_tech = TOPCODER.get_tech_popularity().head(30).tech_name.to_list()
    print('Top 30 most popular technologies', top30_popular_tech)

    print(f'lenge of tech lst by cha: {len(tech_by_cha_rough)}')
    tech_by_cha = {}
    for cha_id, tech_lst in tech_by_cha_rough.items():
        cleaned_tech_lst = ['angularjs' if 'angular' in tech.lower() else tech.lower() for tech in tech_lst]
        tech_by_cha[cha_id] = [tech for tech in cleaned_tech_lst if tech in top30_popular_tech]

    tech_comb_lst = []
    file_size = 100000
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        tech_comb = {'l0': cha_id_a, 'l1': cha_id_b}
        tech_lst_a = [] if cha_id_a not in tech_by_cha else tech_by_cha[cha_id_a]
        tech_lst_b = [] if cha_id_b not in tech_by_cha else tech_by_cha[cha_id_b]

        if len(tech_lst_a) == 0 or len(tech_lst_b) == 0:
            tech_comb['comb_idx_lst'] = [TECH_COMB.index(t) for t in tech_lst_a or tech_lst_b]
        else:
            tech_comb['comb_idx_lst'] = [TECH_COMB.index(sorted([tech_a, tech_b])) for tech_a in tech_lst_a for tech_b in tech_lst_b]

        tech_comb_lst.append(tech_comb)

        if (idx + 1) % file_size == 0:
            print(f'No.{idx + 1 - file_size} - No.{idx} comb. {idx + 1}/{NUM_OF_COMB}', end='\r')
            with open(os.path.join(DATA_PATH, f'tech_comb_{(idx + 1) // file_size}.json'), 'w') as fwrite:
                json.dump(tech_comb_lst, fwrite)
            tech_comb_lst = []

    if tech_comb_lst != []:
        print(f'\nSaving one last file: {len(tech_comb_lst)} records')
        suffix = NUM_OF_COMB // file_size + 1
        with open(os.path.join(DATA_PATH, f'tech_comb_{suffix}.json'), 'w') as fwrite:
            json.dump(tech_comb_lst, fwrite)

def validate_challenge_id_pair():
    """ To check if the challenge id pairs are aligned in different data file."""
    for suffix in range(1, 163):
        print(f'Checking files suffix {suffix}')
        cha_dct = {}

        for fn in 'meta_data_diff', 'subtrack_comb', 'tech_comb':
            with open(os.path.join(DATA_PATH, f'{fn}_{suffix}.json')) as f:
                cha_dct[fn] = [(int(cha['l0']), int(cha['l1'])) for cha in json.load(f)]

        for cha_id_tup in zip(*list(cha_dct.values())):
            if len(set(cha_id_tup)) != 1:
                print(cha_id_tup)
                raise ValueError(f'Challenge ids not aligned! File suffix {suffix}')



if __name__ == '__main__':
    start = datetime.now()
    # calculate_cosine_similarity(TOPCODER.doc_vec_path)
    # get_tech_combination()
    # calculate_metadata_difference()
    # get_subtrack_combination()
    validate_challenge_id_pair()
    end = datetime.now()
    print(end - start)
