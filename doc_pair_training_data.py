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

DATA_PATH = os.path.join(os.curdir, 'pricing_model_6', 'training_data_segments')
TRAINING_DATA_PATH = os.path.join(os.curdir, 'pricing_model_6', 'round1')

TOPCODER = TopCoder()
FILTERED_CHALLENGE_INFO = TOPCODER.get_filtered_challenge_basic_info()
CHALLENGE_ID_COMBINATION = lambda: itertools.combinations(FILTERED_CHALLENGE_INFO.index, 2)

SUBTRACK_COMB = [sorted(subtrack_comb) for subtrack_comb in itertools.combinations_with_replacement(FILTERED_CHALLENGE_INFO.subtrack.unique(), 2)]
TECH_COMB = \
    [sorted(tech_comb) for tech_comb in itertools.combinations_with_replacement(TOPCODER.get_tech_popularity().head(5).tech_name, 2)] +\
    TOPCODER.get_tech_popularity().head(5).tech_name.to_list()

TOP5_SUBTRACK = list(FILTERED_CHALLENGE_INFO.subtrack.value_counts().sort_values(ascending=False).head(5).index)
SUBTRACK_DEDUCTED = [*TOP5_SUBTRACK, 'OTHER']
SUBTRACK_DEDUCTED_COMB = [sorted(st_comb) for st_comb in itertools.combinations_with_replacement(SUBTRACK_DEDUCTED, 2)]

TECH_CAT_DCT = {
    'frontend': ('javascript', 'angularjs', 'css', 'html', 'reactjs', 'html5', 'jquery', 'swift', 'bootstrap', 'jsp', 'ajax'),
    'backend': ('node.js', 'java', 'swift', 'c#', 'spring', 'apex', 'python', 'postgresql', 'mongodb', 'sql', 'sql server'),
    # 'database': ('postgresql', 'mongodb', 'sql', 'sql server'),
    'framework': ('angularjs', 'reactjs', '.net', 'jquery', 'spring', 'bootstrap', 'jsp'),
    'language': ('javascript', 'java', 'swift', 'c#', 'apex', 'python'),
    'other': ('ios', 'android', 'docker', 'rest', 'api', 'elasticsearch', 'qa', 'other')
}
TECH_CAT_COMB =\
    [sorted(tech_cat_comb) for tech_cat_comb in itertools.combinations_with_replacement(TECH_CAT_DCT.keys(), 2)] +\
    list(TECH_CAT_DCT.keys())

NUM_OF_COMB = int(len(FILTERED_CHALLENGE_INFO) * (len(FILTERED_CHALLENGE_INFO) - 1) * 0.5)

# Round 0
def calculate_cosine_similarity(doc_vec_path):
    """ Calculate the cosine similarity of every pair of documents."""
    with open(doc_vec_path) as f:
        challenge_vec = {int(cha_id): np.array(vec) for cha_id, vec in json.load(f).items() if int(cha_id) in FILTERED_CHALLENGE_INFO.index}

    file_size = 100000
    cos_sim_lst = []
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        cos_sim_lst.append({
            'l0': cha_id_a,
            'l1': cha_id_b,
            'cosine': cosine_similarity(challenge_vec[cha_id_a], challenge_vec[cha_id_b])
        })

        if (idx + 1) % file_size == 0:
            print(f'No.{idx + 1 - file_size} - No.{idx} cosine similarity. {idx + 1}/{NUM_OF_COMB}', end='\r')
            with open(os.path.join(DATA_PATH, f'cos_sim_{(idx + 1) // file_size}.json'), 'w') as fwrite:
                json.dump(cos_sim_lst, fwrite)
            cos_sim_lst = []

    if cos_sim_lst != []:
        print(f'\nSaving one last file: {len(cos_sim_lst)} records')
        suffix = NUM_OF_COMB // file_size + 1
        with open(os.path.join(DATA_PATH, f'cos_sim_{suffix}.json'), 'w') as fwrite:
            json.dump(cos_sim_lst, fwrite)

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

def get_subtrack_comb_deducted():
    """ Combine the subtrack with small amounts of challenges into one."""
    subtrack_series = FILTERED_CHALLENGE_INFO['subtrack'].copy()
    subtrack_series.loc[~subtrack_series.isin(TOP5_SUBTRACK)] = 'OTHER'
    file_size = 100000

    print('subtrack \'OTHER\' amount: ', len(subtrack_series.loc[subtrack_series == 'OTHER']))

    subtrack_comb_lst = []
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        subtrack_comb_lst.append({
            'l0': cha_id_a,
            'l1': cha_id_b,
            'comb_idx': SUBTRACK_DEDUCTED_COMB.index(sorted([subtrack_series[cha_id_a], subtrack_series[cha_id_b]]))
        })

        if (idx + 1) % file_size == 0:
            print(f'No.{idx + 1 - file_size} - No.{idx} comb. {idx + 1}/{NUM_OF_COMB}', end='\r')
            with open(os.path.join(DATA_PATH, f'subtrack_comb_dd_{(idx + 1) // file_size}.json'), 'w') as fwrite:
                json.dump(subtrack_comb_lst, fwrite)
            subtrack_comb_lst = []

    if subtrack_comb_lst != []:
        print(f'\nSaving one last file: {len(subtrack_comb_lst)} records')
        suffix = NUM_OF_COMB // file_size + 1
        with open(os.path.join(DATA_PATH, f'subtrack_comb_dd_{suffix}.json'), 'w') as fwrite:
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

def get_tech_comb_deducted():
    """ Pick top 30 most popular technologies, map it to categories than calculate combinations."""
    with open(os.path.join(os.curdir, 'data', 'tech_by_challenge.json')) as f:
        tech_by_cha_rough = {cha['challenge_id']: cha['tech_lst'] for cha in json.load(f) if cha['challenge_id'] in FILTERED_CHALLENGE_INFO.index}

    top30_popular_tech = TOPCODER.get_tech_popularity().head(30).tech_name.to_list()
    print('Top 30 most popular technologies', top30_popular_tech)
    print('TECH_CAT_COMB: ', TECH_CAT_COMB)

    print(f'lenge of tech lst by cha: {len(tech_by_cha_rough)}')
    tech_by_cha = {}
    for cha_id, tech_lst in tech_by_cha_rough.items():
        cleaned_tech_lst = ['angularjs' if 'angular' in tech.lower() else tech.lower() for tech in tech_lst]
        cleaned_popular_tech_lst = [tech for tech in cleaned_tech_lst if tech in top30_popular_tech]
        tech_by_cha[cha_id] = list({cat_cls for tech in cleaned_popular_tech_lst for cat_cls, cat_lst in TECH_CAT_DCT.items() if tech in cat_lst})

    tech_comb_lst = []
    file_size = 100000
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        tech_comb = {'l0': cha_id_a, 'l1': cha_id_b}
        tech_lst_a = [] if cha_id_a not in tech_by_cha else tech_by_cha[cha_id_a]
        tech_lst_b = [] if cha_id_b not in tech_by_cha else tech_by_cha[cha_id_b]

        if len(tech_lst_a) == 0 or len(tech_lst_b) == 0:
            tech_comb['comb_idx_lst'] = [TECH_CAT_COMB.index(t) for t in tech_lst_a or tech_lst_b]
        else:
            tech_comb['comb_idx_lst'] = [TECH_CAT_COMB.index(sorted([tech_a, tech_b])) for tech_a in tech_lst_a for tech_b in tech_lst_b]

        tech_comb_lst.append(tech_comb)

        if (idx + 1) % file_size == 0:
            print(f'No.{idx + 1 - file_size} - No.{idx} comb. {idx + 1}/{NUM_OF_COMB}', end='\r')
            with open(os.path.join(DATA_PATH, f'tech_cat_comb_{(idx + 1) // file_size}.json'), 'w') as fwrite:
                json.dump(tech_comb_lst, fwrite)
            tech_comb_lst = []

    if tech_comb_lst != []:
        print(f'\nSaving one last file: {len(tech_comb_lst)} records')
        suffix = NUM_OF_COMB // file_size + 1
        with open(os.path.join(DATA_PATH, f'tech_cat_comb_{suffix}.json'), 'w') as fwrite:
            json.dump(tech_comb_lst, fwrite)

def validate_challenge_id_pair():
    """ To check if the challenge id pairs are aligned in different data file."""
    for suffix in range(1, 163):
        print(f'Checking files suffix {suffix}')
        cha_dct = {}

        for fn in 'meta_data_diff', 'cos_sim', 'tech_comb_vec', 'st_comb_vec':
            with open(os.path.join(DATA_PATH, f'{fn}_{suffix}.json')) as f:
                cha_dct[fn] = [(int(cha['l0']), int(cha['l1'])) for cha in json.load(f)]

        for cha_id_tup in zip(*list(cha_dct.values())):
            if len(set(cha_id_tup)) != 1:
                print(cha_id_tup)
                raise ValueError(f'Challenge ids not aligned! File suffix {suffix}')

def render_vector(dimension, one_idx):
    """ Render the N-dimension 0-1 vector of one in given index."""
    vec = np.zeros(dimension, dtype=int)
    vec[one_idx] = 1
    return vec

def construct_training_data_round_0():
    """ Construct training data X and y."""
    meta_data_optimum = pd.read_json(os.path.join(os.curdir, 'pricing_model_6', 'meta_data_stat.json')).loc[['min', 'max']].drop('prz_diff', axis=1)

    cos_sim_intv = np.linspace(-1, 1, 11)[:-1]
    md_intv_dct = {col: np.linspace(meta_data_optimum.loc['min', col], meta_data_optimum.loc['max', col], 11)[:-1] for col in meta_data_optimum.columns}

    X_chunks, y_chunks = [], []

    for i in range(1, 163):
        print(f'Processing chunk No.{i}', end='\r')
        cos_val_df = pd.read_json(os.path.join(DATA_PATH, f'cos_sim_{i}.json'), orient='records').set_index(['l0', 'l1'])
        md_val_df = pd.read_json(os.path.join(DATA_PATH, f'meta_data_diff_{i}.json'), orient='records').set_index(['l0', 'l1'])
        st_comb_df = pd.read_json(os.path.join(DATA_PATH, f'subtrack_comb_dd_{i}.json'), orient='records').set_index(['l0', 'l1'])
        tech_comb_df = pd.read_json(os.path.join(DATA_PATH, f'tech_cat_comb_{i}.json'), orient='records').set_index(['l0', 'l1'])

        cos_mapped_df = cos_val_df.apply(lambda v: render_vector(10, np.searchsorted(cos_sim_intv, v, side='right') - 1), axis=1)
        md_mapped_df = md_val_df.reindex(['pltf_diff', 'techn_diff', 'dura_diff'], axis=1).apply({col: lambda v: render_vector(10, np.searchsorted(intv, v, side='right') - 1) for col, intv in md_intv_dct.items()})
        st_mapped_df = st_comb_df.apply(lambda v: render_vector(len(SUBTRACK_DEDUCTED_COMB), v), axis=1)
        tech_mapped_df = tech_comb_df.apply(lambda v: render_vector(len(TECH_CAT_COMB), tuple(v)), axis=1)

        cos_vec_df = pd.DataFrame.from_records(cos_mapped_df, index=cos_mapped_df.index)
        md_vec_df = pd.concat([pd.DataFrame.from_records(md_mapped_df[col], index=md_mapped_df.index) for col in md_mapped_df.columns], axis=1, ignore_index=True)
        st_vec_df = pd.DataFrame.from_records(st_mapped_df, index=st_mapped_df.index)
        tech_vec_df = pd.DataFrame.from_records(tech_mapped_df, index=tech_mapped_df.index)

        X_chunk = pd.concat([cos_vec_df, md_vec_df, st_vec_df, tech_vec_df], axis=1, ignore_index=True)
        y_chunk = (md_val_df['prz_diff'] < 200).astype(int)

        X_chunk.to_json(os.path.join(DATA_PATH, f'processed_X_{i}.json'))
        y_chunk.to_json(os.path.join(DATA_PATH, f'processed_y_{i}.json'))

        X_chunks.append(X_chunk)
        y_chunks.append(y_chunk)

    del cos_val_df, md_val_df, st_comb_df, tech_comb_df
    del cos_mapped_df, md_mapped_df, st_mapped_df, tech_comb_df
    del cos_vec_df, md_vec_df, st_vec_df, tech_vec_df
    del X_chunk, y_chunk

    X, y = pd.concat(X_chunks), pd.concat(y_chunks)

    print()
    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape)

    X.to_json(os.path.join(os.curdir, 'pricing_model_6', 'X_training_data.json'))
    y.to_json(os.path.join(os.curdir, 'pricing_model_6', 'y_training_data.json'))

    return X, y

# Round 1 

def build_tech_comb_vector():
    """ Build technology combination vector."""
    with open(os.path.join(os.curdir, 'data', 'tech_by_challenge_clean.json')) as fread:
        tech_by_cha = {cha['challenge_id']: cha['tech_lst'] for cha in json.load(fread) if cha['challenge_id'] in FILTERED_CHALLENGE_INFO.index}

    print('Dimension of tech comb: ', TECH_COMB)
    tech_comb_lst = []
    file_size = 100000
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        tech_comb = {'l0': cha_id_a, 'l1': cha_id_b}
        tech_lst_a = None if cha_id_a not in tech_by_cha else tech_by_cha[cha_id_a]
        tech_lst_b = None if cha_id_b not in tech_by_cha else tech_by_cha[cha_id_b]

        if tech_lst_a is None and tech_lst_b is None:
            tech_comb['tech_comb_vec'] = np.zeros(len(TECH_COMB), dtype=int)
        elif tech_lst_a is None or tech_lst_b is None:
            tech_comb['tech_comb_vec'] = render_vector(len(TECH_COMB), [TECH_COMB.index(t) for t in tech_lst_a or tech_lst_b])
        else:
            tech_comb['tech_comb_vec'] = render_vector(len(TECH_COMB), [TECH_COMB.index(sorted((ta, tb))) for ta in tech_lst_a for tb in tech_lst_b])

        tech_comb_lst.append(tech_comb)

        if (idx + 1) % file_size == 0:
            print(f'No.{idx + 1 - file_size} - No.{idx} comb. {idx + 1}/{NUM_OF_COMB}', end=' ')
            vec_df = pd.DataFrame(tech_comb_lst).set_index(['l0', 'l1'])
            print(f'Shape of vec_df: {vec_df.shape}', end=' ')
            expd_vec_df = pd.DataFrame.from_records(vec_df['tech_comb_vec'].to_list(), index=vec_df.index).reset_index()
            print(f'Shape of expd_vec_df: {expd_vec_df.shape}', end='\r')
            expd_vec_df.to_json(os.path.join(DATA_PATH, f'tech_comb_vec_{(idx + 1) // file_size}.json'), orient='records')
            tech_comb_lst = []

    if tech_comb_lst != []:
        print(f'Save last {len(tech_comb_lst)} records', end=' ')
        vec_df = pd.DataFrame(tech_comb_lst).set_index(['l0', 'l1'])
        print(f'Shape of vec_df: {vec_df.shape}', end=' ')
        expd_vec_df = pd.DataFrame.from_records(vec_df['tech_comb_vec'].to_list(), index=vec_df.index).reset_index()
        print(f'Shape of expd_vec_df: {expd_vec_df.shape}', end='\r')
        expd_vec_df.to_json(os.path.join(DATA_PATH, f'tech_comb_vec_{NUM_OF_COMB // file_size + 1}.json'), orient='records')

def build_subtrack_comb_vector():
    """ Build subtrack combination vector."""
    subtrack_series = FILTERED_CHALLENGE_INFO['subtrack'].copy()
    subtrack_series.loc[~subtrack_series.isin(TOP5_SUBTRACK)] = 'OTHER'
    file_size = 100000

    print('subtrack \'OTHER\' amount: ', len(subtrack_series.loc[subtrack_series == 'OTHER']))

    subtrack_comb_lst = []
    for idx, (cha_id_a, cha_id_b) in enumerate(CHALLENGE_ID_COMBINATION()):
        subtrack_comb_lst.append({
            'l0': cha_id_a,
            'l1': cha_id_b,
            'st_comb_vec': render_vector(len(SUBTRACK_DEDUCTED_COMB), SUBTRACK_DEDUCTED_COMB.index(sorted([subtrack_series[cha_id_a], subtrack_series[cha_id_b]])))
        })

        if (idx + 1) % file_size == 0:
            print(f'No.{idx + 1 - file_size} - No.{idx} comb. {idx + 1}/{NUM_OF_COMB}', end=' | ')
            vec_df = pd.DataFrame(subtrack_comb_lst).set_index(['l0', 'l1'])
            print(f'Shape of vec_df: {vec_df.shape}', end=' | ')
            expd_vec_df = pd.DataFrame.from_records(vec_df['st_comb_vec'].to_list(), index=vec_df.index).reset_index()
            print(f'Shape of expd_vec_df: {expd_vec_df.shape}', end='\r')
            expd_vec_df.to_json(os.path.join(DATA_PATH, f'st_comb_vec_{(idx + 1) // file_size}.json'), orient='records')
            subtrack_comb_lst = []

    if subtrack_comb_lst != []:
        print(f'\nSaving one last file: {len(subtrack_comb_lst)} records', end=' | ')
        vec_df = pd.DataFrame(subtrack_comb_lst).set_index(['l0', 'l1'])
        print(f'Shape of vec_df: {vec_df.shape}', end=' | ')
        expd_vec_df = pd.DataFrame.from_records(vec_df['st_comb_vec'].to_list(), index=vec_df.index).reset_index()
        print(f'Shape of expd_vec_df: {expd_vec_df.shape}           ', end='\r')
        expd_vec_df.to_json(os.path.join(DATA_PATH, f'st_comb_vec_{NUM_OF_COMB // file_size + 1}.json'), orient='records')

def construct_X(file_idx):
    """ Construct training data X"""
    meta_data_optimum = pd.read_json(os.path.join(os.curdir, 'pricing_model_6', 'meta_data_stat.json')).loc[['min', 'max']].drop('prz_diff', axis=1)
    md_intv_dct = {col: np.linspace(meta_data_optimum.loc['min', col], meta_data_optimum.loc['max', col], 11)[:-1] for col in meta_data_optimum.columns}

    cos_sim_df = pd.read_json(os.path.join(DATA_PATH, f'cos_sim_{file_idx}.json'), orient='records').set_index(['l0', 'l1'])
    md_diff_df = pd.read_json(os.path.join(DATA_PATH, f'meta_data_diff_{file_idx}.json'), orient='records').set_index(['l0', 'l1']).reindex(['pltf_diff', 'techn_diff', 'dura_diff'], axis=1)
    st_comb_vec_df = pd.read_json(os.path.join(DATA_PATH, f'st_comb_vec_{file_idx}.json'), orient='records').set_index(['l0', 'l1'])
    tech_comb_vec_df = pd.read_json(os.path.join(DATA_PATH, f'tech_comb_vec_{file_idx}.json'), orient='records').set_index(['l0', 'l1'])

    md_mapped_df = md_diff_df.apply({col: lambda v: render_vector(10, np.searchsorted(intv, v, side='right') - 1) for col, intv in md_intv_dct.items()})
    md_vec_df = pd.concat([pd.DataFrame.from_records(md_mapped_df[col], index=md_mapped_df.index) for col in md_mapped_df.columns], axis=1, ignore_index=True) 

    X = pd.concat([cos_sim_df, md_vec_df, st_comb_vec_df, tech_comb_vec_df], axis=1, ignore_index=True)
    X.reset_index().to_json(os.path.join(TRAINING_DATA_PATH, f'X_{file_idx}.json'), orient='records')
    print(f'#{file_idx}: X shape: {X.shape}', end='\r')

def construct_y(file_idx, threshold):
    """ Construct training data y"""
    prz_diff = pd.read_json(os.path.join(DATA_PATH, f'meta_data_diff_{file_idx}.json'), orient='records').set_index(['l0', 'l1'])['prz_diff']
    y = (prz_diff <= threshold).astype(int)
    y.reset_index().to_json(os.path.join(TRAINING_DATA_PATH, f'y_{file_idx}.json'), orient='records')
    print(f'#{file_idx}: y shape: {y.shape}', end='\r')

def construct_Xy():
    """ Construct training data X and y"""
    print('\nConstructing X')
    for i in range(1, 163):
        construct_X(i)

    print('\nConstructing y')
    for i in range(1, 163):
        construct_y(i, 50)

def reindex_training_data():
    """ When storing multiindex dataframe/series, don't forget to reset_index before to_json
        Or you will need this function
    """
    for i in range(1, 163):
        X = pd.read_json(f'pricing_model_6/training_data/processed_X_{i}.json')
        X.index = pd.MultiIndex.from_tuples([eval(idx) for idx in X.index])
        X.reset_index().to_json(os.path.join(DATA_PATH, f'ridx_process_X_{i}.json'))
        print(f'#{i} Saved reindexed X. ', end='')

        y = pd.read_json(f'pricing_model_6/training_data/processed_y_{i}.json', typ='series')
        y.index = pd.MultiIndex.from_tuples([eval(idx) for idx in y.index])
        y.reset_index().to_json(os.path.join(DATA_PATH, f'ridx_process_y_{i}.json'))
        print('Saved reindexed y.', end='\r')

def concat_training_data():
    """ Concatenating training X and y
        - Never worked, resulting dataset is too large and not able to fit into RAM
    """
    print('Concat and saving X')
    pd.concat([pd.read_json(f'pricing_model_6/training_data/ridx_process_X_{i}.json') for i in range(1, 163)], ignore_index=True).to_json(os.path.join(os.curdir, 'pricing_model_6', 'gigantic_X.json'))
    print('X saved')
    print('Concat and saving y')
    pd.concat([pd.read_json(f'pricing_model_6/training_data/ridx_process_y_{i}.json') for i in range(1, 163)], ignore_index=True).to_json(os.path.join(os.curdir, 'pricing_model_6', 'gigantic_y.json'))
    print('y saved')

if __name__ == '__main__':
    start = datetime.now()
    # calculate_cosine_similarity(TOPCODER.doc_vec_path)
    # get_tech_combination()
    # get_tech_comb_deducted()
    # calculate_metadata_difference()
    # get_subtrack_combination()
    # get_subtrack_comb_deducted()
    # validate_challenge_id_pair()
    # construct_training_data()
    # reindex_training_data()
    # concat_training_data()
    # build_tech_comb_vector()
    # build_subtrack_comb_vector()
    construct_Xy()
    end = datetime.now()
    print(end - start)
