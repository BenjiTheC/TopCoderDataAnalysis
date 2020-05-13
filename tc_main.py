""" Building the class which bundles all related data object for TopCoder."""

import os
import re
import json

import pandas as pd
import numpy as np

from tc_corpus import TopCoderCorpus

class TopCoder:
    """ Class bundling all related topcoder data object
        includnig pandas DataFrame, dict, etc.
    """

    data_path = os.path.join(os.curdir, 'data')
    corpus_fn = 'detail_requirements.json'
    cha_prz_fn = 'challenge_prz_and_score.json'

    def __init__(self):
        self.corpus = TopCoderCorpus(self.corpus_fn)
        self.challenge_prize_avg_score = self.create_df_from_json(self.cha_prz_fn, index_col='challenge_id')

    def create_df_from_json(self, fn, orient='index', index_col=None):
        """ Read the given json file into a pandas dataframe."""
        with open(os.path.join(self.data_path, fn)) as fread:
            records_obj = json.load(fread)

        if isinstance(records_obj, list):
            df = pd.DataFrame(records_obj)
        elif isinstance(records_obj, dict):
            df = pd.DataFrame.from_dict(records_obj, orient=orient)

        if index_col:
            df.set_index(index_col, inplace=True)
            if 'date' in index_col: # convert the datetime index to datetime object
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)

        return df
