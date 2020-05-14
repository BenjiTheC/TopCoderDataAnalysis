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
    cha_basic_info = 'challenge_basic_info.json'

    def __init__(self):
        self.corpus = TopCoderCorpus(self.corpus_fn)
        self.challenge_prize_avg_score = self.create_df_from_json(self.cha_prz_fn, index_col='challenge_id')
        self.challenge_basic_info = self.create_df_from_json(
            self.cha_basic_info, 
            index_col='challenge_id', 
            convert_dates=['registration_start_date', 'registration_end_date', 'submission_end_date']
        )

    def create_df_from_json(self, fn, orient='records', index_col=None, convert_dates=True):
        """ Read the given json file into a pandas dataframe.
            TODO: data tech_by_start_date.json is in the form of {dt: dct_of_tech_count}
                  we should convert it in to [{dt, dct_of_tech_count}] form in get_date.py
                  so that every data file retrieved from the database are in the same format
        """
        with open(os.path.join(self.data_path, fn)) as fread:
            df = pd.read_json(fread, orient=orient, convert_dates=convert_dates)

        if index_col:
            df.set_index(index_col, inplace=True)
            if 'date' in index_col: # convert the datetime index to datetime object
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)

        return df
