""" Building the class which bundles all related data object for TopCoder."""

import os
import re
import json
import difflib
from collections import defaultdict

import pandas as pd
import numpy as np

from tc_corpus import TopCoderCorpus, tokenize_str, remove_stop_words_from_str

class TopCoder:
    """ Class bundling all related topcoder data object
        includnig pandas DataFrame, dict, etc.
    """

    data_path = os.path.join(os.curdir, 'data')
    corpus_section_similarity_path = os.path.join(data_path, 'corpus_section_similarity.json')
    cha_prz_fn = 'challenge_prz_and_score.json'
    cha_basic_info = 'challenge_basic_info.json'

    def __init__(self):
        self.corpus = TopCoderCorpus()
        self.challenge_prize_avg_score = self.create_df_from_json(self.cha_prz_fn, index_col='challenge_id')
        self.challenge_basic_info = self.create_df_from_json(
            self.cha_basic_info, 
            index_col='challenge_id', 
            convert_dates=['registration_start_date', 'registration_end_date', 'submission_end_date']
        )

        self.corpus_section_similarity = self.calculate_section_similarity()

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

    def get_similarity_score(self, lst_of_str):
        """ Calculate the simliarity scroe from a list of strings"""
        seq_matcher = difflib.SequenceMatcher()
        similarity_score_sum = 0
        num_of_comb = (len(lst_of_str) * (len(lst_of_str) - 1)) / 2 or 1
        
        for idx, s in enumerate(lst_of_str[:-1]):
            seq_matcher.set_seq2(s)
            for s1 in lst_of_str[idx + 1:]:
                seq_matcher.set_seq1(s1)
                similarity_score_sum += round(seq_matcher.ratio(), 3)
                
        return round(similarity_score_sum / num_of_comb, 3)

    def calculate_section_similarity(self):
        """ Find "common sections" of a project and calculate the similarity of text in the section of challenges"""
        if os.path.isfile(self.corpus_section_similarity_path):
            with open(self.corpus_section_similarity_path) as f:
                df_similarity_score = pd.DataFrame.from_dict(
                    {(int(project_id), sec_name): sec for project_id, sections in json.load(f).items() for sec_name, sec in sections.items()},
                    orient='index'
                )
                df_similarity_score.index.names = ['project_id', 'section_name']
                return df_similarity_score

        sections_by_proj = defaultdict(lambda: defaultdict(dict))
        section_similarity_score = defaultdict(lambda: defaultdict(dict))

        for project_id, challenges in self.corpus.sectioned_requirements.groupby(level=0):
            sec_name_freq = defaultdict(int)
            sec_text_by_name = defaultdict(list)
            num_of_challenges = len(challenges.index.unique(level='challenge_id'))

            for sec_name, sec_text in challenges.droplevel(level=[0, 1]).itertuples():
                sec_name_freq[sec_name] += 1
                sec_text_by_name[sec_name].append(sec_text)

            for sec_name, freq in sec_name_freq.items():
                sections_by_proj[project_id][sec_name] = {
                    'section_name_freq': round(freq / num_of_challenges, 2),
                    'lst_of_sec_text': sec_text_by_name[sec_name]
                }

        for project_id, req_sec in sections_by_proj.items():
            for sec_name, section in req_sec.items():
                section_similarity_score[project_id][sec_name] = {
                    'score': self.get_similarity_score(section['lst_of_sec_text']),
                    'freq': section['section_name_freq']
                }

        if not os.path.isfile(self.corpus_section_similarity_path):
            with open(self.corpus_section_similarity_path, 'w') as fwrite:
                json.dump(dict(section_similarity_score), fwrite, indent=4)

        df_similarity_score = pd.DataFrame.from_dict(
            {(project_id, sec_name): sec for project_id, sections in section_similarity_score.items() for sec_name, sec in sections.items()},
            orient='index'
        )
        df_similarity_score.index.names = ['project_id', 'section_name']

        return df_similarity_score

    def get_corpus_section_similarity_score(self, frequency_threshold=0.5, similarity_threshold=0.5):
        """ Select the wanted similarity score for given frequency and similarity threshold"""
        return self.corpus_section_similarity.loc[(self.corpus_section_similarity.freq > frequency_threshold) & (self.corpus_section_similarity.score > similarity_threshold)]

    def get_challenge_ids_by_track(self, track=None):
        """ Get the challenge ids by track for filtering purpose."""
        if track is not None and track.upper() in ('DEVELOP','DESIGN', 'DATA_SCIENCE'):
            return self.challenge_basic_info.loc[self.challenge_basic_info.track == track.upper()].index
        else:
            return self.challenge_basic_info.index

    def get_challenge_req(self, track=None):
        """ Get the challenge requirements, apply track filter if one is given."""
        challenge_req = self.corpus.get_challenge_req()
        return challenge_req.loc[challenge_req.index.isin(self.get_challenge_ids_by_track(track))]

    def get_challenge_req_remove_overlap(self, track=None):
        """ Get the challenge requirements with duplicate sections removed."""
        challenge_req_sec = self.corpus.sectioned_requirements
        sec_sim_score = self.get_corpus_section_similarity_score()

        dropping_index = []
        # dropping_content = []

        for (project_id, section_name), similarity_score, frequency in sec_sim_score.itertuples():
            overlap_sec = challenge_req_sec.loc[pd.IndexSlice[project_id, :, section_name]]
            dropping_index.extend(overlap_sec.index)
            # dropping_content.append(overlap_sec.iloc[overlap_sec.requirements_by_section.str.len().argmax(), 0])

        challenge_req = challenge_req_sec.loc[~challenge_req_sec.index.isin(dropping_index)].groupby(level=1).aggregate(' '.join)
        challenge_req.columns = ['requirements']
        return challenge_req.loc[challenge_req.index.isin(self.get_challenge_ids_by_track(track))]

    def get_word2vec_training_sentences(self, no_overlap=False, track=None, remove_stop_words=True):
        """ Return the challenge requirement text in the form of list of lists of tokens (CBOW)"""
        corpus_df = self.get_challenge_req_remove_overlap(track) if no_overlap else self.get_challenge_req(track)

        if remove_stop_words:
            corpus_df = corpus_df.apply({'requirements': remove_stop_words_from_str})

        return [tokenize_str(req) for cha_id, req in corpus_df.itertuples()]
