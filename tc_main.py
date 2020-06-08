""" Building the class which bundles all related data object for TopCoder."""

import os
import re
import json
import difflib
from collections import defaultdict

import pandas as pd
import numpy as np

from gensim.models.doc2vec import TaggedDocument

from tc_corpus import TopCoderCorpus, tokenize_str, remove_stop_words_from_str

class TopCoder:
    """ Class bundling all related topcoder data object
        includnig pandas DataFrame, dict, etc.
    """

    data_path = os.path.join(os.curdir, 'data')
    corpus_section_similarity_path = os.path.join(data_path, 'corpus_section_similarity.json')
    cha_prz_fn = 'challenge_prz_and_score.json'
    cha_basic_info = 'challenge_basic_info.json'

    develop_challenge_prize_range = {
        'FIRST_2_FINISH': (0, 600),
        'CODE': (250, 2500),
        'ASSEMBLY_COMPETITION': (750, 2750),
        'BUG_HUNT': (0, 750),
        'UI_PROTOTYPE_COMPETITION': (1250, 2750),
        'ARCHITECTURE': (1500, 3000),
        'DEVELOP_MARATHON_MATCH': (1000, 1750),
        'COPILOT_POSTING': (150, 300),
        'TEST_SUITES': (500, 2000),
        'TEST_SCENARIOS': (500, 2000),
        'SPECIFICATION': (1500, 3000),
        'CONTENT_CREATION': (500, 2000),
        'CONCEPTUALIZATION': (1500, 2000)
    }

    def __init__(self):
        self.corpus = TopCoderCorpus()
        self.challenge_prize_avg_score = self.create_df_from_json(self.cha_prz_fn, index_col='challenge_id')
        self.challenge_basic_info = self.create_df_from_json(
            self.cha_basic_info, 
            index_col='challenge_id', 
            convert_dates=['registration_start_date', 'registration_end_date', 'submission_end_date'],
            convert_cat=['track', 'subtrack']
        )

        self.corpus_section_similarity = self.calculate_section_similarity()

    def create_df_from_json(self, fn, orient='records', index_col=None, convert_dates=None, convert_cat=None):
        """ Read the given json file into a pandas dataframe.
            TODO: data tech_by_start_date.json is in the form of {dt: dct_of_tech_count}
                  we should convert it in to [{dt, dct_of_tech_count}] form in get_date.py
                  so that every data file retrieved from the database are in the same format
        """
        with open(os.path.join(self.data_path, fn)) as fread:
            df = pd.read_json(fread, orient=orient, convert_dates=convert_dates or False)

        if index_col:
            df.set_index(index_col, inplace=True)
            if 'date' in index_col: # convert the datetime index to datetime object
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)

        if convert_cat is not None and isinstance(convert_cat, list):
            df[[f'{col}_category' for col in convert_cat]] = df[convert_cat].astype('category')

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

    def doc_collocation_extraction(self, doc: str):
        """ Detect the phrases in a doc."""
        phraser = self.corpus.phraser
        tokens = tokenize_str(doc)
        # phrases = [p for p in phraser[tokens] if p not in tokens] # return a list containing phrases and single words

        # return '{} {}'.format(doc, ' '.join(phrases))
        return ' '.join(phraser[tokens])

    def get_corpus_section_similarity_score(self, frequency_threshold=0.5, similarity_threshold=0.5):
        """ Select the wanted similarity score for given frequency and similarity threshold"""
        return self.corpus_section_similarity.loc[(self.corpus_section_similarity.freq > frequency_threshold) & (self.corpus_section_similarity.score > similarity_threshold)]

    def get_challenge_ids_by_track(self, track=None):
        """ Get the challenge ids by track for filtering purpose."""
        if track is not None and track.upper() in ('DEVELOP','DESIGN', 'DATA_SCIENCE'):
            return self.challenge_basic_info.loc[self.challenge_basic_info.track == track.upper()].index
        else:
            return self.challenge_basic_info.index

    def get_challenge_req(self, track=None, index_filter=None):
        """ Get the challenge requirements, apply track filter if one is given."""
        challenge_req = self.corpus.get_challenge_req()

        if index_filter is not None:
            challenge_req = challenge_req.loc[challenge_req.index.isin(index_filter)]

        return challenge_req.loc[challenge_req.index.isin(self.get_challenge_ids_by_track(track))]

    def get_challenge_req_remove_overlap(self, track=None, index_filter=None):
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

        if index_filter is not None:
            challenge_req = challenge_req.loc[challenge_req.index.isin(index_filter)]

        return challenge_req.loc[challenge_req.index.isin(self.get_challenge_ids_by_track(track))]

    def get_word2vec_training_sentences(self, no_overlap=False, track=None, remove_stop_words=True, with_phrases=False, index_filter=None):
        """ Return the challenge requirement text in the form of list of lists of tokens (CBOW)"""
        corpus_df = self.get_challenge_req_remove_overlap(track=track, index_filter=index_filter) if no_overlap else self.get_challenge_req(track=track, index_filter=index_filter)

        if with_phrases:
            corpus_df = corpus_df.apply({'requirements':self.doc_collocation_extraction})

        if remove_stop_words:
            corpus_df = corpus_df.apply({'requirements': remove_stop_words_from_str})

        return [tokenize_str(req) for cha_id, req in corpus_df.itertuples()]

    def get_doc2vec_training_docs(self, no_overlap=False, track=None, remove_stop_words=True):
        """ Return the challenges in the form of TaggedDoc for doc2vec training."""
        corpus_df = self.get_challenge_req_remove_overlap(track) if no_overlap else self.get_challenge_req(track)
        if remove_stop_words:
            corpus_df = corpus_df.apply({'requirements': remove_stop_words_from_str})

        return [TaggedDocument(tokenize_str(req), [cha_id]) for cha_id, req in corpus_df.itertuples()]

    def get_handpick_dev_cha_id(self):
        """ Get the challenge ids based on handpick prize range."""
        cbi_df = self.challenge_basic_info.loc[self.challenge_basic_info.total_prize > 0] # reduce the length of code each line ;-)
        return pd.concat([
            cbi_df.loc[
                (cbi_df.subtrack == subtrack) & 
                (low <= cbi_df.total_prize) & 
                (cbi_df.total_prize <= high)
            ] for subtrack, (low, high) in self.develop_challenge_prize_range.items()]).index
