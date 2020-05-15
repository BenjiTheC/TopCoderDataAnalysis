
import os
import re
import json
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from tc_main import TopCoder
from tc_corpus import tokenize_str
from tc_pricing_models import train_word2vec_model, reduce_wv_dimensions, plot_word2vec, cosine_similarity, doc_vector_from_word_vectors

TOPCODER = TopCoder()

class ModelBuilder:
    """ A model builder that takes in the corpus in the form of dataframe, run
        word2vec for different dimensions and measure the acurracy of a model
    """


    def __init__(self, corpus_df: pd.DataFrame, base_path):
        self.corpus_df = corpus_df
        self.model_path = os.path.join(base_path, 'models')
        self.measure_path = os.path.join(base_path, 'measures')
        self.doc_vec_path = os.path.join(base_path, 'document_vec')

    def iterate_word2vec_training(self):
        """ Iteratively training the word2vec model."""
        sentences = [tokenize_str(req) for cha_id, req in self.corpus_df.itertuples()]
        for dimension in range(100, 1100, 100):
            print(f'Training model with {dimension} dimensions...')
            train_word2vec_model(sentences=sentences, size=dimension, save_dir=self.model_path, suffix=f'{dimension}D')
            print('Training finished')

    def build_pricing_models(self):
        """ Build pricing models for all wv."""
        for dimension in range(100, 1100, 100):
            print('-' * 15, f'Pricing model 0 :: {dimension}-D', '-' * 15)
            wv = KeyedVectors.load(os.path.join(self.model_path, f'model_{dimension}D'))
            self.build_one_pricing_model(wv=wv, wv_dimensions=dimension)
            print()

    def build_one_pricing_model(self, wv, wv_dimensions):
        """ Build one pricing model with given wv."""
        cleaned_corpus_df = self.corpus_df.loc[self.corpus_df.requirements != '']

        challenge_vec = {cha_id: doc_vector_from_word_vectors(req, wv) for cha_id, req in cleaned_corpus_df.itertuples()}
        zero_vector = {cha_id: vec for cha_id, vec in challenge_vec.items() if not isinstance(vec, np.ndarray)}
        cleaned_challenge_vec = {cha_id: vec for cha_id, vec in challenge_vec.items() if cha_id not in zero_vector}

        removed_challenge_ids = [*(set(self.corpus_df.index) - set(cleaned_corpus_df.index)), *list(zero_vector.keys())]
        print(f'Removed {len(removed_challenge_ids)} challenges that produce meaningless vectors')

        # calculate consine similarity of every pair of challenges, stored in DOK
        # build a DataFrame from DOK to take advantage of outstanding performance in pandas
        print('Calculating cosine similairty...')
        cosine_similarity_dok = {
            (cha_id_a, cha_id_b): cosine_similarity(cleaned_challenge_vec[cha_id_a], cleaned_challenge_vec[cha_id_b])
            for cha_id_a, cha_id_b in itertools.combinations_with_replacement(cleaned_challenge_vec.keys(), 2)
        }
        cosine_similarity_df = pd.DataFrame.from_dict(cosine_similarity_dok, orient='index')
        cosine_similarity_df.index = pd.MultiIndex.from_tuples(cosine_similarity_df.index)
        cosine_similarity_df.index.names, cosine_similarity_df.columns = ['l0', 'l1'], ['similarity']

        # now that we have cosine similarity of every pair of challenges, we can build the prize estimation based on it
        print('Building pricing model...')
        challenge_estimated_prize = {}
        challenge_actual_prize = TOPCODER.challenge_prize_avg_score.total_prize

        for cha_id in cleaned_challenge_vec:
            all_challenge_similarities = cosine_similarity_df.loc[(cosine_similarity_df.index.get_level_values(0) == cha_id) | (cosine_similarity_df.index.get_level_values(1) == cha_id)]
            all_challenge_similarities.index = all_challenge_similarities.index.map(lambda ids: ids[0] if ids[0] != cha_id else ids[1])
            top10_most_similar_cha = all_challenge_similarities.similarity.sort_values(ascending=False).iloc[1: 11].index
            
            challenge_estimated_prize[cha_id] = challenge_actual_prize[challenge_actual_prize.index.isin(top10_most_similar_cha)].mean()

        challenge_estimated_prize_df = pd.DataFrame.from_dict(challenge_estimated_prize, orient='index')
        challenge_estimated_prize_df.columns = ['estimated_total_prize']

        # measure the accuracy of the model
        pricing_model_measure_df = challenge_estimated_prize_df.join(challenge_actual_prize)
        pricing_model_measure_df = pricing_model_measure_df.loc[pricing_model_measure_df.total_prize != 0] # Remove the tasks with zero prize
        pricing_model_measure_df['MRE'] =\
            (pricing_model_measure_df.total_prize - pricing_model_measure_df.estimated_total_prize).abs() / pricing_model_measure_df.total_prize

        print('Pricing model built')

        with open(os.path.join(self.measure_path, f'measure_{wv_dimensions}D.json'), 'w') as fwrite:
            pricing_model_measure_df.reset_index().to_json(fwrite, orient='records', indent=4, index=True)

        mmre = pricing_model_measure_df.MRE.mean()
        print(f'The mean MRE of pricing model 0 with {wv_dimensions}-D word2vec is {mmre}')
        print('-' * 50)

    def build_document_vectors(self):
        """ Build document vectors for all wv."""
        for dimension in range(100, 1100, 100):
            print('-' * 15, f'Document vector 0 :: {dimension}-D', '-' * 15)
            wv = KeyedVectors.load(os.path.join(self.model_path, f'model_{dimension}D'))
            self.build_one_document_vector(wv=wv, wv_dimensions=dimension)
            print()

    def build_one_document_vector(self, wv, wv_dimensions):
        """ Build the document vectors from trained word vectors, store it in the disk."""
        cleaned_corpus_df = self.corpus_df.loc[self.corpus_df.requirements != '']

        challenge_vec = {cha_id: doc_vector_from_word_vectors(req, wv) for cha_id, req in cleaned_corpus_df.itertuples()}
        zero_vector = {cha_id: vec for cha_id, vec in challenge_vec.items() if not isinstance(vec, np.ndarray)}

        cleaned_challenge_vec = {cha_id: vec for cha_id, vec in challenge_vec.items() if cha_id not in zero_vector}

        with open(os.path.join(self.doc_vec_path, f'document_vec_{wv_dimensions}D.json'), 'w') as fwrite:
            json.dump({cha_id: vec.tolist() for cha_id, vec in cleaned_challenge_vec.items()}, fwrite, indent=4)

        return cleaned_challenge_vec

def main():
    """ Main entrance."""
    challenge_req = TOPCODER.corpus.get_challenge_req_sentences(as_dataframe=True)
    for track in ('ALL', 'DEVELOP', 'DESIGN'):
        print('*' * 80)
        print('*' * 20, f'Training models with {track} challenges', '*' * (20 - (len(track) - 6)))
        print('*' * 80)

        challenge_id_by_track = TOPCODER.challenge_basic_info.index if track == 'ALL' else TOPCODER.challenge_basic_info.loc[TOPCODER.challenge_basic_info.track == track].index
        corpus = challenge_req.loc[challenge_req.index.isin(challenge_id_by_track)]
        model_builder = ModelBuilder(corpus_df=corpus, base_path=os.path.join(os.curdir, 'pricing_model_0', f'{track.lower()}_track'))
        # model_builder.iterate_word2vec_training()
        # model_builder.build_pricing_models()
        model_builder.build_document_vectors()
        print()

if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    time_taken = end - start
    print(time_taken)
    print(f'The whole running took {time_taken.days} days {time_taken.seconds // 3600} hours {(time_taken.seconds // 60) % 60} minutes')
