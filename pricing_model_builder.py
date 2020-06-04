
import os
import re
import json
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from gensim.models import KeyedVectors, Doc2Vec

from tc_main import TopCoder
from tc_corpus import tokenize_str
from tc_pricing_models import train_word2vec_model, reduce_wv_dimensions, plot_word2vec, cosine_similarity, doc_vector_from_word_vectors

TOPCODER = TopCoder()

class ModelBuilder:
    """ A model builder that takes in the corpus in the form of dataframe, run
        word2vec for different dimensions and measure the acurracy of a model
    """

    def __init__(self, sentences, corpus_df, base_path):
        self.sentences = sentences
        self.corpus_df = corpus_df
        self.model_path = os.path.join(base_path, 'models')
        self.measure_path = os.path.join(base_path, 'measures')
        self.doc_vec_path = os.path.join(base_path, 'document_vec')

    def iterate_word2vec_training(self):
        """ Iteratively training the word2vec model."""
        for dimension in range(100, 1100, 100):
            print(f'Training model with {dimension} dimensions...')
            train_word2vec_model(sentences=self.sentences, size=dimension, save_dir=self.model_path, suffix=f'{dimension}D')
            print('Training finished')

    def train_one_doc2vec_model(self, vector_size, no_overlap=False, track=None):
        """ Train one doc2vec model."""
        doc2vec_corpus = TOPCODER.get_doc2vec_training_docs(no_overlap=no_overlap, track=track)
        d2v_model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)

        d2v_model.build_vocab(doc2vec_corpus)
        d2v_model.train(documents=doc2vec_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
        d2v_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        d2v_model.save(os.path.join(self.model_path, f'd2v_model_{vector_size}D'))

    def build_one_pricing_models_d2v(self, vec_dimensions, no_overlap=False, track=None):
        """ Build pricing model from d2v."""
        challenge_actual_prize = TOPCODER.challenge_prize_avg_score.total_prize
        challenge_actual_prize = challenge_actual_prize[challenge_actual_prize != 0]

        doc2vec_training_cha_id = [doc.tags[0] for doc in TOPCODER.get_doc2vec_training_docs(no_overlap=no_overlap, track=track)]
        valid_cha_ids = [cha_id for cha_id in challenge_actual_prize.index if cha_id in doc2vec_training_cha_id]

        len_diff = abs(len(doc2vec_training_cha_id) - len(valid_cha_ids))

        d2v_model = Doc2Vec.load(os.path.join(self.model_path, f'd2v_model_{vec_dimensions}D'))

        challenge_estimated_prize = {}
        for cha_id in valid_cha_ids:
            dv = d2v_model.docvecs[cha_id]
            top_sim_ids = [doc_id for doc_id, sim in d2v_model.docvecs.most_similar([dv], topn=11 + len_diff)]
            ids = []
            while len(ids) < 10:
                curr_id = top_sim_ids.pop(0)
                if curr_id != cha_id and curr_id in valid_cha_ids:
                    ids.append(curr_id)

            challenge_estimated_prize[cha_id] = challenge_actual_prize[challenge_actual_prize.index.isin(ids)].mean()

        challenge_estimated_prize = pd.Series(challenge_estimated_prize)
        challenge_estimated_prize.name = 'estimated_total_prize'

        pricing_model_measure_df = pd.concat([challenge_estimated_prize, challenge_actual_prize[challenge_actual_prize.index.isin(challenge_estimated_prize.index)]], axis=1)
        pricing_model_measure_df['MRE'] =\
            (pricing_model_measure_df.total_prize - pricing_model_measure_df.estimated_total_prize).abs() / pricing_model_measure_df.total_prize

        with open(os.path.join(self.measure_path, f'measure_{vec_dimensions}D.json'), 'w') as fwrite:
            pricing_model_measure_df.reset_index().to_json(fwrite, orient='records', indent=4, index=True)

        mmre = pricing_model_measure_df.MRE.mean()
        print(f'd2v_ids: {len(doc2vec_training_cha_id)} | valid ids: {len(valid_cha_ids)} | diff: {len_diff}')
        print(f'The mean MRE of pricing model with {vec_dimensions}-D word2vec is {mmre}')
        print('-' * 50)

    def build_pricing_models(self, statistic='mean'):
        """ Build pricing models for all wv."""
        for dimension in range(100, 1100, 100):
            print('-' * 15, f'Pricing model :: {dimension}-D - {statistic}', '-' * 15)
            self.build_one_pricing_model(wv_dimensions=dimension, statistic=statistic)
            print()

    def build_one_pricing_model(self, wv_dimensions, statistic='mean'):
        """ Build one pricing model with given wv."""
        with open(os.path.join(self.doc_vec_path, f'document_vec_{wv_dimensions}D.json')) as fread:
            cleaned_challenge_vec = {int(cha_id): np.array(vec) for cha_id, vec in json.load(fread).items()}

        challenge_actual_prize = TOPCODER.challenge_prize_avg_score.total_prize
        challenge_actual_prize = challenge_actual_prize[challenge_actual_prize != 0] # non-zero prize challenges
        
        non_zero_prize_cha_id = [cha_id for cha_id in cleaned_challenge_vec if cha_id in challenge_actual_prize.index]

        # calculate consine similarity of every pair of challenges, stored in DOK
        # build a DataFrame from DOK to take advantage of outstanding performance in pandas
        print('Calculating cosine similairty...')
        cosine_similarity_dok = {
            (cha_id_a, cha_id_b): cosine_similarity(cleaned_challenge_vec[cha_id_a], cleaned_challenge_vec[cha_id_b])
            for cha_id_a, cha_id_b in itertools.combinations_with_replacement(non_zero_prize_cha_id, 2)
        }
        cosine_similarity_df = pd.DataFrame.from_dict(cosine_similarity_dok, orient='index')
        cosine_similarity_df.index = pd.MultiIndex.from_tuples(cosine_similarity_df.index)
        cosine_similarity_df.index.names, cosine_similarity_df.columns = ['l0', 'l1'], ['similarity']

        # now that we have cosine similarity of every pair of challenges, we can build the prize estimation based on it
        print('Building pricing model...')
        challenge_estimated_prize = {}

        for cha_id in non_zero_prize_cha_id:
            all_challenge_similarities = cosine_similarity_df.loc[(cosine_similarity_df.index.get_level_values(0) == cha_id) | (cosine_similarity_df.index.get_level_values(1) == cha_id)]
            all_challenge_similarities.index = all_challenge_similarities.index.map(lambda ids: ids[0] if ids[0] != cha_id else ids[1])
            top10_most_similar_cha = all_challenge_similarities.similarity.sort_values(ascending=False).iloc[1: 11].index
            
            challenge_estimated_prize[cha_id] = challenge_actual_prize[challenge_actual_prize.index.isin(top10_most_similar_cha)].apply(statistic)

        challenge_estimated_prize = pd.Series(challenge_estimated_prize)
        challenge_estimated_prize.name = 'estimated_total_prize'

        # measure the accuracy of the model
        pricing_model_measure_df = pd.concat([challenge_estimated_prize, challenge_actual_prize[challenge_actual_prize.index.isin(challenge_estimated_prize.index)]], axis=1)
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
            json.dump({cha_id: vec.tolist() for cha_id, vec in cleaned_challenge_vec.items()}, fwrite)

        return cleaned_challenge_vec
