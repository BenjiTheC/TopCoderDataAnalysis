""" Pricing Model 4
    Handpick the challenges from **DEVELOP** track with
    a specific range of prize every subtrack
"""
import os
from datetime import datetime

from tc_main import TopCoder
from tc_pricing_models import train_word2vec_model
from pricing_model_builder import TOPCODER, ModelBuilder

from gensim.models import KeyedVectors

TRACK = 'DEVELOP'
DOC_VEC_SIZE = 100

def main():
    """ Main entrance."""
    handpick_index = TOPCODER.get_handpick_dev_cha_id()

    for no_overlap in (False, True):
        for with_phrase in (False, True):
            # for stat in ('median', 'mean'):
            print('=' * 15, f'Training with {TRACK} corpus | NO_OVERLAP = {no_overlap} | WITH_PHRASE = {with_phrase}', '=' * 15)
            sentences = TOPCODER.get_word2vec_training_sentences(no_overlap=no_overlap, track=TRACK, with_phrases=with_phrase, index_filter=handpick_index)

            corpus_df = TOPCODER.get_challenge_req_remove_overlap(track=TRACK, index_filter=handpick_index) if no_overlap else TOPCODER.get_challenge_req(track=TRACK, index_filter=handpick_index)
            model_builder = ModelBuilder(sentences=sentences, corpus_df=corpus_df, base_path=os.path.join(os.curdir, 'pricing_model_4'))
            wv = train_word2vec_model(sentences=sentences, size=DOC_VEC_SIZE, save_dir=model_builder.model_path, suffix=f'{str(no_overlap)[0]}{str(with_phrase)[0]}_{DOC_VEC_SIZE}D')

            model_builder.build_one_document_vector(wv=wv, wv_dimensions=f'{str(no_overlap)[0]}{str(with_phrase)[0]}_{DOC_VEC_SIZE}')
            # model_builder.build_one_pricing_model(wv_dimensions=f'{str(no_overlap)[0]}{str(with_phrase)[0]}_{DOC_VEC_SIZE}', statistic='mean')

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    time_taken = end - start

    print(f'Time taken: {time_taken}')
    print(f'The whole running took {time_taken.days} days {time_taken.seconds // 3600} hours {(time_taken.seconds // 60) % 60} minutes')
