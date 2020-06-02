""" Building Pricing Model 3
    This model is built by using Doc2Vec model to do word embedding
    And find the top 10 most similar challenges, use the average of challenge prize as the estimate prize
"""

import os
import json

from tc_main import TopCoder
from pricing_model_builder import ModelBuilder

TOPCODER = TopCoder()
NO_OVERLAP = True

def main():
    """ Main entrance."""
    for track in ('ALL', 'DEVELOP', 'DESIGN'):
        print('*' * 80)
        print('*' * 20, f'Training models with {track} challenges', '*' * (20 - (len(track) - 6)))
        print('*' * 80)

        track_param = track if track != 'ALL' else None

        sentences = TOPCODER.get_word2vec_training_sentences(no_overlap=NO_OVERLAP, track=track_param)
        corpus_df = TOPCODER.get_challenge_req_remove_overlap(track=track_param)
        model_builder = ModelBuilder(sentences=sentences, corpus_df=corpus_df, base_path=os.path.join(os.curdir, 'pricing_model_1', f'{track.lower()}_track'))
        model_builder.iterate_word2vec_training()
        model_builder.build_document_vectors()
        model_builder.build_pricing_models()
        print()

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    time_taken = end - start

    print(f'Time taken: {time_taken}')
    print(f'The whole running took {time_taken.days} days {time_taken.seconds // 3600} hours {(time_taken.seconds // 60) % 60} minutes')
