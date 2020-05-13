""" Functions for training & analyziing the Word2Vec model and building the pricing model for TopCoder"""

import os
import re
import json
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

def train_word2vec_model(sentences, save_when_finished=True, size=200, save_dir=None, suffix=None):
    """ Train and save the word2vec model wv."""
    default_save_dir = os.path.join(os.curdir, 'models')
    default_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')

    model = Word2Vec(sentences=sentences, size=size, workers=6)
    trained_wv = model.wv
    if save_when_finished:
        save_path = os.path.join(save_dir or default_save_dir, f'model_{suffix or default_suffix}')
        trained_wv.save(save_path)

    return trained_wv

def cosine_similarity(vec_a, vec_b):
    """ Calculate the cosine similarity of two given vectors vec_a, vec_b
        The method `consine_similarity` from sci-kit learn sklearn.metrics.pairwise
        is 10-time slower than the implementation in numpy dot & norm
    """
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def doc_vector_from_word_vectors(doc: str, wv: KeyedVectors) -> (np.ndarray):
    """ For a given document, find the words that exist in the KeyedVectors of a 
        trained word2vec model. Use the combination of the vectors to represent
        the given document
    """
    doc_vocabulary = {word for word in doc.split() if word in wv.vocab}
    return sum([wv[word] for word in doc_vocabulary])


def reduce_wv_dimensions(wv):
    """ Reduce the dimensions of a given vector space to 2D
        - src: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#visualising-the-word-embeddings
    """
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in wv.vocab:
        vectors.append(wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return pd.DataFrame.from_dict({'word': labels, 'x': x_vals, 'y': y_vals}, orient='columns')

def plot_word2vec(wv_2D, label_lst):
    """ Ploting the reduced dimension word embedding."""
    fig = plt.figure(figsize=(8, 8), dpi=200)

    with sns.axes_style('ticks'):
        ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
        sns.scatterplot(
            data=wv_2D, 
            x='x', 
            y='y',
            alpha=0.5,
            size=1,
            linewidth=0.2,
            ax=ax
        )

        for idx, word, x, y in wv_2D.loc[wv_2D.word.isin(label_lst)].itertuples():
            ax.text(
                x=x,
                y=y,
                s=word,
                ha='right'
            )
