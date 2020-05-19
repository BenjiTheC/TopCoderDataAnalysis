""" Train the doc vector for a linear regression
    - Doesn't really help
"""

import os
import re
import json
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class PricingModelLinearRegression:
    """ User linear regression to predict the price."""

    base_path = os.path.join(os.curdir, 'pricing_model_0')
    doc_vec_path, mea_path, model_path, ml_model_path = 'document_vec', 'measures', 'models', 'ml_models'

    def __init__(self, folder, vec_dimension):
        self.linear_regression = LinearRegression()
        self.vec_dimension = vec_dimension

        self.measure_data_path = os.path.join(self.base_path, folder, self.mea_path, f'measure_{vec_dimension}D.json')
        self.document_vector_path = os.path.join(self.base_path, folder, self.doc_vec_path, f'document_vec_{vec_dimension}D.json')
        self.ml_model_path = os.path.join(self.base_path, folder, self.ml_model_path, f'ml_model_{vec_dimension}D')

        self.measure_df, self.document_vectors = self.read_data()

    def read_data(self) -> (pd.DataFrame, dict):
        """ Read the needed data for trainining linear regression model and measure
            the ML-based pricing model.
        """
        with open(self.measure_data_path) as f:
            measure_df = pd.read_json(f, orient='records').set_index('index').reindex(columns=['total_prize', 'estimated_total_prize', 'MRE'])

        with open(self.document_vector_path) as f:
            document_vectors = {int(cha_id): np.array(vec) for cha_id, vec in json.load(f).items()}

        return measure_df, document_vectors

    def train_LR(self):
        """ Train and save LR model."""
        training_X = test_X = [self.document_vectors[cha_id] for cha_id in self.measure_df.index]
        training_y = test_y = self.measure_df.total_prize.tolist()

        self.linear_regression.fit(training_X, training_y)

        with open(self.ml_model_path, 'wb') as fwrite:
            pickle.dump(self.linear_regression, fwrite)

        predict_y = self.linear_regression.predict(test_X)
        print(f'Model\'s mean squared error: {mean_squared_error(test_y, predict_y)}')

        return predict_y

    def train_and_update_measure(self):
        """ Train the linear regression model and update the measure_df."""
        self.measure_df['ml_estimated_total_prize'] = self.train_LR()
        self.measure_df['LR_MRE'] = (self.measure_df.total_prize - self.measure_df.ml_estimated_total_prize).abs() / self.measure_df.total_prize

        lr_mmre = self.measure_df.LR_MRE.mean()
        print(f'The MMRE of LR predicted price is {lr_mmre}')

        with open(self.measure_data_path, 'w') as fwrite:
            self.measure_df.reset_index().to_json(fwrite, orient='records', indent=4, index=True)

def main():
    """ Main entrance."""
    for track in ('all', 'develop', 'design'):
        print('*' * 80)
        print('*' * 20, f'Training models with {track}_track', '*' * (20 - (len(track) - 6)))
        print('*' * 80)
        for dimension in range(100, 1100, 100):
            print('\n', '-' * 15, f'Vector dimension {dimension}-D', '-' * 15)
            pricing_model_lr = PricingModelLinearRegression(folder=f'{track}_track', vec_dimension=dimension)
            pricing_model_lr.train_and_update_measure()

if __name__ == "__main__":
    main()
