"""
Here we model the Ground Probability distribution Function (PDF) of the similarity scores
taxonomy labels with 1000 random Wikipedia articles.
We consider three candidate PDFs: Gaussian, Gumbel and Log-Normal distribution.
We use Baysian Information Critiria (BIC) to select the most appropriate distribution: the 
BIC of each PDF is computed for every label and averaged for each dataset.
"""

from typing import List
from glob import glob
import numpy as np
from scipy.stats import gumbel_r, norm, expon, halfnorm, lognorm
from src.utils import FileIO
from src.hyper_inference import DistributionEstimator
from src.encoders import ZeroShooterZSTC
from src.dataset import WebOfScience, AmazonHTC, DBPedia
from globals import Paths


def BIC(data_points: List[float], pdf, k: int, n_bins: int) -> float:
    '''Compute Baysian Information Criterion (BIC)

    Parameters
    ----------
    model: Any  - fitted model we are computing the BIC on
    k: int      - number  of free parameters of the model
    n_bins: int - number of bins to build the pdf out of the data_points

    Return
    ------
    bic_score: flaot - BIC score of the model
    '''
    # Build histogram to build pdf of datapoints.
    Y, bins = np.histogram(data_points, bins=n_bins, density=True)
    # Set X values as the center of the bins.
    X = np.array([(bins[i + 1] + bins[i]) / 2 for i in range(len(bins) - 1)])
    Y_hat = pdf(X)
    n = len(Y)
    sigma_e = np.sum((Y - Y_hat) ** 2) / n
    # Compute and return BIC.
    return n * np.log(sigma_e) + k * np.log(n)


if __name__ == '__main__':

    DATASETS = {'Wos': WebOfScience, 'DBPedia': DBPedia, 'Amazon': AmazonHTC}

    for name, DataSet in DATASETS.items():

        COMPUTE_SCORES = False
        if COMPUTE_SCORES:
            data = DBPedia('test', 1)
            labels = data.labels_flat
            encoder = ZeroShooterZSTC('all-mpnet-base-v2')
            wiki_docs = [FileIO.read_text(filename) for filename in glob(f'{Paths.WIKI_DIR}/*')]
            scores = encoder.compute_labels_scores(wiki_docs, labels)
            label2scores = {l: [float(x) for x in s] for l, s in zip(labels, np.transpose(scores))}
            FileIO.write_json(label2scores, f'{Paths.MAIN_DIR}/saves/{name}Labels2wikiscores.json')


        print(f"\n\n---------------------  {name} --------------------\n")

        label2scores = FileIO.read_json(f'{Paths.MAIN_DIR}/saves/{name}Labels2wikiscores.json')
        n_labels = len(label2scores)
        FUNCS = [
            {'name': 'Gaussian', 'func': norm, 'n_pars': 2},
            {'name': 'Gumbel', 'func': gumbel_r, 'n_pars': 2},
            {'name': 'LogNorm', 'func': lognorm, 'n_pars': 3}
        ]

        for function in FUNCS:

            name = function['name']
            func = function['func']
            n_pars = function['n_pars']

            avg_bic = 0
            for label, label_scores in label2scores.items():
                fitted_pars = func.fit(label_scores)
                rv = func(*fitted_pars)
                avg_bic += BIC(label_scores, rv.pdf, n_pars, 100)

            print(f'Dataset: {name}, avg. BIC: {avg_bic / n_labels}')
