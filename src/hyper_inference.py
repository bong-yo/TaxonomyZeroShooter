"""
Here we try to infer the hyper-parameters alpha and beta of USP, purely from 
the distribution on the data, without using any labelled document.
"""
from typing import List, Tuple, Dict
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r, norm, expon, halfnorm
from src.utils import FileIO
from src.encoders import ZeroShooterZSTE
from globals import Paths


class DistributionEstimator:
    """Fit Right-Gumbel distribution on the similarity scores between each label and 
    each document,
    return mean and sigma of the distribution."""
    GAMMA = 0.5772156649015329  # Gamma Eulero-Mascheroni.
    RANGE = (-0.2, 0.5)
    X = np.linspace(RANGE[0], RANGE[1], 1000)

    def __init__(self) -> None:
        pass

    @classmethod
    def fit_gaussian(cls, similarities: np.array) -> Tuple:
        mu, sigma = norm.fit(similarities)
        pdf = norm.pdf(cls.X, mu, sigma)
        return mu, sigma, pdf

    @classmethod
    def fit_gumbel(cls, similarities: np.array) -> Tuple:
        mu, beta = gumbel_r.fit(similarities)
        mean = mu + cls.GAMMA * beta
        sigma = (beta * np.pi) / np.sqrt(6)
        pdf = gumbel_r.pdf(cls.X, mu, beta)
        return mean, sigma, mu, beta, pdf

    @classmethod
    def fit_exp(cls, similarities: np.array) -> Tuple:
        loc, scale = expon.fit(similarities)
        pdf = expon.pdf(cls.X, loc, scale)
        return loc, scale, pdf

    @classmethod
    def fit_halfnorm(cls, similarities: np.array) -> Tuple:
        mu, sigma = expon.fit(similarities)
        pdf = halfnorm.pdf(cls.X, mu, sigma)
        return mu, sigma, pdf
    
    @classmethod
    def plot(cls, y: np.array, pdf, pdf_label: str, plot_name: str) -> None:
        plt.hist(y, bins=200, range=cls.RANGE, label='Z-STE Ground PDF', density=True)
        plt.plot(cls.X, pdf, label=pdf_label)
        plt.legend(loc='upper right')
        plt.savefig(plot_name)


class VarianceEstimator:
    def __init__(self, docs_folder: str, encoder: ZeroShooterZSTE) -> None:
        independent_documents = docs_folder
        self.texts = [
            FileIO.read_text(filename) 
            for filename in independent_documents
        ]
        self.encoder = encoder
    
    def estimate_gumbel(self, labels: List[str]) -> Dict[str, float]:
        """Estimate Gumbel mean and sigma on the ground Wikipedia articles
        for each label in the taxonomy
        
        Returns
        -------
        label2mean: Dict[str, float] - For each label, mean of Gumble ditribution fit 
                                        on similarity scores of label with every document 
                                        of the ground wikipedia article.
        label2sigma: Dict[str, float] - For each label, sigma of Gumble ditribution.
        """
        scores = self.encoder.compute_labels_scores(
            self.texts, labels, encoding_method='base'
        )
        label2mean, label2sigma = {}, {}
        for i, label in enumerate(labels):
            label_scores = scores[:, i]  # Scores of the label in all documents.
            mean, sigma, _, _, _ = DistributionEstimator.fit_gumbel(label_scores)
            label2mean[label] = mean
            label2sigma[label] = sigma
        return label2mean, label2sigma

    def estimate_naive(self, labels: List[str]) -> Dict[str, float]:
        docs_labels_scores = self.encoder.compute_labels_scores(
            self.texts, labels, encoding_method='base'
        )
        docs_labels_scores = abs(docs_labels_scores)
        sigmas = np.sqrt(np.power(docs_labels_scores, 2).sum(axis=0) / docs_labels_scores.shape[0] )
        return {label: sigma for label, sigma in zip(labels, sigmas)}


    