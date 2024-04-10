'''
Selects N examples form training data based on entropy os similarity with labels.
In particular teh higher the entropy the hardest is the example, the more usefull
the example is supposed to be for the model to learn.
'''
from typing import List, Union
import logging
import numpy as np
from scipy.cluster.vq import vq
import torch
from torch import Tensor, LongTensor
from sklearn.cluster import KMeans
from src.dataset import WebOfScience, DBPedia, AmazonHTC
from src.zero_shooter import TaxZeroShot

logger = logging.getLogger('few-shot_TC')


class ExampleFewShot:
    def __init__(self, text: str, labels: List[str]) -> None:
        self.text = text
        self.labels = labels


class FewShotData:
    def __init__(self, zero_shooter: TaxZeroShot) -> None:
        self.zero_shooter = zero_shooter

    def select_examples(self,
                        data: Union[WebOfScience, DBPedia, AmazonHTC],
                        min_entropy: float,
                        max_entropy: float,
                        n_shots: int) -> List[ExampleFewShot]:
        logger.info('Selecting examples for few-shot training')
        # Get relevant labels (just consider first level).
        labels_relevant = data.labels_levels[0]
        # Select zero_shot probs of the wanted 'labels'.
        with torch.no_grad():
            docs_labels_prob_flat, _ = self.zero_shooter.forward(data.abstracts_embs)
        docs_labels_probs = [
            [label_probs_flat[lab] for lab in labels_relevant]
            for label_probs_flat in docs_labels_prob_flat
        ]
        # Compute entropy.
        entropies = [self.compute_entropy(prbs) for prbs in docs_labels_probs]

        # Get most representative and diverse 'n_shot' examples 
        # (with entropy in the wanted range) by selecting n_shot centroids.
        examples_ids = [
            i for i, s in enumerate(entropies) if min_entropy < s < max_entropy
        ]
        # texts = [x for i, x in enumerate(data.abstracts) if i in examples_ids]
        embs = data.abstracts_embs[examples_ids]
        centroids_ids = self.get_centroids_ids(embs, examples_ids, n_shots)
        res = [
            ExampleFewShot(
                text=data.abstracts[i],
                labels=[data.Y[j][i] for j in range(len(data.Y))]
            )
            for i in centroids_ids
        ]
        return res

    def compute_entropy(self, p: List[float]):
        """Compute normalized entropy i.e. entropy divided by the maximum
        entropy: - sum_i p_i ln(p_i) / ln(N) where N is the number of classes."""
        p = torch.stack(p)
        p[p < 0.3] = 0  # Filter out low prob. labels (helps denoise entropy).
        p = p + 1e-8  # Avoid log(0).
        sum_p = sum(p)
        p = p / sum_p  # Normalize prob.
        return (- sum(p * p.log()) / torch.tensor(p.size(0)).log())

    def get_centroids_ids(self, texts: Union[list[str], Tensor],
                          ids: list[int], n_shots: int) -> LongTensor:
        logger.debug('Computing centroids')
        ids = torch.LongTensor(ids)
        if isinstance(texts[0], str):  # Compute docs embs.
            with torch.no_grad():
                embs = self.zero_shooter.zstc_encoder.encoder.encode(texts).cpu().numpy()
        elif isinstance(texts[0], Tensor):  # Use precomputed docs embs.
            embs = texts.cpu().numpy()

        # Centroids method.
        kmeans = KMeans(n_clusters=n_shots)
        kmeans.fit(embs)
        centroids = np.array(kmeans.cluster_centers_)
        closest, _ = vq(centroids, embs)
        return ids[closest]
