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
        docs_labels_prob_flat, _ = self.zero_shooter.forward(data.abstracts, no_grad=True)
        docs_labels_probs = [
            [label_probs_flat[lab] for lab in labels_relevant]
            for label_probs_flat in docs_labels_prob_flat
        ]
        # Compute entropy.
        entropies = [self.compute_entropy(probs) for probs in docs_labels_probs]

        # Select (randomly) 'n_shot' examples with entropy in the wanted range.
        examples_ids = [
            i for i, s in enumerate(entropies) if min_entropy < s < max_entropy
        ]

        texts = [x for i, x in enumerate(data.abstracts) if i in examples_ids]
        centroids_ids = self.get_centroids_ids(texts, examples_ids, n_shots)

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
        sum_p = sum(p)
        p = [x / sum_p for x in p]
        p = torch.stack(p) + 1e-8  # Avoid log(0).
        return (- sum(p * torch.log(p)) / np.log(len(p))).tolist()

    def get_centroids_ids(self, texts: List[str], ids: List[int], n_shots: int):
        logger.debug('Computing centroids')
        # ids = torch.arange(len(texts))
        ids = torch.LongTensor(ids)
        # Filter doc embeddings within a certain range of entropy.
        embs = self.zero_shooter.encoder.encoder.encode(texts, show_progress_bar=True)
        # Centroids method.
        kmeans = KMeans(n_clusters=n_shots)
        kmeans.fit(embs)
        centroids = np.array(kmeans.cluster_centers_)
        closest, _ = vq(centroids, embs)
        return ids[closest]
