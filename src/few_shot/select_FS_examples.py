'''
Selects N examples form training data based on entropy os similarity with labels.
In particular teh higher the entropy the hardest is the example, the more usefull 
the example is supposed to be for the model to learn.
'''
from typing import List, Union, Tuple
import json
from tqdm import tqdm
import logging
import math
import numpy as np
from scipy.cluster.vq import vq
from random import shuffle
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
from src.dataset import WebOfScience, DBPedia, AmazonHTC
from src.zero_shooter import TaxZeroShot

logger = logging.getLogger('few-shot_TC')


class ExampleFewShot:
    def __init__(self, text: str, labels: List[str]) -> None:
        self.text = text
        self.labels = labels


class FewShotData:
    def __init__(self, zero_shooter: TaxZeroShot,) -> None:
        self.zero_shooter = zero_shooter

    def select_examples(self,
                        data: Union[WebOfScience, DBPedia, AmazonHTC],
                        min_entropy: float,
                        max_entropy: float,
                        n_shots: int) -> List[ExampleFewShot]:
        logger.debug('Encoding training examples')
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
        sum_p = sum(p)
        p = [x / sum_p for x in p]
        return sum(p * math.log(p))

    def get_centroids_ids(self, texts: List[str], ids: List[int], n_shots: int):
        logger.debug('Computing centroids')
        ids = torch.arange(len(texts))
        # Filter doc embeddings within a certain range of entropy.
        embs = self.encoder.encode(texts)
        # Centroids method.
        kmeans = KMeans(n_clusters=n_shots)
        kmeans.fit(embs)
        centroids = np.array(kmeans.cluster_centers_)
        closest, _ = vq(centroids, embs)
        return ids[closest]


class SelectLabelled:
    def __init__(self) -> None:
        pass

    def forward(self):
        logger.info('Loading encoder')
        encoder = TextEncoder()

        # Encode text examples.
        logger.info('Encoding training examples')
        size = 64
        train = WebOfScience.trainingSet(topn=None)
        n_batches = len(train) // size + int(len(train) % size != 0)
        embeds = []
        for i in tqdm(range(n_batches)):
            batch = train[i * size: (i + 1) * size]
            with torch.no_grad():
                embeds.append(encoder([x.text for x in batch]))
        embeds = torch.vstack(embeds)

        # Encode unique labels.
        unique_labels = WebOfScience.get_unique_labels()
        with torch.no_grad():
            labels = encoder(unique_labels)

        # Compute examples entropy and sort according to it.
        logger.info('Computing Entropies')
        entropies = []
        for i, ex in enumerate(embeds):
            dot = torch.mm(ex.unsqueeze(0), labels.T).squeeze(0)
            p = torch.abs(dot) / sum(torch.abs(dot))
            H = -sum(p * torch.log(p)).item()
            entropies.append((H, i))
        res = sorted(entropies, key=lambda x: x[0], reverse=True)

        # Get N most entropic examples for each label.
        N = 1000
        logger.info(f'Getting top {N} highest entropy examples per label')
        groups = defaultdict(list)
        examples = {i: x for i, x in enumerate(train)}
        examples = [(examples[i], h) for h, i in res]
        for ex, h in examples:
            if len(groups[ex.label_level1]) == N:
                continue
            groups[ex.label_level1].append((h, ex.text, ex.label_level1, ex.label_level2))


        filepath = f'{Paths.WOS_DIR}/fewShots.json'
        logger.info(f'Saving to {filepath}')
        with open(filepath, 'w') as f:
            json.dump(groups, f)
