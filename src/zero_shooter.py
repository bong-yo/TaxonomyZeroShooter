from typing import Dict, List, Tuple, Union
from torch import Tensor
from sentence_transformers.util import cos_sim
from src.encoders import ZeroShooterZSTC
from src.dataset import BaseData
from src.scoring_functions import PriorScoresZeroShooting
from src.score_propagation import UpwardScorePropagation
from src.utils import FileIO, flatten_tree


class TaxZeroShot:
    def __init__(self, taxonomy: Dict, label_thresholds_file: str = None,
                 no_grad_zstc: bool = True,
                 no_grad_usp: bool = True) -> None:
        self.encoder = ZeroShooterZSTC('sentence-transformers/all-mpnet-base-v2')
        self.data = BaseData(taxonomy)
        self.prior_scores = PriorScoresZeroShooting(
            self.zstc_encoder, self.data.tax_tree, self.data.labels_flat
        )
        self.label2id = self.prior_scores.label2id
        if label_thresholds_file is not None:
            self.label2alpha = FileIO.read_json(label_thresholds_file)
        else:
            self.label2alpha = self.prior_scores
        self.USP = UpwardScorePropagation(self.label2alpha, self.label2id)

        # Freeze parameters if no_grad.
        if no_grad_zstc:
            for p in self.encoder.encoder.model.parameters():
                p.requires_grad = False
        if no_grad_usp:
            for p in self.USP.sigmoid_gate_model.parameters():
                p.requires_grad = False

    def forward(self, documents: Union[List[str], Tensor]) -> Tuple[List[Dict], List[Dict]]:
        if isinstance(documents[0], str):  # Compute docs embs.
            priors_flat = self.prior_scores.compute_prior_scores(documents)
        elif isinstance(documents[0], Tensor):  # Use precomputed docs embs.
            labels_embs = self.zstc_encoder.encode_labels(self.data.labels_flat)
            priors_flat = cos_sim(documents, labels_embs)
            priors_flat[priors_flat < 0] = 0
        res_flat, res_trees = [], []
        for prior_scores_flat in priors_flat:
            posterior_tree = self.USP.gate_H(prior_scores_flat,
                                             self.data.tax_tree)
            posterior_flat = flatten_tree(posterior_tree)
            res_trees.append(posterior_tree)
            res_flat.append(posterior_flat)
        return res_flat, res_trees
