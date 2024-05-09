from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from torch import Tensor
import torch.nn as nn
from sentence_transformers.util import cos_sim
from src.encoders import ZeroShooterZSTC
from src.dataset import BaseData
from src.hyper_inference import compute_labels_alpha
from src.scoring_functions import PriorScoresZeroShooting
from src.score_propagation import UpwardScorePropagation
from src.utils import FileIO, flatten_tree, build_prior_scores_trees
from globals import Globals, Paths


class TaxZeroShot(nn.Module):
    def __init__(self, taxonomy: Dict,
                 compute_label_thresholds: bool = False,
                 label_thresholds_file: str = None,
                 freeze_zstc: bool = True,
                 freeze_usp: bool = True) -> None:
        super(TaxZeroShot, self).__init__()
        self.zstc = ZeroShooterZSTC('sentence-transformers/all-mpnet-base-v2')
        self.taxonomy = taxonomy
        self.data = BaseData(taxonomy)
        self.prior_scores = PriorScoresZeroShooting(
            self.zstc, self.data.tax_tree, self.data.labels_flat
        )
        self.label2id = self.prior_scores.label2id
        if compute_label_thresholds:
            self.label2alpha = compute_labels_alpha(self.data.labels_flat,
                                                    Paths.WIKI_DIR, self.zstc)
            FileIO.write_json(self.label2alpha, label_thresholds_file)
        else:
            self.label2alpha = FileIO.read_json(label_thresholds_file)
        self.USP = UpwardScorePropagation(self.label2alpha, self.label2id)

        # Freeze parameters if no_grad.
        if freeze_zstc:
            for p in self.zstc.encoder.model.parameters():
                p.requires_grad = False
        if freeze_usp:
            self.USP.alphas.requires_grad = False

    def forward(self, documents: Union[List[str], Tensor],
                hide_prograssbar: bool = True) -> Tuple[List[Dict], List[Dict]]:
        # Compute docs embs.
        if isinstance(documents[0], str):
            priors_flat = self.prior_scores.compute_prior_scores(documents)
        # Use precomputed docs embs.
        elif isinstance(documents[0], Tensor):
            labels_embs = self.zstc.encode_labels(self.data.labels_flat)
            priors_flat = cos_sim(documents.to(Globals.DEVICE), labels_embs.to(Globals.DEVICE))
            priors_flat[priors_flat < 0] = 0
        # Distrib. labels priors on the tax tree.
        prior_trees = build_prior_scores_trees(priors_flat, self.taxonomy, self.label2id)
        # Compute posteriors with USP mechanism.
        res_flat, res_trees = [], []
        for prior in tqdm(prior_trees, disable=hide_prograssbar,
                          total=len(documents), mininterval=4):
            posterior_tree = self.USP.scaling_H(prior)
            posterior_flat = flatten_tree(posterior_tree)
            res_trees.append(posterior_tree)
            res_flat.append(posterior_flat)
        return res_flat, res_trees
