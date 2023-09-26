from typing import Dict, List, Tuple
from tqdm import tqdm
from src.encoders import ZeroShooterZSTC
from src.dataset import BaseData
from src.scoring_functions import PriorScoresZeroShooting
from src.score_propagation import UpwardScorePropagation
from src.utils import FileIO, flatten_tree


class TaxZeroShot:
    def __init__(self, taxonomy: Dict, label_thresholds_file: str = None) -> None:
        self.encoder = ZeroShooterZSTC('all-mpnet-base-v2')
        self.data = BaseData(taxonomy)
        self.prior_scores = PriorScoresZeroShooting(
            self.encoder, self.data.tax_tree, self.data.labels_flat
        )
        self.label2id = self.prior_scores.label2id
        if label_thresholds_file is not None:
            self.label2alpha = FileIO.read_json(label_thresholds_file)
        else:
            self.label2alpha = self.prior_scores
        self.USP = UpwardScorePropagation(self.label2alpha, self.label2id)

    def forward(self, documents: List[str], no_grad: bool = True
                ) -> Tuple[List[Dict], List[Dict]]:
        priors_flat = self.prior_scores.compute_prior_scores(documents)
        res_flat, res_trees = [], []
        for prior_scores_flat in priors_flat:
            posterior_tree = self.USP.gate_H(prior_scores_flat, self.data.tax_tree, no_grad)
            posterior_flat = flatten_tree(posterior_tree)
            res_trees.append(posterior_tree)
            res_flat.append(posterior_flat)
        return res_flat, res_trees
