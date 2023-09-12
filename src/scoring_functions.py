import logging
from glob import glob
from typing import Iterable, List, Union, Dict
import numpy as np
from copy import deepcopy
from collections import deque
from src.hyper_inference import VarianceEstimator
from src.encoders import ZeroShooterZSTC, ZeroshooterBART_2, ZeroshooterTARS
from src.dataset import WebOfScience, AmazonHTC, DBPedia
from src.score_propagation import UpwardScorePropagation
from globals import Paths

logger = logging.getLogger("zeroshot-logger")


class PriorScoresZeroShooting:
    """Use a Text Encoder to ZeroShoot all labels prior probabilities on each document.
    At this point, prior probs. are purely based on labels and docs semantics"""
    def __init__(self,
                 zero_shooter: Union[ZeroShooterZSTC, ZeroshooterBART_2, ZeroshooterTARS],
                 tax_tree: Dict,
                 labels_flat: List[str]) -> None:
        self.zero_shooter = zero_shooter
        self.tax_tree = tax_tree
        self.labels_flat = labels_flat
        self.label2id = {l: i for i, l in enumerate(labels_flat)}

    def compute_prior_scores(self, texts: List[str]) -> np.array:
        """Compute the matrix of similarity of each document to each label"""
        logger.debug("computing Z-STC")
        return self.zero_shooter.compute_labels_scores(texts, self.labels_flat)

    def build_prior_scores_trees(self, simil_matrix: np.array) -> Iterable[Dict]:
        def _fill_scores_dfs(root: Dict) -> None:
            """Inplace modify a copy of taxonomy tree adding prior score for each label."""
            nonlocal simil_matrix, doc_num
            for label in root:
                child = root[label]
                lab_id = self.label2id[label]
                _fill_scores_dfs(child)
                child['prob'] = simil_matrix[doc_num][lab_id]

        for doc_num in range(simil_matrix.shape[0]):
            probs_tree = deepcopy(self.tax_tree)
            _fill_scores_dfs(probs_tree)
            yield probs_tree

    def ZS_best_labels(self, texts: List[str], levels_labels: List[List[str]]):
        """For each level of the taxonomy, chose the label with higher score.
        Parameters
        ----------
        texts: List[str] -  List of texts for ZSTE.
        levels_labels: List[List[str]]  -  list of labels of every level of the Taxonomy Tree.

        Returns
        -------
        res: List[List[str]]  -  List of best labels for every document:
                                 res[0] will have the best label of lev 0 for every document.
                                 res[1]    "   "    "   "           lev 1   "  " 
        """
        docs_scores = self.compute_prior_scores(texts)
        levels_labels_ids = [[self.label2id[lab] for lab in lev] for lev in levels_labels]

        return [
            [
                levels_labels[i][best_label_id]  # Convert best label-id into label name.
                for best_label_id in docs_scores[:, level_ids].argmax(-1)  # Get best i-th level label of every doc.
            ]
            for i, level_ids in enumerate(levels_labels_ids)  # Loop through each taxonomy level.
        ]


class PosteriorScoresPropagation:
    """
    Class to apply the overall process of taxonomy scoring
    - compute PRIOR Z-STC scores
    - compute relevance thresholds alpha (one for each label)
    - get POSTERIOR scores by applying Upwards Score Propagation (USP)
    - select top scoring label for each level
    """
    def __init__(self,
                 data: Union[WebOfScience, AmazonHTC, DBPedia],
                 encoder: ZeroShooterZSTC,
                 label2alpha: Dict[str, float]) -> None:
        self.data = data
        self.encoder = encoder
        self.USP = UpwardScorePropagation(label2alpha)

    def compute_prior_trees(self) -> Iterable[Dict]:
        """Use Zero-Shot Semantic Text Classification (Z-STC) to assign a prior score for each
        label purely based on the semantics of tghe label and of the documents.

        Return
        ------
        prior_trees: Iterable[Dict]  -  An iterable of the taxonomy tree for each doc,
                                        where each label in the tree has associated the 
                                        prior score computed with Z-STC.
        """
        logger.info('computing Prior Relevance Scores')
        prior_scores = PriorScoresZeroShooting(
            self.encoder, self.data.tax_tree, self.data.labels_flat
        )
        prior_flat = prior_scores.compute_prior_scores(self.data.abstracts)
        prior_trees = prior_scores.build_prior_scores_trees(prior_flat)
        return prior_trees, prior_flat, prior_scores.label2id

    def compute_labels_alpha(self) -> Dict[str, float]:
        """Compute Relevance Threshold alpha for each label.
        For each label:
        - Compute distribution of Z-STC scores on 1000 ground Wikipedia articles 
            (randomly selected therefore unrelated to the label)
        - Gumbel distribution mean and sigma are fitted on Z-STC scores
        - Compute alpha(label) = mean(label) + 3 sigma(label)

        Return
        ------
        label2alpha: Dict[str, float]  -  Dictionary of label: alpha(label)
        """
        logger.info('computing Relevance Thresholds alphas')
        variance_estimator = VarianceEstimator(glob(f'{Paths.WIKI_DIR}/*'), self.encoder)
        label2alpha = \
            variance_estimator.estimate_lognormal(self.data.labels_flat, thresh_perc=0.99)
        return label2alpha

    def apply_USP(self,
                  prior_scores_flat: Iterable[Dict],
                  label2id: Dict[str, int]) -> List:
        """
        Compute posterior scores trees by apply Upwards Score Propagation (USP).
        """
        logger.info('applying USP')
        # For each document prior-tree apply USP to get posterior scores.
        for prior_scores in prior_scores_flat:
            yield self.USP.gate_H(prior_scores, label2id, self.data.tax_tree)

    @staticmethod
    def get_levels_top_label(tree: Dict) -> List[str]:
        """BFS of the taxonomy tree of scores relative to one document,
        and for each level get top label.
        Note: With this algo it might be that the top label of level N is not a child of
        top label of level N-1.

        Parameters
        ----------
        tree: Dict  -  Taxonomy tree with scores relative to one document.

        Returns
        -------
        top_labels_levels: List[str]  -  List of top label for each level.
        """
        labels_scores_level, top_labels_levels = [], []
        queue = deque([(k, v, 0) for k, v in tree.items()])
        while queue:
            node, children, level = queue.popleft()
            if level == len(labels_scores_level) or not queue:  # Catch last level with 'not queue'.
                if len(labels_scores_level) != 0:
                    # Get last-level label with higher score.
                    top_label = max(labels_scores_level[-1], key=lambda x: x[1])[0]
                    top_labels_levels.append(top_label)
                labels_scores_level.append([])
            score = children.pop('prob')
            labels_scores_level[-1].append((node, score))
            for k, child in children.items():
                queue.append((k, child, level + 1))
        return top_labels_levels

    def get_branches_top_label(self, tree: Dict) -> List[str]:
        """DFS of the taxonomy tree of scores relative to one document,
        every time, only the branch relative to the top label at level N is descended,
        in this way the top label at level N + 1 will be children of top label lev N

        Parameters
        ----------
        tree: Dict  -  Taxonomy tree with scores relative to one document.

        Returns
        -------
        top_labels_branches: List[str]  -  List of top label for each level.
        """
        def _dfs(root):
            nonlocal top_labels_branches
            if not root:
                return
            top_score = -1000000
            top_label = ''
            for label in root:
                score = root[label].pop('prob')
                if score > top_score:
                    top_score = score
                    top_label = label
            top_labels_branches.append(top_label)
            _dfs(root[top_label])
        top_labels_branches = []
        _dfs(tree)
        return top_labels_branches