from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from src.models import SigmoidModel
from globals import Globals


class UpwardScorePropagation:
    def __init__(self, label2alpha: Dict[str, float], label2id: Dict[str, int]) -> None:
        self.label2alpha = label2alpha  # Scores below this threshold do not contibute to USP.
        self.label2id = label2id
        self.id2label = {id: lab for lab, id in label2id.items()}
        alphas = [self.label2alpha[self.id2label[i]] for i in range(len(self.label2alpha))]
        self.alphas = nn.Parameter(Tensor(alphas)).to(Globals.DEVICE)
        self.sigmoid_gate_model = SigmoidModel(label2alpha, label2id).to(Globals.DEVICE)

    def additive_H(self, prob_tree, alpha, beta):
        '''Performs the Upward Propagation System (UPS)
        by adding the scores of the children of x: S(x) = p(x) + H(S(y1), S(y2), ..S(yN))'''
        def dfs(node):
            nonlocal alpha, beta
            mids, highs = [], 0
            for k, child in node.items():
                child_prior = child.pop('prob')
                child_ups_score = dfs(child)
                child_posterior = min(1, child_prior + child_ups_score)
                child['prob'] = child_posterior
                if child_posterior <= alpha:
                    continue
                elif alpha < child_posterior <= beta:
                    mids.append(child_posterior)
                elif child_posterior > beta:
                    highs += child_posterior
            mids = 0 if not mids else sum(mids) / len(mids)
            ups_score = mids + highs
            return ups_score
        dfs(prob_tree)
        return prob_tree

    def scaling_H_old(self, prob_tree: Dict) -> Dict:
        '''Perform the Upward Score Propagation (USP):
        by SCALING UP the score of the node x according to the difference of similarity
        with each children y: S(x) = S(x) * exp( min(0, sim_y - sim_x) )'''
        def _usp(node, is_root: bool = False):
            score = 0 if is_root else node.pop('prob')
            score = abs(score)  # Make sure score is positive.
            for label, child in node.items():
                child_score = _usp(child, is_root=False)
                if child_score >= self.label2alpha[label] and child_score > score:
                    score = child_score
                else:
                    delta = max(0, child_score - score)
                    score *= np.exp(delta)
                    score = min(1, score)
            if not is_root:
                node['prob'] = score
            return score
        _usp(prob_tree, is_root=True)
        return prob_tree

    def scaling_H(self, prob_tree: Dict) -> Dict:
        '''Perform the Upward Score Propagation (USP):
        by SCALING UP the score of the node x according to the difference of similarity
        with each children y: S(x) = S(x) * exp( min(0, sim_y - sim_x) )'''
        def _usp(node, is_root: bool = False):
            score = 0 if is_root else node.pop('prob')
            score = abs(score)  # Make sure score is positive.
            for label, child in node.items():
                child_post_score = _usp(child, is_root=False)
                alpha = self.alphas[self.label2id[label]]
                if child_post_score >= alpha and child_post_score > score:
                    score = child_post_score
                else:
                    delta = torch.max(torch.tensor(0), (child_post_score - score) / alpha)
                    score = score * torch.exp(delta)
                    score = torch.min(torch.tensor(1), score)
            if not is_root:
                node['prob'] = score
            return score
        _usp(prob_tree, is_root=True)
        return prob_tree

    def gate_H(self, prior_scores_flat: List[float], tax_tree: Dict) -> Dict:
        '''Perform the Upward Score Propagation (USP):
        Relevance Threshold alpha acts like a GATE, i.e., score is propagated
        from children to parent ONLY IF children score is > alpha.
        Children scores are summed and scaled by tanh.

        Parameters
        ----------
        :prior_scores_flat List[float]: List of ZSTC scores for each label (labels
                                        position in the list are saved in self.label2id);
        :tax_tree Dict: Taxonomy tree (dict of dicts).
        :grad bool: if True compute gradients of the 'sigmoid_gate_model', else no_grad().

        Return
        ------
        :posterior_tree Dict: Taxonomy tree with posterior scores for each label.
        '''
        def _usp(root):
            nonlocal prior_scores_flat
            upwards_propag_score = 0
            for label, children in root.items():
                prior_score = prior_scores_flat[self.label2id[label]]
                children_score = _usp(children)
                posterior_score = torch.tanh(prior_score + children_score)
                propag_coeff = self.sigmoid_gate_model(posterior_score,
                                                       self.label2id[label])
                upwards_propag_score += propag_coeff * posterior_score
                root[label]['prob'] = posterior_score
            return upwards_propag_score

        prior_scores_flat = Tensor(prior_scores_flat)

        posterior_tree = deepcopy(tax_tree)
        _usp(posterior_tree)
        return posterior_tree
