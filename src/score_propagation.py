from typing import Dict
import numpy as np


class UpwardScorePropagation:
    def __init__(self, label2alpha: Dict[str, float]) -> None:
        self.label2alpha = label2alpha  # Scores below this threshold do not contibute to USP.

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
                elif  child_posterior > beta:
                    highs += child_posterior
            mids = 0 if not mids else sum(mids)/len(mids)
            ups_score = mids + highs
            return ups_score
        dfs(prob_tree)
        return prob_tree

    def scaling_H(self, prob_tree: Dict):
        '''Perform the Upward Propagation System (UPS)
        by SCALING UP the score of the node x according to the difference of similarity
        with each children y: S(x) = S(x) * exp( min(0, sim_y - sim_x) )'''
        def _usp(node, is_root: bool = False):
            score = 0 if is_root else node.pop('prob')
            for label, child in node.items():
                child_score = _usp(child, is_root=False)
                if child_score >= self.label2alpha[label] and child_score > score:
                    score = child_score
                else:
                    delta = max(0, child_score - abs(score))
                    score *= np.exp(delta)
                    score = min(1, score)
            if not is_root:
                node['prob'] = score
            return score
        _usp(prob_tree, is_root=True)
        return prob_tree
