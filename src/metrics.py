import logging
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

logger = logging.getLogger('zeroshot-logger')



class ResultsBase:
    def __init__(self, levels_performance: List[Tuple[float]]) -> None:
        self.prec_macro, self.rec_macro, self.f1_macro= [], [], []
        for prec_lev, rec_lev, f1_lev, support_lev in levels_performance:
            self.prec_macro.append(prec_lev)
            self.rec_macro.append(rec_lev)
            self.f1_macro.append(f1_lev)

class Results:
    def __init__(self,
                 levels_accuracy: List[Dict],
                 levels_f1: List[Dict]) -> None:
        assert(len(levels_accuracy) == len(levels_f1)), print("Something Wrong with results levels")
        self.n_levels = len(levels_accuracy)
        self.accuracy, self.f1, self.acc_macro, self.f1_macro= [], [], [], []
        for accuracy, f1 in zip(levels_accuracy, levels_f1):
            # Ignore classes which we have no data for.
            accuracy = {k: v for k, v in accuracy.items() if v != 'no values'}
            f1 = {k: v for k, v in f1.items() if v != 'no values'}
            self.accuracy.append(accuracy)
            self.f1.append(f1)
            self.acc_macro.append(sum(accuracy.values()) / len(accuracy))
            self.f1_macro.append(sum(f1.values()) / len(f1))

class PerformanceDisplay:
    def __init__(self,
                 true_labels_levels: List[List[str]],
                 pred_labels_levels: List[List[str]]) -> None:
        self.trues_levels = true_labels_levels  # Each list is 
        self.preds_levels = pred_labels_levels
        n_levels = len(self.trues_levels)
        self.msg = "level | Prec  | Rec   | F1    |" + \
                   "\n    %d | %.3f | %.3f | %.3f |" * n_levels

    def compute_levels_performance(self):
        res = []
        for level, (trues, preds) in enumerate(zip(self.trues_levels, self.preds_levels)):
            p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='macro', zero_division=0)
            res.extend([level, p, r, f1])
        return self.msg % (tuple(res))
        