import logging
import argparse
from src.utils import FileIO
from src.encoders import ZeroShooterZSTC
from src.dataset import WebOfScience, AmazonHTC, DBPedia
from src.scoring_functions import PosteriorScoresPropagation
from src.metrics import PerformanceDisplay
from globals import Paths

logger = logging.getLogger('zeroshot-logger')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--gpu_no", help='GPU to run model on', type=int, default=0, required=False
    )
    parser.add_argument(
        "-t", "--topn", help='only take top-n docs', type=int, default=None, required=False
    )
    parser.add_argument(
        "-ds", "--dataset",
        help='Select one of the available datasets [WebOfScience, DBPedia, AmazonHTC]',
        type=str, default='WebOfScience', required=False
    )
    parser.add_argument(
        "-sf", "--savefile", help='file where to save performance results',
        type=str, default='USP-performance_branchwise.txt', required=False
    )
    args = parser.parse_args()

    savefile = f'{Paths.RESULTS_DIR}/{args.savefile}'
    DATASETS = {'WebOfScience': WebOfScience, 'DBPedia': DBPedia, 'AmazonHTC': AmazonHTC}
    MODEL_NAME = 'all-mpnet-base-v2'
    encoder = ZeroShooterZSTC(MODEL_NAME)

    for name, DataSet in DATASETS.items():
        msg = f"\n\n---------------------  {name} - {MODEL_NAME} --------------------\n"
        data = DataSet('test', topn=None)
        scorer = PosteriorScoresPropagation(data, encoder)

        # Compute Z-STC prior scores and propagate with USP.
        prior_scores_trees = scorer.compute_prior_trees()
        label2alpha = scorer.compute_labels_alpha()
        posterior_scores_trees = scorer.apply_USP(prior_scores_trees, label2alpha)
        
        # Select top labels for each level (according to posterior scores).
        preds = [scorer.get_levels_top_label(tree) for tree in posterior_scores_trees]
        preds_level = zip(*preds)

        # Compute and display performance.
        perf_displayer = PerformanceDisplay(data.Y, preds_level)
        performance = perf_displayer.compute_levels_performance()
        msg += performance
        print(msg)
        # FileIO.append_text(msg, savefile)