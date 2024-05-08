import logging
import argparse
import torch
from src.utils import FileIO
from src.encoders import ZeroShooterZSTC
from src.zero_shooter import TaxZeroShot
from src.dataset import WebOfScience, AmazonHTC, DBPedia
from src.metrics import PerformanceDisplay
from src.hyper_inference import compute_labels_alpha
from src.utils import get_taxonomy_levels_top_label
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
    MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
    COMPUTE_ALPHAS = False
    encoder = ZeroShooterZSTC(MODEL_NAME)

    for name, DataSet in DATASETS.items():
        msg = f"\n\n---------------------  {name} - {MODEL_NAME} --------------------\n"
        data = DataSet('test', topn=10)

        # Compute label alphas.
        label_alphas_filename = f'{Paths.SAVE_DIR}/label_alphas_{name}.json'
        if COMPUTE_ALPHAS:
            logger.info('Computing label alphas.')
            label2alpha = compute_labels_alpha(data.labels_flat, f'{Paths.WIKI_DIR}', encoder)
            FileIO.write_json(label2alpha, label_alphas_filename)

        # Compute posterior scores for each doc.
        tax_zero_shooter = TaxZeroShot(taxonomy=data.tax_tree,
                                       compute_label_thresholds=False,
                                       label_thresholds_file=label_alphas_filename)
        with torch.no_grad():
            posterior_scores_flat, posterior_scores_trees = \
                tax_zero_shooter.forward(data.abstracts)

        # Select top labels for each level (according to posterior scores).
        preds = [get_taxonomy_levels_top_label(tree) for tree in posterior_scores_trees]
        preds_level = zip(*preds)

        # Compute and display performance.
        perf_displayer = PerformanceDisplay(data.Y, preds_level)
        performance = perf_displayer.compute_levels_performance()
        msg += performance
        print(msg)
        # FileIO.append_text(msg, savefile)
