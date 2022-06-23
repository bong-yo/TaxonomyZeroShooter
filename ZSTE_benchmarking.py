"""
This script runs Z-STE (Zero-Shot Semantic Text Embedding).
Each label is assigned a score purely based on semantics, i.e. both labels and docs
are encoded with several text encoders, and a similarity score is assigned based on these
embeddings.
The structure of tazonomy is not taken into account yet for Z-STE.
"""
import argparse
import logging
import torch
import src
from src.metrics import PerformanceDisplay
from src.dataset import WebOfScience, DBPedia, AmazonHTC
from src.scoring_functions import PriorScoresZeroShooting
from src.encoders import ZeroShooterZSTE
from src.utils import FileIO
from globals import Globals, Paths

logger = logging.getLogger('zeroshot-logger')


if __name__ == '__main__':
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
        type=str, default='ZSTE-performance_2.txt', required=False
    )
    args = parser.parse_args()

    if Globals.DEVICE == "cuda":
        torch.cuda.set_device(args.gpu_no)

    savefile = f'{Paths.RESULTS_DIR}/{args.savefile}'
    DATASETS = {'WebOfScience': WebOfScience, 'DBPedia': DBPedia, 'AmazonHTC': AmazonHTC}
    DATAFOLDERS = {'WebOfScience': Paths.WOS_DIR, 'DBPedia': Paths.DBP_DIR, 'AmazonHTC': Paths.AHTC_DIR}
    ZSTE_MODELS = [
        'all-mpnet-base-v2',
        'all-roberta-large-v1',
        'paraphrase-mpnet-base-v2',
        'multi-qa-mpnet-base-cos-v1',
        'msmarco-bert-base-dot-v5'
    ]

    # Loop through all benchmarking datasets.
    for name, DataSet in DATASETS.items():
        msg = f"\n\n---------------------  {name}  --------------------"
        logger.info(msg)
        FileIO.append_text(msg, savefile)

        # Load texts and labels from dataset.
        data_test = DataSet('test', topn=args.topn)

        # Loop through all ZSTE models.
        for model_name in ZSTE_MODELS:
            msg = f"ZSTE model:  {model_name}"
            logger.info(msg)
            FileIO.append_text("\n" + msg, savefile)

            # Encode docs and labels & compute ZSTE scores.
            zste_model = ZeroShooterZSTE(model_name)
            scores_zs = PriorScoresZeroShooting(zste_model, data_test.tax_tree, data_test.labels_flat)

            # Measure performance of given model on given dataset.
            res = scores_zs.ZS_best_labels(data_test.abstracts, data_test.labels_levels)
            perf_displayer = PerformanceDisplay(data_test.Y, res)
            performance = perf_displayer.compute_levels_performance()
            logger.info(performance)
            FileIO.append_text(performance, savefile)

