"""
Here we benchmark the Entropy based Sentence Selection (ESS).
It is a way to automatically select the most informative sentences out of a long text
so to reduce noise in the final embeddings.
"""
from sklearn.metrics import precision_recall_fscore_support
import argparse
import logging
import numpy as np
import torch
from src.dataset import WebOfScience, DBPedia, AmazonHTC
from src.encoders import ZeroShooterZSTC
from globals import Globals

logger = logging.getLogger('zeroshot-logger')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--gpu_no", help='GPU to run model on', type=int, default=0, required=False
    )
    parser.add_argument(
        "-l", "--level", help='Taxonomy level to consider', type=int, default=0, required=False
    )
    parser.add_argument(
        "-t", "--topn", help='only take top-n docs', type=int, default=None, required=False
    )
    parser.add_argument(
        "-ds", "--dataset",
        help='Select one of the available datasets [WebOfScience, DBPedia, AmazonHTC]',
        type=str, default='WebOfScience', required=False
    )
    args = parser.parse_args()
    DATASETS = {'WebOfScience': WebOfScience, 'DBPedia': DBPedia, 'AmazonHTC': AmazonHTC}

    if Globals.DEVICE == "cuda":
        torch.cuda.set_device(args.gpu_no)

    data = DATASETS[args.dataset]('test', args.topn)
    zste_model = ZeroShooterZSTC('all-mpnet-base-v2')

    # Encode labels of the n-th level.
    zste_model.encode_labels(data.labels_levels[args.level])
    trues = data.Y[args.level]

    # Baseline (naive doc embedding).
    cos_scores = zste_model.compute_labels_scores(data.abstracts, encoding_method='base')
    preds = [zste_model.id2label[i] for i in np.argmax(cos_scores, axis=-1)]
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='micro')
    logger.info(f"Baseline:  Prec: {p},  Rec: {r},  F1: {f1}")

    # Entropy based Sentence Selection (ESS).
    cos_scores = zste_model.compute_labels_scores(data.abstracts, encoding_method='ess')
    preds = [zste_model.id2label[i] for i in np.argmax(cos_scores, axis=-1)]
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='micro')
    logger.info(f"ESS:  Prec: {p},  Rec: {r},  F1: {f1}")
