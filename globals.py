from os.path import dirname, abspath
from torch import cuda
import argparse


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--gpu_no",
        help='GPU to run model on',
        type=int,
        default=2,
        required=False
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help='batch size',
        type=int,
        default=64,
        required=False
    )
    parser.add_argument(
        "-t",
        "--topn",
        help='only consider top-n examples of the dataset',
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "-cp",
        "--compute_priors",
        help='compute from scratch labels priors for each document',
        type=bool,
        default=False,
        required=False
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        help='Select one of the available datasets [WebOfScience, DBPedia, AmazonHTC]',
        type=str,
        default='AmazonHTC',
        required=False
    )
    return parser.parse_args()


class Paths:
    MAIN_DIR = dirname(abspath(__file__))
    TBI_DIR = dirname(MAIN_DIR)
    SRC_DIR = f"{MAIN_DIR}/src"
    RESULTS_DIR = f"{MAIN_DIR}/results"
    SAVE_DIR = f"{MAIN_DIR}/saves"
    DATASETS_DIR = f"{TBI_DIR}/datasets"
    WOS_DIR = f'{DATASETS_DIR}/WebOfScience/Meta-data'
    DBP_DIR = f'{DATASETS_DIR}/DBPedia_Extract'
    AHTC_DIR = f'{DATASETS_DIR}/Amazon_Reviews'
    WIKI_DIR = f'{MAIN_DIR}/wiki_pages'
    JOANNE_DIR = f'{TBI_DIR}/data/manual_annotations'


class Globals:
    DEVICE = 'cuda:0' if cuda.is_available() else 'cpu'
