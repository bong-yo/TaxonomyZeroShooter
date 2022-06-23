'''
This scpript is to benchmarck the performance of simply encoding of Different text Encoders:

TARS
BART-MNLI
all-mpnet-base-v2
all-roberta-large-v1
paraphrase-mpnet-base-v2
multi-qa-mpnet-base-cos-v1
msmarco-bert-base-dot-v5
'''
from time import time
import torch
from sklearn.metrics import precision_recall_fscore_support
from src.dataset import WebOfScience, DBPedia, AmazonHTC
from src.encoders import ZeroshooterMPNet, ZeroshooterTARS, ZeroshooterBART, ZeroshooterBART_2
from globals import Paths, arguments, Globals


args = arguments()
if Globals.DEVICE == "cuda":
    torch.cuda.set_device(args.gpu_no)

DataSets = {'WebOfScience': WebOfScience, 'DBPedia': DBPedia, 'AmazonHTC': AmazonHTC}
datafolders = {'WebOfScience': Paths.WOS_DIR, 'DBPedia': Paths.DBP_DIR, 'AmazonHTC': Paths.AHTC_DIR}

models = [
    'all-mpnet-base-v2',
    'all-roberta-large-v1',
    'paraphrase-mpnet-base-v2',
    'multi-qa-mpnet-base-cos-v1',
    'msmarco-bert-base-dot-v5'
]

models = ['BART']
topn = 2000



for model_name in models:
    print(f'\n\n ------------------------- {model_name} --------------------------\n\n')
    if model_name == 'TARS':
        encoder = ZeroshooterTARS(batch_size=32)
    elif model_name == 'BART':
        encoder = ZeroshooterBART_2(batch_size=20)
    else:
        encoder = ZeroshooterMPNet(batch_size=64, model_name=model_name)

    # ----- WoS
    print('------- WoS ----------')
    start = time()
    docs = WebOfScience('test', topn)
    # Lev 1.
    preds = [x[0][0] for x in encoder.classify(docs.abstracts, list(docs.unique_labels_lev1))]
    trues = docs.Y1
    p, r, f1, s = precision_recall_fscore_support(preds, trues, average='macro', zero_division=0)
    print('level 1: ', f1, ',  time: ', time() - start)

    # ----- DBPedia
    print('------- DBPedia ----------')
    docs = DBPedia('test', topn)
    # Lev 1.
    start = time()
    preds = [x[0][0] for x in encoder.classify(docs.abstracts, list(docs.unique_labels_lev1))]
    trues = docs.Y1
    p, r, f1, s = precision_recall_fscore_support(preds, trues, average='macro', zero_division=0)
    print('level 1: ', f1, ',  time: ', time() - start)
    # Lev 2.
    start = time()
    preds = [x[0][0] for x in encoder.classify(docs.abstracts, list(docs.unique_labels_lev2))]
    trues = docs.Y2
    p, r, f1, s = precision_recall_fscore_support(preds, trues, average='macro', zero_division=0)
    print('level 2: ', f1, ',  time: ', time() - start)

    # ---- Amazon.
    print('------- Amazon ----------')
    docs = AmazonHTC('test', topn)
    # Lev 1.
    start = time()
    preds = [x[0][0] for x in encoder.classify(docs.abstracts, list(docs.unique_labels_lev1))]
    trues = docs.Y1
    p, r, f1, s = precision_recall_fscore_support(preds, trues, average='macro', zero_division=0)
    print('level 1: ', f1, ',  time: ', time() - start)
    # Lev 2.
    start = time()
    preds = [x[0][0] for x in encoder.classify(docs.abstracts, list(docs.unique_labels_lev2))]
    trues = docs.Y2
    p, r, f1, s = precision_recall_fscore_support(preds, trues, average='macro', zero_division=0)
    print('level 2: ', f1, ',  time: ', time() - start)
    print(0)
