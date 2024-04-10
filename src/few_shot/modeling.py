import logging
from tqdm import tqdm
from typing import List
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import SGD
from sklearn.metrics import precision_recall_fscore_support
from src.zero_shooter import TaxZeroShot
from src.few_shot.select_FS_examples import ExampleFewShot
from globals import Globals

# Ignore sklearn warnings.
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger('few-shot_TC')


class FewShotTrainer(nn.Module):
    def __init__(self, labels_all: List[str], labels_train: List[str]) -> None:
        super(FewShotTrainer, self).__init__()
        self.criterion = nn.BCELoss()
        self.labels_all = labels_all  # ALl labels to consider (every tax layer but last).
        self.labels_train = labels_train  # Subset of labels seen during FS train.

    def train(self,
              tzs_model: TaxZeroShot,
              examples_train: List[ExampleFewShot],
              examples_valid: List[ExampleFewShot],
              lr: float,
              n_epochs: int):

        optimizer = SGD(tzs_model.zstc_encoder.encoder.model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            logger.info(f'Epoch {epoch+1}/{n_epochs}')
            docs = [example.text for example in examples_train]
            targets = [example.labels[0] for example in examples_train]
            loss_train = 0
            for doc, true_lab in tqdm(list(zip(docs, targets))):
                optimizer.zero_grad()
                posterior_scores_flat, _ = tzs_model.forward([doc])
                preds = torch.stack(
                    [posterior_scores_flat[0][lab] for lab in self.labels_all]
                ).to(Globals.DEVICE)
                true_lab = Tensor(
                    [1 if lab == true_lab else 0 for lab in self.labels_all]
                ).to(Globals.DEVICE)
                loss = self.criterion(preds, true_lab)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            logger.info('Loss train: %.3f' % loss_train)

            # Evaluate.
            self.evaluate(tzs_model, examples_valid)

    def evaluate(self,
                 tzs_model: TaxZeroShot,
                 data: List[ExampleFewShot]) -> List[str]:
        prec_seen, rec_seen, f1_seen, prec_unseen, rec_unseen, f1_unseen = \
            FewShotEvaluator.run(tzs_model, data, self.labels_all,
                                 self.labels_train)
        message = "prec: %.3f, rec: %.3f, f1: %.3f"
        logger.info(f'SEEN: {message % (prec_seen, rec_seen, f1_seen)}')
        logger.info(f'UNSEEN: {message % (prec_unseen, rec_unseen, f1_unseen)}')


class FewShotEvaluator:
    @staticmethod
    def run(tzs_model: TaxZeroShot,
            data: List[ExampleFewShot],
            labels_all: list[str], labels_train: list[str]) -> List[str]:
        texts = [example.text for example in data]
        targets = np.array([example.labels[0] for example in data])
        with torch.no_grad():
            posterior_scores_flat, _ = tzs_model.forward(texts)
        predictions = []
        for scores in posterior_scores_flat:
            scores_subset = {lab: scores[lab] for lab in labels_all}
            # Get the label with the highest score.
            predictions.append(max(scores_subset, key=scores.get))
        predictions = np.array(predictions)
        # Compute performance separately for labels seen at training time and not.
        ids_seen_during_training = np.array([t in labels_train for t in targets])
        # Labels seen during FS training.
        targs = targets[ids_seen_during_training]
        preds = predictions[ids_seen_during_training]
        prec_seen, rec_seen, f1_seen, _ = \
            precision_recall_fscore_support(targs, preds, average='macro')
        # Labels not seen during FS training.
        targs = targets[~ids_seen_during_training]
        preds = predictions[~ids_seen_during_training]
        prec_unseen, rec_unseen, f1_unseen, _ = \
            precision_recall_fscore_support(targs, preds, average='macro')
        return prec_seen, rec_seen, f1_seen, prec_unseen, rec_unseen, f1_unseen