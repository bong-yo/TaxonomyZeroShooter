from typing import List
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.optim import SGD
from src.zero_shooter import TaxZeroShot


class Trainer(nn.Module):
    def __init__(self) -> None:
        super(Trainer, self).__init__()
        self.criterion = nn.BCELoss()

    def train(self,
              tzs_model: TaxZeroShot,
              docs: List[str],
              targets: List[str],
              lr: float, n_epochs: int):

        optimizer = SGD(tzs_model.UPS.sigmoid_gate_model.parameters(), lr=lr)

        for epoch in range(n_epochs):

            for doc, doc_labels in zip(docs, targets):
                optimizer.zero_grad()

                posterior_scores_flat = tzs_model.forward([doc])[0]

                preds, targs = [], []
                doc_labels = set(doc_labels)
                for label, score in posterior_scores_flat.items():
                    preds.append(score)
                    targs.append(1 if label in doc_labels else 0)

                preds = torch.vstack(preds).view(-1)
                targs = Tensor(targs)
                loss = self.criterion(preds, targs)

                loss.backward()
                optimizer.step()
