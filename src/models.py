from typing import Dict
import torch.nn as nn
from torch import Tensor


class SigmoidModel(nn.Module):
    def __init__(self, label_alpha: Dict[str, float], label2id: Dict[str, int]) -> None:
        super(SigmoidModel, self).__init__()
        self.label_alpha = label_alpha
        self.label2id = label2id
        id2label = {id: lab for lab, id in label2id.items()}
        alphas = [label_alpha[id2label[i]] for i in range(len(label_alpha))]
        self.a = nn.Parameter(Tensor(alphas), requires_grad=True)
        self.b = nn.Parameter(Tensor([10]), requires_grad=True)
        self.g = nn.Sigmoid()  # Sigmoid = 1 / (1 + exp(-x)).

    def forward(self, x: Tensor, idx: int):
        h = self.b * (x - self.a[idx])
        return self.g(h)[0]
