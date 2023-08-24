#from https://github.com/TinyZeaMays/CircleLoss/blob/master/circle_loss.py 

#from typing import Tuple

import torch
from torch import nn, Tensor
import numpy as np


#def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
def convert_label_to_similarity(normed_feature, label):
    #similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)



    similarity_matrix = torch.matmul(normed_feature, normed_feature.transpose(1, 0))


    #N = normed_feature.size(0)
    #is_pos = label.expand(N, N).eq(labels.expand(N, N).t())
    #is_neg = label.expand(N, N).ne(labels.expand(N, N).t())


    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    label_matrix_1 = label.unsqueeze(1) != label.unsqueeze(0)


    positive_matrix = label_matrix.triu(diagonal=1)
    #negative_matrix = label_matrix.logical_not().triu(diagonal=1)
    negative_matrix = label_matrix_1.triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    #def __init__(self, m: float, gamma: float) -> None:
    def __init__(self, m, gamma):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    #def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
    def forward(self, sp, sn):
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


if __name__ == "__main__":
    feat = nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
    lbl = torch.randint(high=10, size=(256,))

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    criterion = CircleLoss(m=0.25, gamma=256)
    circle_loss = criterion(inp_sp, inp_sn)

    print(circle_loss)
