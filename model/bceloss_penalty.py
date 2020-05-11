import torch
import torch.nn as nn


class BCEPLoss(torch.nn.Module):

    def __init__(self, device, C=0.005):
        super(BCEPLoss, self).__init__()
        self.bceloss = nn.BCELoss()
        self.device = device
        self.C = C

    def forward(self, input, target, A):
        A_inner_product = torch.bmm(A, A.permute(0, 2, 1))
        unit_matrix = torch.eye(A_inner_product.size(1), A_inner_product.size(2)).to(self.device)
        M = A_inner_product - unit_matrix
        P = (torch.sum(torch.sum(torch.sum(M ** 2, 1), 1) ** 0.5)) / input.size(0)
        return self.bceloss(input, target) + self.C * P
