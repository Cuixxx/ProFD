"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
        self.soft_plus = nn.Softplus()
    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        # for numerical stability stable softmax
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()
        # exp_logits = torch.exp(logits)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss = - mean_log_prob_pos.mean()
        loss = 0
        for i, item in enumerate(logits):
            sp = item[mask[i]==1]
            sn = item[mask[i]==0]
            logit_p = - sp
            logit_n = sn
            loss = loss + self.soft_plus(torch.logsumexp(logit_n, dim=0)+torch.logsumexp(logit_p, dim=0))

        return loss/logits.shape[0]