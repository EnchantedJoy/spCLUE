import torch
from torch import nn
import math
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, x, xbar, eps=1e-8):
        posScores = torch.exp((x * xbar).sum(dim=1) / self.temperature)
        negScores = torch.exp((x @ xbar.T) / self.temperature).sum(dim=1)
        return -torch.log(posScores / (negScores + eps)).mean()

class ClusterLoss(nn.Module):
    def __init__(
            self,
            n_classes,
            device=torch.device("cuda:0"),
            temperature=0.2,
    ):
        super(ClusterLoss, self).__init__()
        self.n_classes = n_classes
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(n_classes)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, n_classes):
        N = 2 * n_classes
        mask = torch.ones(N, N)
        mask.fill_diagonal_(0)
        for i in range(n_classes):
            mask[i, i + n_classes] = 0
            mask[i + n_classes, i] = 0
        mask = mask.bool()
        return mask

    def normalizeLabel(self, c_i, c_j):
        c_i = torch.square(c_i)
        c_j = torch.square(c_j)
        p_i = c_i.sum(dim=0).view(-1)
        c_i /= p_i
        p_i = c_i.sum(dim=1).view(-1)
        c_i /= p_i.unsqueeze(1)
        p_j = c_j.sum(dim=0).view(-1)
        c_j /= p_j
        p_j = c_j.sum(dim=1).view(-1)
        c_j /= p_j.unsqueeze(1)
        return c_i, c_j

    def forward(self, c_i, c_j):
        p_i = c_i.sum(dim=0).view(-1)
        p_i /= p_i.sum()
        neg_entropy_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        neg_entropy_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        neg_entropy_loss = neg_entropy_i + neg_entropy_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.n_classes
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1),
                                c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.n_classes)
        sim_j_i = torch.diag(sim, -self.n_classes)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=-1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + 1. * neg_entropy_loss

class GraphConsis(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, emb, graphWeight):
        dist1 = torch.cdist(emb, emb, p=2)
        dist1 = torch.div(dist1, torch.max(dist1))
        return torch.mean((1 - dist1) * graphWeight)


class GraphRecLoss(nn.Module):
    def __init__(self, norm_val, pos_weight) -> None:
        super().__init__()
        self.norm_val = norm_val
        self.pos_weight = pos_weight

    def forward(self, emb, target):
        # emb = F.normalize(emb, p=2, dim=1)
        input = emb @ emb.T
        logits = F.binary_cross_entropy_with_logits(input,
                                                    target,
                                                    pos_weight=self.pos_weight)
        return self.norm_val * logits


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, xbar):
        return torch.square(x - xbar).mean(dim=1).mean()


class ZINBLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, mean, disp, pi=0, scale_factor=1.0, ridge_lambda=0.0):
        '''
        args: x, raw count, [N, hvgs]
              scale_factor, [n,]
        '''
        eps = 1e-10
        mean = (mean.T * scale_factor).T
        if pi == 0:
            pi = torch.tensor(0.0)

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(
            x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
            x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result
