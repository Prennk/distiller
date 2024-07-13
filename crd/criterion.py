import torch
from torch import nn
from .memory import ContrastMemory,\
    ContrastMemoryWithTopkSampling, ContrastMemoryWithHardNegative,\
    ContrastMemoryCC

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.distill, opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.distill, opt.t_dim, opt.feat_dim)
        if opt.distill == 'crd':
            self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        elif opt.distill == 'crd_topk':
            self.contrast = ContrastMemoryWithTopkSampling(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        elif opt.distill == 'crd_hardneg':
            self.contrast = ContrastMemoryWithHardNegative(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        elif opt.distill == 'crd_cc':
            self.embed_cluster_s = Embed_2(opt.nce_k, opt.s_dim)
            self.embed_cluster_t = Embed_2(opt.nce_k, opt.t_dim)
            self.contrast = ContrastMemoryCC(100, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
            self.criterion_cluster_t = ContrastLoss(opt.n_data)
            self.criterion_cluster_s = ContrastLoss(opt.n_data)
        else:
            raise KeyError('Invalid CRD variant')
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)
        self.distill = opt.distill

    def forward(self, x_s, x_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(x_s)
        f_t = self.embed_t(x_t)
        y_s = self.embed_cluster_s(x_s)
        y_t = self.embed_cluster_t(x_t)
        if self.distill == 'crd_cc':
            out_s, out_t, out_cluster_s, out_cluster_t = self.contrast(f_s, f_t, y_s, y_t, idx, contrast_idx)
            s_loss = self.criterion_s(out_s)
            t_loss = self.criterion_t(out_t)
            s_cluster_loss = self.criterion_cluster_s(out_cluster_s)
            t_cluster_loss = self.criterion_cluster_s(out_cluster_t)

            # loss = loss = ((s_loss + t_loss) * 0.5) + ((s_cluster_loss + t_cluster_loss) * 0.5)
            loss = s_cluster_loss + t_cluster_loss
        else:
            out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
            s_loss = self.criterion_s(out_s)
            t_loss = self.criterion_t(out_t)

            loss = s_loss + t_loss
        

        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, distill, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)
        self.linear_class = nn.Linear(dim_in, 100)
        self.distill = distill

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        if self.distill == 'crd_cc':
            x = self.linear_class(x)
        else:
            x = self.linear(x)

        x = self.l2norm(x)
        return x
    
class Embed_2(nn.Module):
    """Embedding module"""
    def __init__(self, K, dim_in=1024):
        super(Embed_2, self).__init__()
        self.linear = nn.Linear(dim_in, K + 1)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out