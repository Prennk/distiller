import torch
from torch import nn
from .memory import ContrastMemory

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
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
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

# class ContrastLoss(nn.Module):
#     """
#     Contrastive loss with hybrid regularization, corresponding to Eq (18)
#     """
#     def __init__(self, n_data, lambda_reg):
#         super(ContrastLoss, self).__init__()
#         self.n_data = n_data
#         self.lambda_reg = lambda_reg

#     def forward(self, x, features_s, features_t):
#         # Calculate batch size
#         bsz = x.shape[0]
#         # Calculate the number of negative samples
#         m = x.size(1) - 1

#         # Small epsilon value for numerical stability
#         eps = torch.finfo(torch.float32).eps

#         # Calculate noise distribution
#         Pn = 1 / float(self.n_data)

#         # Calculate loss for positive pair
#         P_pos = x.select(1, 0)
#         log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

#         # Calculate loss for K negative pairs
#         P_neg = x.narrow(1, 1, m)
#         log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

#         # Calculate total loss
#         loss_contrastive = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

#         # Regularization term
#         reg_term_s = torch.norm(features_s, p=2)
#         reg_term_t = torch.norm(features_t, p=2)
#         loss_regularization = self.lambda_reg * (reg_term_s + reg_term_t)

#         # Combine contrastive loss with regularization
#         total_loss = loss_contrastive + loss_regularization

#         return total_loss



# class Embed(nn.Module):
#     """Embedding module"""
#     def __init__(self, dim_in=1024, dim_out=128):
#         super(Embed, self).__init__()
#         self.linear = nn.Linear(dim_in, dim_out)
#         self.l2norm = Normalize(2)

#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.linear(x)
#         x = self.l2norm(x)
        return x

import torch.nn.functional as F
class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, num_layers=4, num_heads=8):
        super(Embed, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_layers1 = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads)
        self.attention_layers2 = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads)
        self.fc = nn.Linear(dim_out, dim_out)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(dim_out)
        self.norm2 = nn.LayerNorm(dim_out)
        
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.linear(x)
        x = self.l2norm(x)

        x = x.unsqueeze(0)

        residual_1 = x
        x, _ = self.attention_layers1(x, x, x)
        x = self.l2norm(x)
        x = self.gelu(x)
        x += residual_1

        residual_2 = x
        x, _ = self.attention_layers2(x, x, x)
        x = self.l2norm(x)
        x = self.gelu(x)
        x += residual_2

        x += residual_1

        x = x.squeeze(0)

        x = self.fc(x)

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