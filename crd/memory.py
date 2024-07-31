import torch
from torch import nn
import torch.nn.functional as F
import math


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, rep_dim, n_data, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = n_data
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(rep_dim / 3)
        self.register_buffer('memory_v1', torch.rand(n_data, rep_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(n_data, rep_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        n_data = self.memory_v1.size(0)
        rep_dim = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, rep_dim)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, rep_dim, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, rep_dim)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, rep_dim, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * n_data
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * n_data
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj
    
#-----------------------------------------------------------------------------#
#                                                                             #
#-----------------------------------------------------------------------------#

class ContrastMemoryModified(nn.Module):
    """
    Memory buffer that supplies a large amount of negative samples.
    """
    def __init__(self, rep_dim, n_data, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = n_data
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(rep_dim / 3)
        self.memory_v1 = nn.Parameter(torch.randn(n_data, rep_dim).mul_(2 * stdv).add_(stdv))
        self.memory_v2 = nn.Parameter(torch.randn(n_data, rep_dim).mul_(2 * stdv).add_(stdv))

        self.query_layer_A = nn.Linear(rep_dim, rep_dim)
        self.key_layer_A = nn.Linear(rep_dim, rep_dim)
        self.value_layer_A = nn.Linear(rep_dim, rep_dim)
        self.query_layer_B = nn.Linear(rep_dim, rep_dim)
        self.key_layer_B = nn.Linear(rep_dim, rep_dim)
        self.value_layer_B = nn.Linear(rep_dim, rep_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        n_data = self.memory_v1.size(0)
        rep_dim = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        
        # contrast v2
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()

        query_v2 = self.query_layer_A(v2)
        key_v1 = self.key_layer_A(weight_v1)
        value_v1 = self.key_layer_A(weight_v1)
        attention_score_v2 = torch.matmul(query_v2, key_v1.transpose(-2, -1) / math.sqrt(query_v2.size(-1)))
        attention_weight_v2 = self.softmax(attention_score_v2)
        out_v2 = torch.matmul(attention_weight_v2, value_v1)
        out_v2 = torch.exp(torch.div(out_v2, T))

        # contrast v1
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()

        query_v1 = self.query_layer_B(v1)
        key_v2 = self.key_layer_B(weight_v2)
        value_v2 = self.key_layer_B(weight_v2)
        attention_score_v1 = torch.matmul(query_v1, key_v2.transpose(-2, -1) / math.sqrt(query_v1.size(-1)))
        attention_weight_v1 = self.softmax(attention_score_v1)
        out_v1 = torch.matmul(attention_weight_v1, value_v2)
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * n_data
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * n_data
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2
