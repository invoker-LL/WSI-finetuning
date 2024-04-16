import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
from torch.distributions import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
mse_loss = torch.nn.MSELoss()
# ce_loss_forKL = torch.nn.CrossEntropyLoss()
from torch.nn.functional import binary_cross_entropy as ce_loss_forKL
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}

        self.threshold = 0.1
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        self.update_vib=True
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # fc.append(attention_net)
        self.fc = nn.Sequential(*fc)
        self.attention_net = attention_net
        self.classifiers = nn.Linear(size[1], n_classes)
        # instance_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]

        self.instance_classifier = nn.Linear(size[1], 1)
        # self.instance_classifier = nn.Sequential(*xx)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifier = self.instance_classifier.to(device)

    def prior(self, var_size, device, threshold=0.2):
        p = torch.tensor([threshold], device=device)
        p = p.view(1, 1)
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]
        return p_prior

    def reparameterize(self, p_i, tau,num_sample=10):
        p_i_ = p_i.view(p_i.size(0), 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, p_i_.size(-1))  # Batch size, Feature size,num_samples
        C_dist = RelaxedBernoulli(tau, logits=p_i_)
        V = C_dist.sample().mean(dim=1)
        return V

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, train_epoch=-1,
                testing=False, instance_mask=None):
        h = self.fc(h)
        total_inst_loss = torch.tensor(0)

        if self.update_vib: # vib here
            instance_pred = self.instance_classifier(h)
            inst_logits = torch.nn.functional.logsigmoid(instance_pred)

            z_mask = self.reparameterize(p_i=inst_logits, tau=0.1)
            sigmoid_logits = torch.sigmoid(instance_pred)

            # option 1: use binary CE loss as KL div given predefined target
            # info_loss = ce_loss_forKL(sigmoid_logits, torch.ones_like(sigmoid_logits) * self.threshold)

            # option 2: use MSE on mean value of all instances
            mean_logits = torch.mean(sigmoid_logits)
            info_loss = mse_loss(mean_logits, torch.ones_like(mean_logits) * self.threshold)
            # option 2.1: use MSE on all single instances
            # info_loss = mse_loss(sigmoid_logits, torch.ones_like(sigmoid_logits) * self.threshold)

            total_inst_loss = info_loss
            if testing:
                Att_scores = inst_logits
                thresh = torch.topk(Att_scores.squeeze(), min(1024, Att_scores.shape[0]), sorted=True)[0][-1]
                masker = Att_scores.squeeze() >= thresh
                # pdb.set_trace()
                h = h[masker]
            else:
                z_mask = (z_mask.view(-1,1) + torch.sigmoid(instance_pred))/(1.0+self.threshold)
                h = z_mask*h

        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A

        if instance_mask is not None:
            A = A.masked_fill(instance_mask == 1, -1e9)

        A = F.softmax(A, dim=1)  # softmax over N
        # pdb.set_trace()
        M = torch.mm(A, h)

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(0),
                            'inst_preds': np.array(0)}

        if return_features:
            results_dict.update({'features': M})
        if instance_eval:
            return logits, Y_prob, Y_hat, Att_scores, results_dict
        else:
            return logits, Y_prob, Y_hat, Att_scores, results_dict

