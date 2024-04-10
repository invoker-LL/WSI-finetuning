import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 512), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 512)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        ### 1D convolutional layer that can handle multiple class (including binary)
        # self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

        self.fcc = nn.Linear(input_size, output_class)
    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats).view(feats.shape[1], -1)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[1], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, dim=1,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        # pdb.set_trace()
        m_feats = torch.index_select(feats, dim=1,index=m_indices[0,0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats).reshape(1,-1)  # compute queries of critical instances, q_max in shape C x Q
        # A = torch.mm(q_max.transpose(0,1),Q)
        A = torch.mm(Q, q_max.transpose(0,1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
                      0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, input_size=384, n_classes=2):
        super(DSMIL, self).__init__()
        self.input_size = input_size
        i_classifier = FCLayer(self.input_size, 1)
        b_classifier = BClassifier(input_size=self.input_size, output_class=n_classes)

        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        x = x[:, :self.input_size].unsqueeze(0)
        # x = kwargs['data'].float().cuda()
        feats, classes = self.i_classifier(x)
        logits, A, B = self.b_classifier(feats, classes)

        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)
        return logits, Y_hat, Y_prob, None, Y_hat

        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict



if __name__ == "__main__":
    data = torch.randn((1, 6000, 384)).cuda()
    model = DSMIL(num_feats=384).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)