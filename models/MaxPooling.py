import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPooling(nn.Module):
    def __init__(self, n_classes, input_size=384,feat_size=384):
        super(MaxPooling, self).__init__()
        self.input_size = input_size
        self._fc1 = nn.Sequential(nn.Linear(input_size,feat_size), nn.ReLU())
        self.n_classes = n_classes
        self._fc0 = nn.Linear(feat_size, 1)
        self._fc2 = nn.Linear(feat_size, self.n_classes)

    def forward(self, x):
        h = x[:, :self.input_size].unsqueeze(0)
        # h = kwargs['data'].float().cuda() #[B, n, 1024]
        # pdb.set_trace()
        h = self._fc1(h) #[B, n, 512]
        # pdb.set_trace()
        h = torch.max(h, dim=1)[0]
        # pdb.set_trace()
        # logits0 = self._fc0(h)
        # max_pos = torch.argmax(logits0)
        # h = h[:,max_pos]
        # pdb.set_trace()
        # pick max


        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)
        return logits, Y_hat, Y_prob, None, Y_hat
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 384)).cuda()
    model = MaxPooling(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
