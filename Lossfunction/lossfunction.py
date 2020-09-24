from torch import nn
import torch
from torch.nn import functional as F
import time

class focal_loss(nn.Module):
    def __init__(self, alpha = 0.25, gamma=2, num_class=11, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha: 阿尔法α,类别权重.
                    当α是列表时,为各类别权重；
                    当α为常数时,类别权重为[α, 1-α, 1-α, ....],
                    常用于目标检测算法中抑制背景类,
                    retainnet中设置为0.25
        :param gamma: 伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes: 类别数量
        :param size_average: 损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()

        self.size_average = size_average

        if isinstance(alpha, list):
            assert len(alpha) == num_class   #对不同类别赋予不同权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1 #a为常数， 降低第一类的影响
            self.alpha = torch.zeros(num_class)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds: 预测类别. size:[B,N,C] or [B,C]    分
                别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        #self.alpha = torch.FloatTensor(alpha).cuda()
        preds = preds.view(-1, preds.size(-1))  #[2,40]
        self.alpha = self.alpha.cuda()
        preds_softmax = F.softmax(preds, dim=1)

        preds_logsoft = torch.log(preds_softmax)
        # preds_softmax = preds_softmax.gather(dim=1, index=labels)
        # preds_logsoft = preds_logsoft.gather(1, labels)
        preds_softmax = preds_softmax.gather(dim=1, index=labels)
        preds_logsoft = preds_logsoft.gather(1, labels)
        self.alpha = self.alpha.gather(0, labels.view(-1))                         # [80]
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) #[2,40]

        loss = torch.mul(self.alpha, loss.view(-1))
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

