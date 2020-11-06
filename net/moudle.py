from torchvision import models
from torch import nn
from classBlock import ClassBlock
import torch


class resnet50(nn.Module):
    def __init__(self, class_num, pretrained=False, model_name='resnet50'):
        super(resnet50, self).__init__()
        self.model_name = model_name
        self.class_num = class_num
        self.pretrained = pretrained

        model_ft = getattr(models, self.model_name)(pretrained=self.pretrained)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 2048

        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid') )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        return pred_label


if __name__ == "__main__":
    #model_xioayin = resnet50(class_num=11).cuda()
    #input = torch.randn(size=(1,3,224,224), dtype=torch.float32)
    #print(model_xioayin)
    # from torchsummary import summary
    # summary(model_xioayin, (3,224,224))
    #
    model = resnet50(class_num=40)
    tet = torch.Tensor(2, 3, 224, 224)

    print(torch.ones_like(tet) * 0.6)
    #print(model_xioayin(tet))         * 0.6
