import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time


class hswish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        out = x*self.relu6(x+3)/6
        return out
class hsigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        out = self.relu6(x+3)/6
        return out
class SE(nn.Module):
    def __init__(self, in_channels, reduce=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduce, 1, bias=False),
            nn.BatchNorm2d(in_channels//reduce),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels // reduce, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            hsigmoid()
        )
    def forward(self, x):
        out = self.se(x)
        out = x*out
        return out
class Block(nn.Module):
    def __init__(self, kernel_size, in_channels, expand_size, out_channels, stride, se=False, nolinear='RE'):
        super().__init__()
        self.se = nn.Sequential()
        if se:
            self.se = SE(expand_size)
        if nolinear == 'RE':
            self.nolinear = nn.ReLU6(inplace=True)
        elif nolinear == 'HS':
            self.nolinear = hswish()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expand_size, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_size),
            self.nolinear,
            nn.Conv2d(expand_size, expand_size, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            self.se,
            self.nolinear,
            nn.Conv2d(expand_size, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.stride = stride
    def forward(self, x):
        out = self.block(x)
        if self.stride == 1:
            out += self.shortcut(x)
        return out

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.numBottleNeck = 1000 #特征2048维，512是其后继层节点数
        self.modelName = str(type(self)).split('\'')[-2].split('.')[-1]

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            torch.save(self.state_dict(), time.strftime('snapshots/' + self.modelName + '%H:%M:%S.pth'))
        else:
            torch.save(self.state_dict(), path)

    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)


    def weights_init_classifier(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init.normal_(m.weight.data, std=0.001)
            init.constant_(m.bias.data, 0.0)


class ClassBlock(BasicNet):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        # add_block = []
        # add_block += [nn.Linear(input_dim, num_bottleneck)]
        # add_block += [nn.BatchNorm1d(num_bottleneck)]
        # add_block += [nn.LeakyReLU(0.1)]
        # add_block += [nn.Dropout(p=0.5)]
        #
        # add_block = nn.Sequential(*add_block)
        # add_block.apply(self.weights_init_kaiming)

        classifier = []
        classifier += [nn.Dropout(p=0.2)]
        classifier += [nn.Linear(input_dim, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)

        #self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        #x = self.add_block(x)
        x = self.classifier(x)
        return x

class MobileNetV3_Small(nn.Module):
    def __init__(self, class_num=40):
        super().__init__()

        self.class_num = class_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )
        self.neck = nn.Sequential(
            Block(3, 16, 16, 16, 2, se=True),
            Block(3, 16, 72, 24, 2),
            Block(3, 24, 88, 24, 1),
            Block(5, 24, 96, 40, 2, se=True, nolinear='HS'),
            Block(5, 40, 240, 40, 1, se=True, nolinear='HS'),
            Block(5, 40, 240, 40, 1, se=True, nolinear='HS'),
            Block(5, 40, 120, 48, 1, se=True, nolinear='HS'),
            Block(5, 48, 144, 48, 1, se=True, nolinear='HS'),
            Block(5, 48, 288, 96, 2, se=True, nolinear='HS'),
            Block(5, 96, 576, 96, 1, se=True, nolinear='HS'),
            Block(5, 96, 576, 96, 1, se=True, nolinear='HS'),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            hswish()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            hswish()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for c in range(class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=1280, class_num=1, activ='sigmoid') )

    def forward(self, x):
        x = self.conv1(x)
        x = self.neck(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = x.view(x.size(0), -1)
        x = self.pool(x)
        x = x.flatten(1)
        output = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        return torch.cat(output, dim=1)

if __name__ == "__main__":
    from torchsummary import summary
    model =  MobileNetV3_Small(40)
    summary(model, (3,224,224), device='cpu')
