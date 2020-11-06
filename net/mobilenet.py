import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time

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

class MobileNet(BasicNet):
    def __init__(self, numClass=40):
        super(MobileNet, self).__init__()
        self.numClass = numClass
        self.model = models.mobilenet_v2(pretrained=True).features
        #print(self.model)
        last_chanel = models.mobilenet_v2().last_channel
        print(last_chanel)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        for c in range(numClass):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=last_chanel, class_num=1, activ='sigmoid') )

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #x =torch.flatten(x, 1)
        output = [self.__getattr__('class_%d' % c)(x) for c in range(self.numClass)]
        # output = torch.cat(output, dim=1)
        return torch.cat(output, dim=1)

if __name__ == "__main__":
    from torchsummary import summary
    model =  MobileNet(40)
    summary(model, (3,224,224), device='cpu')

    #testTensor = torch.Tensor(2,3,224,224)
    # print(testTensor)
    # print(model_xioayin.parameters().class_0)
    #print(model)