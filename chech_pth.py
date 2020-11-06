import torch
from net.model import resnet50
import torchvision
pthfile = './model_xioayin/resnet50-19c8e357.pth'
net =torchvision.models.resnet50(pretrained=True)
pretrained_dict =net.state_dict()

# for name, param in net.named_parameters():
#     print(name, param)
#net = resnet50(pretrained=True, model_path=pthfile)
model = resnet50()
model_dict = model.state_dict()


pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
#model_xioayin.load_state_dict(torch.load(pthfile), False)
for name, param in model.named_parameters():
    print(name, param)

'''tensor([[-0.0018,  0.0378, -0.0395,  ...,  0.0490, -0.0205,  0.0345]],
       requires_grad=True)
class_37.classifier.0.bias Parameter containing:
tensor([0.], requires_grad=True)
class_38.classifier.0.weight Parameter containing:
tensor([[-0.0097, -0.0005,  0.0666,  ..., -0.0596, -0.0085,  0.0394]],
       requires_grad=True)
class_38.classifier.0.bias Parameter containing:
tensor([0.], requires_grad=True)
class_39.classifier.0.weight Parameter containing:
tensor([[ 0.0328, -0.0045, -0.0083,  ..., -0.0460, -0.0646,  0.0164]],
       requires_grad=True)
class_39.classifier.0.bias Parameter containing:
tensor([0.], requires_grad=True)

tensor([[[[ 1.3335e-02,  1.4664e-02, -1.5351e-02,  ..., -4.0896e-02,
           -4.3034e-02, -7.0755e-02],
          [ 4.1205e-03,  5.8477e-03,  1.4948e-02,  ...,  2.2060e-03,
           -2.0912e-02, -3.8517e-02],
          [ 2.2331e-02,  2.3595e-02,  1.6120e-02,  ...,  1.0281e-01,
            6.2641e-02,  5.1977e-02],
          ...,
          [-9.0349e-04,  2.7767e-02, -1.0105e-02,  ..., -1.2722e-01,
           -7.6604e-02,  7.8453e-03],
          [ 3.5894e-03,  4.8006e-02,  6.2051e-02,  ...,  2.4267e-02,
           -3.3662e-02, -1.5709e-02],
          [-8.0029e-02, -3.2238e-02, -1.7808e-02,  ...,  3.5359e-02,
            2.2439e-02,  1.7077e-03]],

         [[-1.8452e-02,  1.1415e-02,  2.3850e-02,  ...,  5.3736e-02,
            4.4022e-02, -9.4675e-03],
          [-7.7273e-03,  1.8890e-02,  6.7981e-02,  ...,  1.5956e-01,
            1.4606e-01,  1.1999e-01],
          [-4.6013e-02, -7.6075e-02, -8.9648e-02,  ...,  1.2108e-01,
            1.6705e-01,  1.7619e-01],
          ...,
          [ 2.8818e-02,  1.3665e-02, -8.3825e-02,  ..., -3.8081e-01,
           -3.0414e-01, -1.3966e-01],
          [ 8.2868e-02,  1.3864e-01,  1.5241e-01,  ..., -5.1232e-03,
           -1.2435e-01, -1.2967e-01],
          [-7.2789e-03,  7.7021e-02,  1.3999e-01,  ...,  1.8427e-01,
            1.1144e-01,  2.3438e-02]],'''