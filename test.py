from net.model import resnet50
from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),#224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

model = torch.load('./checkpoint/net_0.pth')
model.cuda()

img = Image.open("000001.jpg")
img = img.resize((224,224))
img=img.convert('RGB')
img = transform(img)
img = img.unsqueeze(dim=0)

out = model(img.cuda())
pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5
print(pred)
print(out)
