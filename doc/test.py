import json
from net.moudle import resnet50
import torch
from torchvision import transforms
from PIL import Image

class predict_decoder(object):
    def __init__(self, dataset="CelebA", list=""):
        with open('./label.json', 'r') as f:
            label = json.load(f)[dataset]
            if list == "":
                self.label_list = label
            else:
                self.label_list = [label[i] for i in list]
        with open('./attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        print(self.label_list)
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred, source):
        #pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.label_list[idx], self.attribute_dict[self.label_list[idx]][pred[idx]]
            #if chooce
            print('{}: {} source: {}'.format(name, chooce, source[idx]))
            #return name, chooce


if __name__ == '__main__':
    checkpoint = torch.load("")
    model = resnet50(class_num=11)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    img = Image.open("")
    img = img.convert("RGB")
    img = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    ])(img)
    img = img.unsqueeze(dim=0)

    out = model(img.cuda())
    pred = torch.gt(out, torch.ones_like(out) * 0.8)

    dec = predict_decoder(dataset="CelebA", list=[6, 7, 8, 9, 11, 15, 20, 31, 32, 25, 33])
    dec.decode(pred=pred, source = out.data.cpu().numpy())