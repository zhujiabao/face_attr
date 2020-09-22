#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: zhujiabao
@project: celeba_recognition
@file: train.py
@function:
@time: 9/10/20 9:56 AM
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:54:36 2018
@author: sky-hole
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np
import os
from argparse import ArgumentParser
from model import resnet50

parser = ArgumentParser(description="Train the model of face attrbute recongition")
parser.add_argument("--train", type=str, default="train",help="train/val")
parser.add_argument("--img_root", type=str, default="data/img_align_celeba/")
parser.add_argument("--train_txt", type=str, default="data/Anno/list_attr_celeba.txt")
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--epoch",type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--use_gpu", type=bool, default=True)

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def default_loader(path):
    try:
        img = Image.open(path)
        img = img.resize((224,224))
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))


class myDataset(Data.DataLoader):
    def __init__(self, img_dir, img_txt=args.train_txt, transform=None, loader=default_loader):
        img_list = []
        img_labels = []

        #line_num = 0 # 计数器
        fp = open(img_txt, 'r')
        for line in fp.readlines():
            #if line_num < 100:
            if len(line.split()) != 41:
                continue
            img_list.append(line.split()[0])
            img_label_single = []
            for value in line.split()[1:]:
                if value == '-1':
                    img_label_single.append(0)
                if value == '1':
                    img_label_single.append(1)
            img_labels.append(img_label_single)
                #line_num +=1
        #print("img_labels=",img_labels)
        self.imgs = [os.path.join(img_dir, file) for file in img_list]
        #print("imgs",self.imgs)
        self.labels = img_labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index])).long()
        #label = np.array(self.labels[index])
        img = self.loader(img_path)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Cannot transform image: {}'.format(img_path))

        return img, label


transform = transforms.Compose([
    transforms.Resize(224),
    #transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    #transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
])

dataset = myDataset(img_dir=args.img_root, img_txt=args.train_txt, transform=transform)
train_dataloader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
dataset_size = len(dataset)

# model optimizer and loss
if torch.cuda.is_available():
    model = resnet50().cuda()
else:
    model = resnet50()

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

def train(model, optimizer, scheduler, num_epoces):
    for epoch in range(num_epoces):
        if args.train == 'train':
            scheduler.step()
            model.train(True)
        else:
            model.train(False)

        running_loss = 0.0
        running_corrects = 0.0

        for count, (img, labels) in enumerate(train_dataloader):
            #labels = torch.LongTensor(labels)
            if args.use_gpu:
                img = img.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            pred_label = model(img)
            total_loss = loss_function(pred_label, torch.max(labels, 1)[1])
            total_loss.backward()
            optimizer.step()
            preds = torch.gt(pred_label, torch.ones_like(pred_label)/2)

            running_loss += total_loss.item()
            running_corrects += torch.sum(preds == labels.byte()).item() / 40

            if count % 100 == 0:
                print("Epoch {}/{} step: {}/{}  |  labels loss: {:.4f}".format(epoch,num_epoces, count*args.batch_size, dataset_size, total_loss.item()))

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_corrects / dataset_size
        print("{} Loss: {:.4f}    Acc: {:.4f}".format("Train", epoch_loss, epoch_acc))

        torch.save(model, "model/net_%s.pth" %epoch)

if __name__ == '__main__':
    train(model, optimizer, exp_lr_scheduler, num_epoces=args.epoch)
