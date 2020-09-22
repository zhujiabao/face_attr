import torch.utils.data as Data
from  torch.utils import data
import torchvision.transforms as transforms
import torch
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def default_loader(path):
    try:
        img = Image.open(path)
        img = img.resize((224,224))
        #print(img)
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))



class myDataset(Data.Dataset):
    def __init__(self, img_dir, img_txt, transform=None, loader=default_loader):
        img_list = []
        img_labels = []

        #line_num = 0 # 计数器
        fp = open(img_txt, 'r')
        for line in fp.readlines():
            #if line_num < 100:
                if len(line.split()) != 12:
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
        if transform is None:
            self.transform = transforms.Compose([
                                    transforms.Resize(224), #224
                                    #transforms.CenterCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                                ])

        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index])).float()
        #label = np.array(self.labels[index])
        img = self.loader(img_path)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print('Cannot transform image: {}'.format(img_path))

        return img, label

    def split_dataset(dataset, batch_size):
        data_size = len(dataset)
        validation_split = .2
        shuffle = True
        random_seed = 42

        indices = list(range(data_size))
        #print(validation_split * data_size)
        split = int(np.floor(validation_split * data_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sample = Data.SubsetRandomSampler(train_indices)
        validation_sample = Data.SubsetRandomSampler(val_indices)

        train_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=train_sample)
        val_loader = Data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sample)

        return train_loader, val_loader


    def compute_attr_weights(dir):
        #train_txt = open("../data/train.txt", "w")
        #val_txt = open("../data/val.txt", "w")
        #list = [7, 8, 9, 10, 12, 14, 16, 21, 32, 33, 26, 34]
        with open(dir)as f:
            numofimgs = int(f.readline())
            line = f.readline()
            items = line.split()
            attrs = []
            for i in range(len(items)):
                attrs.append(items[i])
            # print(attrs)
            stats = []
            for i in range(len(attrs)):
                stat = []
                stat.append(0)
                stat.append(0)
                stats.append(stat)
            for i in range(numofimgs):
                line = f.readline()
                items = line.split()[1:]
                print(line.split()[0] ,len(items))
                for j in range(len(attrs)):
                    if items[j] == "1":
                        stats[j][0] += 1
                    else:
                        stats[j][1] += 1
            for i in range(len(attrs)):
                print(attrs[i], stats[i][0]+stats[i][1])


    def split_trainandval(dir):
            train_txt = open("../data/train.txt", "w")
            val_txt = open("../data/val.txt", "w")
            list = [6, 7, 8, 9, 11, 15, 20, 31, 32, 25, 33]
            with open(dir)as f:
                numofimgs = int(f.readline())
                line = f.readline()
                items = line.split()
                attrs = []
                for i in range(len(list)):
                    attrs.append(items[list[i]])
                print(attrs)
                info = ""
                for i in range(len(list)):
                    #info.append(attrs[i])
                    info = info + " " + attrs[i]
                train_txt.write(info+"\r\n")
                val_txt.write(info+"\r\n")
                for i in range(numofimgs):
                    line = f.readline()
                    img_path = line.split()[0]
                    items = line.split()[1:]
                    #info_2 = ""
                    if i % 2 ==0 or i%3==0:
                        for i in range(len(list)):
                            img_path = img_path + " " + items[list[i]]
                        train_txt.write(img_path+"\r\n")
                    else:
                        for i in range(len(list)):
                            img_path = img_path + " " + items[list[i]]
                        val_txt.write(img_path+"\r\n")




if __name__ == '__main__':
    from config import  args
    args = args()
    train_data= myDataset(img_dir=args.img_root, img_txt="../data/train.txt")
    #train_loader, val_loader = myDataset.split_dataset(dataset=train_data, batch_size=args.batch_size)
    #print(len(train_loader), len(val_loader))
    print(myDataset.compute_attr_weights("../data/train.txt"))
    #myDataset.split_trainandval("../data/Anno/list_attr_celeba.txt")
