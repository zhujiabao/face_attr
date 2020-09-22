from torch import nn
from torch.nn import init
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
write = SummaryWriter("runs/log")


def attr_weights(labels):
    weights = []
    states = np.zeros([40, 2])
    # print(states)
    for i in range(len(labels)):
        for j in range(40):
            # print(labels[i][j])
            if labels[i][j] == 1:
                states[j][0] += 1
            else:
                states[j][1] += 1

    for i in range(len(states)):
        # print(states[i][0] / (states[i][0] + states[i][1]))
        weights.append(states[i][0] / (states[i][0] + states[i][1]))
        # print(states[i][0] , states[i][1])

    # print(len(weights))
    return weights

class Trainer(object):
    def __init__(self, model, optimizer, scheduler, empochs,batch_size,train_loader, use_cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.enpochs = empochs
        self.train_loader = train_loader
        self.use_cuda = use_cuda
        self.scheduler = scheduler



    def train(self, epoch, criterion_ce):
        self.model.train(True)

        running_loss = 0.0
        running_corrects = 0.0

        for count, (img, labels) in enumerate(self.train_loader):
            if self.use_cuda:
                img = img.cuda()
                #alpha = attr_weights(labels.numpy())
                labels = labels.cuda()

            self.optimizer.zero_grad()
            pred_label = self.model(img)
            #print(pred_label.size(), labels.size())
            #total_loss = criterion_ce(pred_label, labels, alpha= alpha)
            total_loss = criterion_ce(pred_label, labels)
            #print("total_loss", len(total_loss))
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            preds = torch.gt(pred_label, torch.ones_like(pred_label)/2)

            running_loss += total_loss.item()
            running_corrects += torch.sum(preds == labels.byte()).item() / 11

            write.add_scalar("Training loss",total_loss.item() , epoch*len(self.train_loader) + count)
            write.add_scalar("lr", self.optimizer.param_groups[0]['lr'],  epoch*len(self.train_loader) + count)
            if count%20 == 0:
                print("Epoch {} step {} training loss {:.4f} ".format(epoch, epoch*len(self.train_loader) + count, total_loss.item()))
                # print("Epoch {}/{} train step: {}/{}  |  train labels loss: {:.4f}".format(epoch, self.enpochs,
                #                                                                count * self.batch_size,
                #                                                          len(self.train_loader), total_loss.item()))
        # log
        epoch_train_loss = running_loss / len(self.train_loader)
        epoch_train_acc = running_corrects / (len(self.train_loader)*self.batch_size)
        write.add_scalar("Training acc", epoch_train_acc , epoch)
        for name, param in self.model.named_parameters():
            write.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        #print("{} Loss: {:.4f}    Acc: {:.4f}".format("Train", epoch_loss, epoch_acc))
        return  epoch_train_loss, epoch_train_acc


class Tester(object):
    def __init__(self,  val_loader, model, batch_size, use_cuda=True):
        self.val_loader = val_loader
        self.model = model
        self.batch_size = batch_size
        self.use_cuda = use_cuda


    def val_test(self, epoch, criterion_ce,current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        val_loss = 0.0
        val_corrects = 0.0
        with torch.no_grad():
            for count, (img, labels) in enumerate(self.val_loader):
                if self.use_cuda:
                    img = img.cuda()
                    #alpha = attr_weights(labels.numpy())
                    labels = labels.cuda()

                pre_val_labels = self.model(img)
                #print(pre_val_labels ,labels)
                #total_val_loss = criterion_ce(pre_val_labels, labels, alpha=alpha)
                total_val_loss = criterion_ce(pre_val_labels, labels)
                preds = torch.gt(pre_val_labels, torch.ones_like(pre_val_labels) / 2)

                val_loss += total_val_loss.item()
                val_corrects += torch.sum(preds == labels.byte()).item() / 11

                #if count % 100 == 0:
                write.add_scalar("val loss", total_val_loss.item(), epoch*len(self.val_loader) + count)
                    # print("Epoch {}/{} val step: {}/{}  |  val labels loss: {:.4f}".format(epoch, 40,
                    #                                                                count * self.batch_size,
                    #                                                      len(self.val_loader), total_val_loss.item()))
            epoch_val_loss = val_loss / len(self.val_loader)
            epoch_val_acc = val_corrects / (len(self.val_loader)*self.batch_size)
            write.add_scalar("val acc", epoch_val_acc, epoch)
            return epoch_val_loss, epoch_val_acc
            #print("{} Loss: {:.4f}    Acc: {:.4f}".format("Val", epoch_val_loss, epoch_val_acc))





