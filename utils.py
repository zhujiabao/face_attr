'''
draw loss and acc
'''
import numpy as np
import matplotlib
matplotlib.use('agg')
import  matplotlib.pyplot as plt
import importlib


fig = plt.figure()
loss = fig.add_subplot(121, title="loss")
acc = fig.add_subplot(122, title="acc")
def draw(train_loss, train_acc, val_loss, val_acc, epoch):
    loss.plot(epocch,train_loss, 'bo-', label="train")
    loss.plot(epocch, val_loss, 'ro-', label="val")
    acc.plot(epocch,train_acc, 'bo-', label="train")
    acc.plot(epocch, val_acc, 'ro-', label="val")
    fig.savefig("train.jpg")