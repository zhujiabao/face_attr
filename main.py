import config
from net.moudle import resnet50
import torch
from datafolder.data_load import myDataset
from core.base import Trainer, Tester
import torch.optim as optim
import os
from utils import draw
import torch.nn as nn
from Lossfunction.lossfunction import focal_loss


#获取config信息
args = config.args()
start_epoch = 0
#获取训练集和验证集
train_set= myDataset(img_dir=args.img_root, img_txt=args.train_txt)
num_class = train_set.num_class
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           shuffle=True, num_workers=2, drop_last=True)
#print(train_loader)
val_set= myDataset(img_dir=args.img_root, img_txt=args.val_txt)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                           shuffle=True, num_workers=2, drop_last=True)
#train_loader, val_loader = myDataset.split_dataset(dataset=dataset, batch_size=args.batch_size)

#获取模型结构
if  args.pretrained:
    model = resnet50(class_num=num_class,pretrained=args.pretrained)
    print("model load success")
else:
    model = resnet50(class_num=num_class)

if args.use_gpu and torch.cuda.is_available():
    model.cuda()

#定义优化器和损失函数
#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
# exp_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,  base_lr=args.lr, max_lr=0.005, step_size_up=20000, step_size_down=20000,
#                                                      scale_fn=None, scale_mode='cycle',cycle_momentum=False)

#自适应学习率ReduceLROnPlaateau
# exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, verbose=True, threshold=0.001, threshold_mode="rel",
#                                                               cooldown= 0, min_lr= 0, eps= 0.0001)


#加载原模型继续训练
if args.is_load_checkpoint:
    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    exp_lr_scheduler.last_epoch = start_epoch

#定义loss function
if args.lossfunc == "focalloss":
    loss = focal_loss()
else:
    loss = nn.BCELoss()

#plt
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
epoch_list = []
#获取trainer and tester 且训练
if args.train == "train":
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=exp_lr_scheduler,batch_size=args.batch_size,
                      empochs=args.epoch,
                      train_loader=train_loader, use_cuda=args.use_gpu)
    tester = Tester(val_loader=val_loader, model=model,
                    batch_size=args.batch_size, use_cuda=args.use_gpu)
    # epoch训练
    for epoch in range(start_epoch, args.epoch):
        train_loss, train_acc=trainer.train(epoch=epoch, criterion_ce=loss, numclass=num_class)
        val_loss, val_acc=tester.val_test(epoch=epoch, current_model=trainer.model, criterion_ce=loss, numclass=num_class)
        #exp_lr_scheduler.step()
        print("Epoch: {}/{} lr:{:.4f} train loss:{:.4f} train Acc:{:.4f} Val loss{:.4f} Val Acc:{:.4f}".format(epoch, args.epoch,
                                                                                                               optimizer.param_groups[0]['lr'], train_loss, train_acc, val_loss, val_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        epoch_list.append(epoch)

        checkpoint = {"model_state_dict": trainer.model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "lr": optimizer.param_groups[0]['lr']}
        save_model_file = os.path.join(args.save_model, 'net_%s.pkl'%epoch)
        #print(save_model_file)
        torch.save(checkpoint, save_model_file)

    draw(train_loss=train_loss_list, train_acc=train_acc_list, val_loss=val_loss_list, val_acc=val_acc_list, epoch=epoch_list)

else:
    tester = Tester(val_loader=val_loader, model=model,
                    batch_size=args.batch_size, use_cuda=args.use_gpu)
    tester.val_test(epoch=0, numclass=num_class)


