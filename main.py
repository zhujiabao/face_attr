import config
from net.mobilenet_v3 import MobileNetV3_Small
import torch
from datafolder.data_load import myDataset, load_dataset
from core.base import Trainer, Tester
import torch.optim as optim
import os
from utils import draw
import torch.nn as nn
from Lossfunction.lossfunction import focal_loss
from sklearn.model_selection import KFold

# 获取config信息
args = config.args()
start_epoch = 0
# 获取训练集和验证集
dataset = myDataset(img_dir=args.img_root, img_txt=args.train_txt, transform="xxx")
num_class = dataset.num_class
print("the class number is", num_class)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
#                                            shuffle=True, num_workers=8, drop_last=True)
# # print(train_loader)
# val_set = myDataset(img_dir=args.img_root, img_txt=args.val_txt)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
#                                          shuffle=True, num_workers=8, drop_last=True)
# train_loader, val_loader = myDataset.split_dataset(dataset=dataset, batch_size=args.batch_size)

# 获取模型结构
if args.pretrained:
    model = MobileNetV3_Small(class_num=num_class)
    print("MobileNetV3_Small load success")
else:
    model = MobileNetV3_Small(class_num=num_class)
    print("MobileNetV3_Small load without pretrained model")

if args.use_gpu and torch.cuda.is_available():
    model.cuda()

# 定义优化器和损失函数
# optimizer = optim.Adam(model_xioayin.parameters(), lr=args.lr, weight_decay=1e-8)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.6)
# exp_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,  base_lr=args.lr, max_lr=0.005, step_size_up=20000, step_size_down=20000,
#                                                      scale_fn=None, scale_mode='cycle',cycle_momentum=False)
# ignored_params = list(map(id, model.features.parameters()))
# classifier_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
# optimizer = torch.optim.SGD([
#     {'params': model.features.parameters(), 'lr': 0.001},
#     {'params': classifier_params, 'lr': 0.01},
# ], momentum=0.9, weight_decay=5e-4, nesterov=True)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6335, gamma=0.9)

# 加载原模型继续训练
if args.is_load_checkpoint:
    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    exp_lr_scheduler.last_epoch = start_epoch
    print("load the {} success, start_epoch is {} ".format(args.checkpoint_file, start_epoch))

# 定义loss function
if args.lossfunc == "focalloss":
    loss = focal_loss()
else:
    loss = nn.BCELoss()

# plt
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
epoch_list = []
# 获取trainer and tester 且训练
if args.train == "train":
    # 7折交叉验证
    kf = KFold(7, shuffle=True, random_state=10)
    k = 0
    for train_index, val_index in kf.split(dataset):
        train_set = []
        val_set = []
        print("start")
        for i in train_index:
            train_set.append(dataset[i])
        print("success train_Set size", len(train_set))
        # print(train_set[0].shape)
        for j in val_index:
            val_set.append(dataset[j])
        print("success val_set size", len(val_set))

        train_set = load_dataset(data_set=train_set)
        val_set = load_dataset(data_set=val_set)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=4, drop_last=True)
        # train_set.clear()
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=4, drop_last=True)
        # val_set.clear()
        trainer = Trainer(model=model, optimizer=optimizer, scheduler=exp_lr_scheduler, batch_size=args.batch_size,
                          empochs=args.epoch,
                          train_loader=train_loader, use_cuda=args.use_gpu)
        tester = Tester(val_loader=val_loader, model=model,
                        batch_size=args.batch_size, use_cuda=args.use_gpu)

        # epoch训练
        for epoch in range(args.epoch):
            curr_epoch = start_epoch + k * args.epoch + epoch + 1
            train_loss, train_acc = trainer.train(epoch=curr_epoch, criterion_ce=loss, numclass=num_class)
            val_loss, val_acc = tester.val_test(epoch=curr_epoch, current_model=trainer.model, criterion_ce=loss,
                                                numclass=num_class)
            # exp_lr_scheduler.step()
            print(
                "Epoch: {}/{} lr:{:.6f}  train loss:{:.4f} train Acc:{:.4f} Val loss{:.4f} Val Acc:{:.4f}".format(
                    curr_epoch, start_epoch + 7 * args.epoch,
                    optimizer.param_groups[0]['lr'], train_loss, train_acc, val_loss,
                    val_acc))

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            epoch_list.append(epoch)

            # if epoch % 5 == 0:
        checkpoint = {"model_state_dict": trainer.model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": curr_epoch,
                      "lr": optimizer.param_groups[0]['lr']}
        save_model_file = os.path.join(args.save_model, 'net_%s.pkl' % curr_epoch)
        # print(save_model_file)
        torch.save(checkpoint, save_model_file)
        k += 1
        # torch.save(trainer.model.state_dict(), "final.pth")
    draw(train_loss=train_loss_list, train_acc=train_acc_list, val_loss=val_loss_list, val_acc=val_acc_list,
         epoch=epoch_list)

else:
    print("...... finish")
    # tester = Tester(val_loader=val_loader, model=model,
    #                 batch_size=args.batch_size, use_cuda=args.use_gpu)
    # tester.val_test(epoch=0, numclass=num_class)


