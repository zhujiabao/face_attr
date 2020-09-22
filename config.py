from argparse import ArgumentParser

def args():
    parser = ArgumentParser(description="Train the model of face attrbute recongition")
    parser.add_argument("--train", type=str, default="train",help="train/val")
    parser.add_argument("--img_root", type=str, default="./data/img_align_celeba/")
    parser.add_argument("--train_txt", type=str, default="./data/train.txt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch",type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--is_load_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint_file", type=str, default="")
    parser.add_argument("--pretrained", type=bool, default=True, help="false means don't load pretrain model, True means load")
    parser.add_argument("--model_file", type=str, default="model/resnet50-19c8e357.pth", help="load the pretrain model")
    parser.add_argument("--save_model", type=str, default="checkpoint/")



    return parser.parse_args()

#args = args()