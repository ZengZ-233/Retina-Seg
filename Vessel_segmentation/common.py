import numpy as np
import os, joblib
import torch, random
import torch.nn as nn
import cv2, PIL


def setpu_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_args(args, save_path):
    if not os.path.exists(save_path):
        os.makedirs('%s' % save_path)

    print('Config info -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    with open('%s/args.txt' % save_path, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)
    joblib.dump(args, '%s/args.pkl' % save_path)
    print('\033[0;33m================config infomation has been saved=================\033[0m')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        result=self.nll_loss(torch.log(inputs), targets)
        return result


def readImg(img_path):
    img_format = img_path.split(".")[-1]
    img = None  # 初始化img变量
    try:
        img = PIL.Image.open(img_path)
    except Exception as e:
        ValueError("Reading failed, please check path of dataset,", img_path)
    return img


class AverageMeter(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)


def dict_round(dic, num):
    for key, value in dic.items():
        dic[key] = round(value, num)
    return dic