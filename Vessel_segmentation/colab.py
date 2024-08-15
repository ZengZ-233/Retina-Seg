import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from albumentations.augmentations import Resize
from torch.utils.data import DataLoader, Dataset
import numpy
import cv2
import os
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from glob import glob
from torch.nn import functional as F
import logging
import random, time
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(filename='/content/drive/MyDrive/Gnet.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s:  -%(acc)s: -%(se)s: -%(sp)s: -%(F1): -%(auc):',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds()


class blood_dataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_ids, mak_dir, num_class, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_ids = mask_ids
        self.mak_dir = mak_dir
        self.num_class = num_class
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id))
        mask = []
        # 将单通道灰度图像变为具有一个通道的多维数组
        # 需要单独截取
        mask_id = img_id.replace('images', 'vessel')
        # 1 DRIVE_AV/training/vessel/9.png
        # 2 DRIVE_AV/training/images/9.png
        mask = cv2.imread(os.path.join(self.mak_dir, mask_id), cv2.IMREAD_GRAYSCALE)
        new_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        new_mask[:, :, 0] = mask
        new_mask[:, :, 1] = mask
        new_mask[:, :, 2] = mask
        mask = new_mask
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        img = img.astype('float') / 225
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float') / 255
        mask = mask.transpose(2, 0, 1)
        # print(mask_id, img_id)
        return img, mask


with open(r'/content/drive/MyDrive/blood/train_imgs.txt', mode='r') as f:
    img_ids = f.readlines()
with open(r'/content/drive/MyDrive/blood/train_masks.txt', mode='r') as f:
    mask_ids = f.readlines()

img_ids = [''.join(list(i.strip())[2:]) for i in img_ids]
mask_ids = [''.join(list(i.strip())[2:]) for i in mask_ids]

train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=404)
train_mask_ids, val_mask_ids = train_test_split(mask_ids, test_size=0.2, random_state=404)
train_transform = Compose([
    OneOf([
        transforms.HueSaturationValue(),
        transforms.RandomBrightness(),
        transforms.RandomContrast()
    ], p=1),
    Resize(height=512, width=512),
    transforms.Normalize()])
val_transform = Compose([
    Resize(height=512, width=512),
    transforms.Normalize()])

img_dir = r'/content/drive/MyDrive/blood'
mask_dir = r'/content/drive/MyDrive/blood'
train_dataset = blood_dataset(img_ids, img_dir, mask_ids, mask_dir, 1, train_transform)
val_dataset = blood_dataset(val_img_ids, img_dir, mask_ids, mask_dir, 1, val_transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


class DAC_Block(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DAC_Block, self).__init__()
        self.one = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, dilation=1, padding=1)

        self.two = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=1, dilation=1)
        )
        self.three = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=1, dilation=3),
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=1, dilation=1),
        )
        self.four = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, dilation=5, padding=5),
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=1, dilation=1),
        )

    def forward(self, x):
        x1 = self.one(x)
        # print(x1.shape)
        x2 = self.two(x)
        # print(x2.shape)
        x3 = self.three(x)
        # print(x3.shape)
        x4 = self.four(x)
        # print(x4.shape)
        x = x1 + x2 + x3 + x4
        return x


class DropBlock2d(nn.Module):
    def __init__(self, p=0.5, block_size=3 * 3, inplace=False):
        super(DropBlock2d, self).__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError("p的范围是0-1")
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input):
        if not self.training:
            return input

        N, C, H, W = input.size()
        gamma = (self.p * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.full(mask_shape, gamma)
        mask = torch.bernoulli(mask)  # 伯努利分布0-1
        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size),
                            padding=self.block_size // 2)

        mask = 1 - mask
        normalize_scale = mask.numel() / (1e-6 + mask.sum())
        if self.inplace:
            input.mul_(mask * normalize_scale)  # 使mask的数变成0或者1
        else:
            input = input * mask * normalize_scale
        return input


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class Rse(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Rse, self).__init__()
        self.start = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
            DropBlock2d(p=0.5, block_size=3 * 3),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
            DropBlock2d(p=0.5, block_size=3 * 3),
        )
        self.se = SE_Block(input_channels)
        self.out = nn.Conv2d(input_channels, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        input = self.start(x)
        SE_ = self.se(input)
        SE_out = SE_ * input
        out = x + SE_out
        out = self.out(out)
        return out


class DACRse_Unet(nn.Module):
    def __init__(self, in_channel):
        super(DACRse_Unet, self).__init__()
        self.rse1 = Rse(input_channels=in_channel, output_channels=64)
        self.rse2 = Rse(input_channels=64, output_channels=128)
        self.rse3 = Rse(input_channels=128, output_channels=256)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dac = DAC_Block(256, mid_channel=256, out_channel=512)
        self.Deconv_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.Deconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.Deconv_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.drse3 = Rse(input_channels=256, output_channels=256)
        self.drse2 = Rse(input_channels=128, output_channels=128)
        self.drse1 = Rse(input_channels=64, output_channels=64)

        self.Last = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.rse1(x)
        # print(x1.shape)
        x1_pool = self.pool(x1)
        x2 = self.rse2(x1_pool)
        # print(x2.shape)
        x2_pool = self.pool(x2)
        # print(x2_pool.shape)
        x3 = self.rse3(x2_pool)
        # print(x3.shape)
        x3_pool = self.pool(x3)
        # print(x3_pool.shape)

        DAC_ = self.dac(x3_pool)
        # print(DAC_.shape)
        dx3 = self.Deconv_3(DAC_)
        # print(dx3.shape)
        rx3 = self.drse3(dx3)
        # print(rx3.shape)
        dx2 = self.Deconv_2(rx3)
        rx2 = self.drse2(dx2)
        dx1 = self.Deconv_1(rx2)
        rx1 = self.drse1(dx1)

        out = self.Last(rx1)
        return out


def recall(predict, target):
    """
    SE=TP/TP+FN
    """
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    fn = numpy.count_nonzero(~predict & target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def specificity(predict, target):
    """
    SP=TN/TN+FP
    """
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tn = numpy.count_nonzero(~predict & ~target)
    fp = numpy.count_nonzero(predict & ~target)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity


class Evaluate():
    def __init__(self, save_path=None):
        self.target = None
        self.output = None
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        self.threshold_confusion = 0.5

    # Add data pair (target and predicted value)
    def add_batch(self, batch_tar, batch_out):
        batch_tar = batch_tar.flatten()
        batch_out = batch_out.flatten()

        self.target = batch_tar if self.target is None else np.concatenate((self.target, batch_tar))
        self.output = batch_out if self.output is None else np.concatenate((self.output, batch_out))

    # Plot ROC and calculate AUC of ROC
    def auc_roc(self, plot=False):
        AUC_ROC = roc_auc_score(self.target, self.output)
        # print("\nAUC of ROC curve: " + str(AUC_ROC))
        if plot and self.save_path is not None:
            fpr, tpr, thresholds = roc_curve(self.target, self.output)
            # print("\nArea under the ROC curve: " + str(AUC_ROC))
            plt.figure()
            plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
            plt.title('ROC curve')
            plt.xlabel("FPR (False Positive Rate)")
            plt.ylabel("TPR (True Positive Rate)")
            plt.legend(loc="lower right")
            plt.savefig(join(self.save_path, "ROC.png"))
        return AUC_ROC

    # Plot PR curve and calculate AUC of PR curve
    def auc_pr(self, plot=False):
        precision, recall, thresholds = precision_recall_curve(self.target, self.output)
        precision = np.fliplr([precision])[0]
        recall = np.fliplr([recall])[0]
        AUC_pr = np.trapz(precision, recall)
        # print("\nAUC of P-R curve: " + str(AUC_pr))
        if plot and self.save_path is not None:
            plt.figure()
            plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_pr)
            plt.title('Precision - Recall curve')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower right")
            plt.savefig(join(self.save_path, "Precision_recall.png"))
        return AUC_pr

    # Accuracy, specificity, sensitivity, precision can be obtained by calculating the confusion matrix
    def confusion_matrix(self):
        # Confusion matrix
        y_pred = self.output >= self.threshold_confusion
        confusion = confusion_matrix(self.target, y_pred)
        # print(confusion)
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        # print("Global Accuracy: " +str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        # print("Specificity: " +str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        # print("Sensitivity: " +str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        # print("Precision: " +str(precision))
        return confusion, accuracy, specificity, sensitivity, precision

    # calculating f1_score
    def f1_score(self):
        pred = self.output >= self.threshold_confusion
        F1_score = f1_score(self.target, pred, labels=None, average='binary', sample_weight=None)
        # print("F1 score (F-measure): " +str(F1_score))
        return F1_score

    # Save performance results to specified file
    def save_all_result(self, plot_curve=True, save_name=None):
        # Save the results
        AUC_ROC = self.auc_roc(plot=plot_curve)
        AUC_pr = self.auc_pr(plot=plot_curve)
        F1_score = self.f1_score()
        confusion, accuracy, specificity, sensitivity, precision = self.confusion_matrix()
        if save_name is not None:
            file_perf = open(join(self.save_path, save_name), 'w')
            file_perf.write("AUC ROC curve: " + str(AUC_ROC)
                            + "\nAUC PR curve: " + str(AUC_pr)
                            + "\nF1 score: " + str(F1_score)
                            + "\nAccuracy: " + str(accuracy)
                            + "\nSensitivity(SE): " + str(sensitivity)
                            + "\nSpecificity(SP): " + str(specificity)
                            + "\nPrecision: " + str(precision)
                            + "\n\nConfusion matrix:"
                            + str(confusion)
                            )
            file_perf.close()
        return OrderedDict([("AUC_ROC", AUC_ROC), ("AUC_PR", AUC_pr),
                            ("f1-score", F1_score), ("Acc", accuracy),
                            ("SE", sensitivity), ("SP", specificity),
                            ("precision", precision)
                            ])


model = DACRse_Unet(in_channel=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


bce_fn = nn.BCEWithLogitsLoss()  # nn.NLLLoss()
dice_fn = SoftDiceLoss()


def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    if hard:
        dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    else:
        dice = dice_fn(y_pred.sigmoid(), y_true)
    return ratio * bce + (1 - ratio) * dice


optimizer = optim.Adam(model.parameters(), lr=0.001)
best_loss = 10
num_epochs = 10
raw_line = '{:6d}' + '\u2502{:7.4f}' + '\u2502{:6.2f}' + '\u2502{:6.2f}' + '\u2502{:6.2f}' + '\u2502{:6.2f}' + '\u2502{:6.2f}' + '\u2502{:6.2f}'
for epoch in range(num_epochs):
    losses = []
    start_time = time.time()
    model.train()
    running_loss = 0.0
    # for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
    #     inputs = inputs.to(device, dtype=torch.float32)
    #     masks = masks.to(device, dtype=torch.float32)

    #     optimizer.zero_grad()

    #     outputs = model(inputs)
    #     loss = loss_fn(outputs, masks)
    #     loss.backward()
    #     optimizer.step()
    #     losses.append(loss.item())
    #     running_loss += loss.item()

    # average_loss = running_loss / len(train_loader)
    # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    """
    验证
    """
    evaluater = Evaluate()
    model.eval()
    val_loss = 0.0
    acc = 0.0
    se = 0.0
    sp = 0.0
    f1 = 0.0
    auc_ = 0.0
    val_se = 0.0
    val_sp = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    val_auc = 0.0
    with torch.no_grad():
        for inputs, masks in tqdm(val_loader, desc=f"Validation"):
            inputs = inputs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            outputs = model(inputs)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            se = recall(outputs, masks)
            val_se += se
            sp = specificity(outputs, masks)
            val_sp += sp
            outputs = outputs.data.cpu().numpy()
            masks = masks.data.cpu().numpy()
            evaluater.add_batch(masks, outputs[:, 1])

    se = val_se / len(val_loader)
    sp = val_sp / len(val_loader)
    average_val_loss = val_loss / len(val_loader)

    logging.info(
        raw_line.format(epoch, np.array(losses).mean(), average_val_loss, evaluater.confusion_matrix()[1], se, sp,
                        evaluater.f1_score(), evaluater.auc_roc(), (time.time() - start_time) / 60 ** 1))

    print(f"Validation Loss: {average_val_loss:.4f}")
    if average_val_loss < best_loss:
        best_loss = average_val_loss
        torch.save(model.state_dict(), '/content/drive/MyDrive/Gnet.pth')
        print("best loss is{}".format(best_loss))

