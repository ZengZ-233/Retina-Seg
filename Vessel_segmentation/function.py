from torch.utils.data import DataLoader
from collections import OrderedDict
from metrics import Evaluate
from visualize import group_images, save_img
from DataSet import data_preprocess, create_patch_idx, TrainDataset
from tqdm import tqdm
from os.path import join
from common import *
from PIL import Image


def get_dataloader(args):
    imgs_train, masks_train = data_preprocess(data_path_list=args.train_data_path_list)
    patches_idx = create_patch_idx(imgs_train, args)

    train_idx, val_idx = np.vsplit(patches_idx, (int(np.floor((1 - args.val_ratio) * patches_idx.shape[0])),))
    # patches_idx_ = create_patch_idx(imgs_train, args)
    # val_idx,a=np.vsplit(patches_idx_, (int(np.floor((1 - args.val_ratio) * patches_idx.shape[0])),))

    train_set = TrainDataset(imgs_train, masks_train, train_idx, mode="train", args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_set = TrainDataset(imgs_train, masks_train, val_idx, mode="val", args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        visual_set = TrainDataset(imgs_train, masks_train, val_idx, mode="val", args=args)
        visual_loader = DataLoader(visual_set, batch_size=1, shuffle=True, num_workers=0)
        N_sample = 50
        visual_imgs = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))
        for i, (img, mask) in tqdm(enumerate(visual_loader)):
            visual_imgs[i] = np.squeeze(img.numpy(), axis=0)
            visual_masks[i, 0] = np.squeeze(mask.numpy(), axis=0)
            if i >= N_sample - 1:
                break
        save_img(group_images((visual_imgs[0:N_sample, :, :, :] * 255).astype(np.uint8), 10),
                 join(args.outf, args.save, "sample_input_imgs.png"))
        save_img(group_images((visual_masks[0:N_sample, :, :, :] * 255).astype(np.uint8), 10),
                 join(args.outf, args.save, "sample_input_masks.png"))

    return train_loader, val_loader


"""-------------------Train-------------------------"""
"""https://blog.csdn.net/weixin_39190382/article/details/114433884"""


def train(train_loader, net, criterion, optimizer, device, args):
    train_loss = AverageMeter()
    net=net.to(device)
    for batch_idx, (inputs, targets, conn) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, targets, conn = inputs.to(device), targets.to(device), conn.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        save_img_path = join(args.outf, args.save)
        save_img_path = join(save_img_path, "Train_Result")
        # concat_result_train(inputs.to("cpu"), outputs.to("cpu"), targets.to("cpu"), save_img_path)
        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([
        ('train_loss', float(train_loss.avg))
    ])
    return log


"""----------------------Val---------------------------"""


def val(val_loader, net, criterion, device):
    net.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))
            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets, outputs[:, 1])
    log = OrderedDict([
        ('val_loss', float(val_loss.avg)),
        ('val_acc', float(evaluater.confusion_matrix()[1])),
        ('val_f1', float(evaluater.f1_score())),
        ('val_auc_roc', float(evaluater.auc_roc()))
    ])

    return log
