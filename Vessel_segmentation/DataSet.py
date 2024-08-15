from torch.utils.data import Dataset
from pre_processing import *
from common import *
from torchvision import transforms
from PIL import Image
import re


def load_file_path_txt(file_path):
    img_list = []
    gt_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # read a line
            if not lines:
                break
            parts = re.split(r'\s+', lines)  # 使用正则表达式按照空格拆分
            if len(parts) == 2:
                img, gt = parts
                img_list.append(img)
                gt_list.append(gt)
            else:
                print(f"Skipping invalid line: {lines}")
    # print("img_list:",len(img_list))
    return img_list, gt_list


def load_data(data_path_list_file, target_size=(584, 584)):
    print('\033[0;33mload data from {} \033[0m'.format(data_path_list_file))
    img_list, gt_list = load_file_path_txt(data_path_list_file)
    imgs = None
    groundTruth = None
    for i in range(len(img_list)):
        # 加载图像
        img = Image.open(img_list[i])
        gt = Image.open(gt_list[i])
        # 调整图像大小为目标尺寸
        img = img.resize(target_size, Image.ANTIALIAS)
        gt = gt.resize(target_size, Image.ANTIALIAS)
        # 转换为NumPy数组
        img = np.array(img)
        gt = np.array(gt)
        # 如果地面真相是彩色的，则转换为单通道
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        # 扩展图像数组的维度
        img = np.expand_dims(img, axis=0)
        gt = np.expand_dims(gt, axis=0)
        # 初始化imgs和groundTruth
        if imgs is None:
            imgs = img
        else:
            imgs = np.concatenate((imgs, img), axis=0)
        if groundTruth is None:
            groundTruth = gt
        else:
            groundTruth = np.concatenate((groundTruth, gt), axis=0)

    assert ((np.min(groundTruth) == 0 and (np.max(groundTruth) == 255 or np.max(groundTruth) == 1)))
    if np.max(groundTruth) == 1:
        print("\033[0;31m Single channel binary image is multiplied by 255 \033[0m")
        groundTruth = groundTruth * 255

    # 转置imgs的维度为[N, C, H, W]
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    groundTruth = np.expand_dims(groundTruth, 1)
    print('ori data shape < ori_imgs:{} GTs:{}'.format(imgs.shape, groundTruth.shape))
    print("imgs pixel range %s-%s: " % (str(np.min(imgs)), str(np.max(imgs))))
    print("GTs pixel range %s-%s: " % (str(np.min(groundTruth)), str(np.max(groundTruth))))
    print("==================data have loaded======================")
    return imgs, groundTruth


def data_preprocess(data_path_list):
    train_imgs_original, train_masks = load_data(data_path_list)

    # print("train_imgs_original",train_imgs_original.max())
    # print(train_masks.max())
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks // 255
    return train_imgs, train_masks


"""---------------------------------------------------------------------------------------------"""


def is_patch_inside_FOV(x, y, img, patch_h, patch_w, mode='center'):
    """
    check if the patch is contained in the FOV,
    The center mode checks whether the center pixel of the patch is within fov,
    the all mode checks whether all pixels of the patch are within fov.
    """
    if mode == 'center':
        return img[y, x]
    elif mode == 'all':
        fov_patch = img[y - int(patch_h / 2):y + int(patch_h / 2), x - int(patch_w / 2):x + int(patch_w / 2)]
        return fov_patch.all()
    else:
        raise ValueError("\033[0;31mmode is incurrent!\033[0m")


def create_patch_idx(img, args):
    assert len(img.shape) == 4
    N, C, img_h, img_w = img.shape
    res = np.empty((args.N_patches, 3), dtype=int)

    seed = 2023
    count = 0
    while count < args.N_patches:
        random.seed(seed)  # fuxian
        seed += 1
        n = random.randint(0, N - 1)
        x_center = random.randint(0 + int(args.train_patch_width / 2), img_w - int(args.train_patch_width / 2))
        y_center = random.randint(0 + int(args.train_patch_height / 2), img_h - int(args.train_patch_height / 2))

        # check whether the patch is contained in the FOV
        if args.inside_img == 'center' or args.inside_img == 'all':
            if not is_patch_inside_FOV(x_center, y_center, img[n, 0], args.train_patch_height, args.train_patch_width,
                                       mode=args.inside_img):
                continue
        res[count] = np.asarray([n, x_center, y_center])
        count += 1

    return res


"""-----------------------Traindataset----------------------------------"""
"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""
"""
数据预处理方面
"""
import os
import cv2
import torch
import glob
import numpy as np
import torch.nn.functional as F

Directable = {'upper_left': [-1, -1], 'up': [0, -1], 'upper_right': [1, -1], 'left': [-1, 0], 'right': [1, 0],
              'lower_left': [-1, 1], 'down': [0, 1], 'lower_right': [1, 1]}
TL_table = ['lower_right', 'down', 'lower_left', 'right', 'left', 'upper_right', 'up', 'upper_left']

"""
sal2conn用在数据集制作用，用于之后损失函数边缘损失的计算
获得图像的边缘
"""


def sal2conn(mask):
    ## converte the saliency mask into a connectivity mask
    ## mask shape: H*W, output connectivity shape: 8*H*W
    [rows, cols] = mask.shape
    conn = torch.zeros([8, rows, cols])
    up = torch.zeros([rows, cols])  # move the orignal mask to up
    down = torch.zeros([rows, cols])
    left = torch.zeros([rows, cols])
    right = torch.zeros([rows, cols])
    up_left = torch.zeros([rows, cols])
    up_right = torch.zeros([rows, cols])
    down_left = torch.zeros([rows, cols])
    down_right = torch.zeros([rows, cols])

    up[:rows - 1, :] = mask[1:rows, :]
    down[1:rows, :] = mask[0:rows - 1, :]
    left[:, :cols - 1] = mask[:, 1:cols]
    right[:, 1:cols] = mask[:, :cols - 1]
    up_left[0:rows - 1, 0:cols - 1] = mask[1:rows, 1:cols]
    up_right[0:rows - 1, 1:cols] = mask[1:rows, 0:cols - 1]
    down_left[1:rows, 0:cols - 1] = mask[0:rows - 1, 1:cols]
    down_right[1:rows, 1:cols] = mask[0:rows - 1, 0:cols - 1]

    conn[0] = mask * down_right
    conn[1] = mask * down
    conn[2] = mask * down_left
    conn[3] = mask * right
    conn[4] = mask * left
    conn[5] = mask * up_right
    conn[6] = mask * up
    conn[7] = mask * up_left
    conn = conn.float()
    return conn


"""
以下部分是用于验证集上面的，计算MAP分数，
作为保存模型的最终评估
"""


def bv_test(output_test):
    '''
    generate the continous global map from output connectivity map as final saliency output
    via bilateral voting
    通过双边投票，从输出的连接图中生成连续的全局图，作为最终的显著性输出。
    '''

    # construct the translation matrix
    num_class = 1
    hori_translation = torch.zeros([output_test.shape[0], num_class, output_test.shape[3], output_test.shape[3]])
    for i in range(output_test.shape[3] - 1):
        hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
    verti_translation = torch.zeros([output_test.shape[0], num_class, output_test.shape[2], output_test.shape[2]])
    # print(verti_translation.shape)
    for j in range(output_test.shape[2] - 1):
        verti_translation[:, :, j, j + 1] = torch.tensor(1.0)

    hori_translation = hori_translation.float().cuda()
    verti_translation = verti_translation.float().cuda()
    output_test = F.sigmoid(output_test)
    pred = ConMap2Mask_prob(output_test, hori_translation, verti_translation)
    return pred


def shift_diag(img, shift, hori_translation, verti_translation):
    ## shift = [1,1] moving right and down
    # print(img.shape,hori_translation.shape)
    batch, class_num, row, column = img.size()

    if shift[0]:  ###horizontal
        img = torch.bmm(img.view(-1, row, column), hori_translation.view(-1, column, column)) if shift[
                                                                                                     0] == 1 else torch.bmm(
            img.view(-1, row, column), hori_translation.transpose(3, 2).view(-1, column, column))
    if shift[1]:  ###vertical
        img = torch.bmm(verti_translation.transpose(3, 2).view(-1, row, row), img.view(-1, row, column)) if shift[
                                                                                                                1] == 1 else torch.bmm(
            verti_translation.view(-1, row, row), img.view(-1, row, column))
    return img.view(batch, class_num, row, column)


def ConMap2Mask_prob(c_map, hori_translation, verti_translation):
    c_map = c_map.view(c_map.shape[0], -1, 8, c_map.shape[2], c_map.shape[3])
    batch, class_num, channel, row, column = c_map.size()

    shifted_c_map = torch.zeros(c_map.size()).cuda()
    for i in range(8):
        shifted_c_map[:, :, i] = shift_diag(c_map[:, :, 7 - i].clone(), Directable[TL_table[i]], hori_translation,
                                            verti_translation)
    vote_out = c_map * shifted_c_map

    pred_mask, _ = torch.max(vote_out, dim=2)
    # print(pred_mask)
    return pred_mask


class TrainDataset(Dataset):
    def __init__(self, imgs, masks, patches_idx, mode, args):
        self.imgs = imgs
        self.masks = masks
        self.patch_h, self.patch_w = args.train_patch_height, args.train_patch_width
        self.patches_idx = patches_idx
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                # RandomResize([56,72],[56,72]),
                RandomCrop((48,48)),
                # RandomCrop((224, 224)),
                # RandomCrop((64,64)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate(),
                # transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485],
                #     std=[0.229])
            ])
        self.mode = mode

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        n, x_center, y_center = self.patches_idx[idx]
        # print("patches_idx:",self.patches_idx.shape)
        data = self.imgs[n, :, y_center - int(self.patch_h / 2):y_center + int(self.patch_h / 2),
               x_center - int(self.patch_w / 2):x_center + int(self.patch_w / 2)]
        mask = self.masks[n, :, y_center - int(self.patch_h / 2):y_center + int(self.patch_h / 2),
               x_center - int(self.patch_w / 2):x_center + int(self.patch_w / 2)]
        # print("data",data.shape)
        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()
        if self.transforms:
            data, mask = self.transforms(data, mask)
        # mask=torch.Tensor(np.array(mask))
        mask = mask.squeeze(0)
        # print(mask.shape)
        if self.mode == "train":
            conn = sal2conn(mask)
            # print(conn.shape)
            # print(data.shape)
            return data, mask, conn
        else:
            return data, mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Resize:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape

    def __call__(self, img, mask):
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].byte()


class RandomResize:
    def __init__(self, w_rank, h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank

    def __call__(self, img, mask):
        random_w = random.randint(self.w_rank[0], self.w_rank[1])
        random_h = random.randint(self.h_rank[0], self.h_rank[1])
        self.shape = [random_w, random_h]
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].long()


class RandomCrop:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape
        self.fill = 0
        self.padding_mode = 'constant'

    def _get_range(self, shape, crop_shape):
        if shape == crop_shape:
            start = 0
        else:
            start = random.randint(0, shape - crop_shape)
        end = start + crop_shape
        return start, end

    def __call__(self, img, mask):
        _, h, w = img.shape
        sh, eh = self._get_range(h, self.shape[0])
        sw, ew = self._get_range(w, self.shape[1])
        return img[:, sh:eh, sw:ew], mask[:, sh:eh, sw:ew]


class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)


class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(1)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)


class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img, cnt, [1, 2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0, self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)


class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


"""-------------------------------Test-----------------------------------------------------------------"""


class TestDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs):
        self.imgs = patches_imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx, ...]).float()
