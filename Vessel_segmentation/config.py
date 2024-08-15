import argparse


def parse_args(size,pre_train,ep):
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default=r'D:\Pytouch\Vessel_segmentation\Vessel_segmentation\result')
    parser.add_argument('--save', default='Unet_vessel_seg')
    parser.add_argument("--pre_train",default=pre_train)
    # parser.add_argument("pre_trained",
    #                     default=r'D:\Pycharm_save\Vessel_segmentation\result\Unet_vessel_seg\best_model.pth')
    parser.add_argument('--N_epochs', default=ep, type=int)
    # parser.add_argument('--train_img_path_list',default='D:/vscode_/vesselSeg_segmentation/blood/train_imgs.txt')
    # parser.add_argument('--tarin_mask_path_list',default='D:/vscode_/vesselSeg_segmentation/blood/train_masks.txt')
    parser.add_argument('--train_data_path_list', default=r'D:\Pytouch\Vessel_segmentation\Vessel_segmentation\blood\a.txt')
    parser.add_argument('--N_patches', default=2000)
    parser.add_argument('--inside_img', default='center')
    parser.add_argument('--train_patch_height', default=size,type=int)
    parser.add_argument('--train_patch_width', default=size,type=int)
    parser.add_argument('--val_ratio', default=0.2)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--sample_visualization', default=True)
    parser.add_argument('--val_on_test', default=False)
    parser.add_argument('--start_epoch', default=1)
    """-----------------------------Model--------------------------"""
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--early-stop', default=10, type=int)
    """------------------------------Test------------"""
    parser.add_argument('--stride_height', default=16)
    parser.add_argument('--stride_width', default=16)
    parser.add_argument('--test_patch_height', default=96)
    parser.add_argument('--test_patch_width', default=96)
    parser.add_argument('--test_data_path_list', default=r'D:\Pytouch\Vessel_segmentation\Vessel_segmentation\blood\test_.txt')
    args = parser.parse_args()
    return args


def lr_change(epoch, lr):
    if epoch != 0 and epoch % 50 == 0:
        lr = lr / 10
        return lr
    else:
        return lr
