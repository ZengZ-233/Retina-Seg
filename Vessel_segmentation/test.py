from test_function import *
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch, sys
from tqdm import tqdm
from visualize import concat_result, save_img
import os
from logger import Print_Logger
from os.path import join
from DataSet import TestDataset
from metrics import Evaluate
from common import setpu_seed, dict_round
from config import parse_args
from collections import OrderedDict
from ADUnet import ADUNet

setpu_seed(2023)


class Test():
    def __init__(self, args):
        self.args = args
        self.path_experiment = join(args.outf, args.save)
        self.patches_imgs_test, self.test_imgs, self.test_masks, self.new_height, self.new_width = get_data_test_overlap(
            test_data_path_list=args.test_data_path_list,
            patch_height=args.test_patch_height,
            patch_width=args.test_patch_width,
            stride_height=args.stride_height,
            stride_width=args.stride_width
        )
        # self.patches_imgs_test = self.patches_imgs_test[:1024*61, :, :, :]
        print("patches:", len(self.patches_imgs_test))
        self.img_height = self.test_imgs.shape[2]
        self.img_width = self.test_imgs.shape[3]
        test_set = TestDataset(self.patches_imgs_test)
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    def inference(self, net):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.eval()
        preds = []
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs = outputs[:, 1].data.cpu().numpy()
                preds.append(outputs)
        # print("preds:",len(preds))
        predictions = np.concatenate(preds, axis=0)
        self.pred_patches = np.expand_dims(predictions, axis=1)
        print("pred_patches:",self.pred_patches.shape)

    def evaluate(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## restore to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]
        print("pred_img[2]:",self.pred_imgs.shape)
        print("pred_img[2]:",self.pred_imgs.max())
        print("pred_img[2]:",self.pred_imgs.min())
        # 将输出值 self.output 转换为二进制标签，阈值通常为 0.5
        threshold = 0.5
        binary_pred_imgs = (self.pred_imgs >= threshold).astype(int)
        binary_test_masks = (self.test_masks >= threshold).astype(int)
        # y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks)
        y_scores, y_true = pred_only_in_FOV(binary_pred_imgs, binary_test_masks)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        log = eval.save_all_result(plot_curve=True,save_name="performance.txt")
        # save labels and probs for plot ROC and PR curve when k-fold Cross-validation
        np.save('{}/result.npy'.format(self.path_experiment), np.asarray([y_true, y_scores]))
        return dict_round(log, 6)

    def save_segmentation_result(self):
        img_path_list, _ = load_file_path_txt(self.args.test_data_path_list)
        img_name_list = [item.split('/')[-1].split('.')[0] for item in img_path_list]
        self.save_img_path = join(self.path_experiment, 'result_img')
        if not os.path.exists(join(self.save_img_path)):
            os.makedirs(self.save_img_path)
        # self.test_imgs = my_PreProc(self.test_imgs) # Uncomment to save the pre processed image
        print(self.test_imgs.shape)
        print(self.pred_imgs.shape)
        print(self.test_masks.shape)
        for i in range(self.test_imgs.shape[0]):
            total_img = concat_result(self.test_imgs[i], self.pred_imgs[i], self.test_masks[i])
            save_img(total_img, join(self.save_img_path, "Result_" + img_name_list[i] + '.png'))

    def val(self):
        self.pred_imgs = recompone_overlap(
            self.pred_patches, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)
        ## recover to original dimensions
        self.pred_imgs = self.pred_imgs[:, :, 0:self.img_height, 0:self.img_width]

        #predictions only inside the FOV
        y_scores, y_true = pred_only_in_FOV(self.pred_imgs, self.test_masks)
        eval = Evaluate(save_path=self.path_experiment)
        eval.add_batch(y_true, y_scores)
        confusion,accuracy,specificity,sensitivity,precision = eval.confusion_matrix()
        log = OrderedDict([('val_auc_roc', eval.auc_roc()),
                           ('val_f1', eval.f1_score()),
                           ('val_acc', accuracy),
                           ('SE', sensitivity),
                           ('SP', specificity)])
        return dict_round(log, 6)

if __name__ == '__main__':
    args = parse_args()
    save_path = join(args.outf, args.save)
    sys.stdout = Print_Logger(os.path.join(save_path, 'test_log.txt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ADUNet(1, 2).to(device)
    cudnn.benchmark = True
    print('==> Loading checkpoint...')
    checkpoint = torch.load(join(save_path, 'best_model.pth'))
    net.load_state_dict(checkpoint['net'])
    eval = Test(args)
    eval.inference(net)
    print(eval.evaluate())
    eval.save_segmentation_result()
