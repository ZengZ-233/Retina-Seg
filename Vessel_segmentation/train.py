from config import parse_args
# import torch.backends.cudnn as cudnn
from logger import *
import torch.optim as optim
from function import *
from loss_ import *
import streamlit as st
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


def train_and_test(op, model, size, ep, ptrain):
    # lr_init = 1e-2
    setpu_seed(2023)
    args = parse_args(size, ptrain, ep)
    # wandb.init(config=args)
    save_path = join(args.outf, args.save)
    save_args(args, save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path, 'train_log.txt'))
    print('The computing device used is: ', 'GPU' if device.type == 'cuda' else 'CPU')
    # net = Net_.Net(inplanes=args.in_channels, num_classes=args.classes, layers=3, filters=16).to(device)

    net = model(1, 2)
    print("Total number of parameters: " + str(count_parameters(net)))
    # wandb.watch(models=net,log_freq=100)
    # log.save_graph(net, torch.randn((1, 1, 224, 224)).to(device).to(device=device))
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = op(net.parameters(), lr=args.lr)
    if args.pre_train:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(save_path + '/latest_model.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    # criterion_val = CrossEntropyLoss2d()
    criterion_val = nn.CrossEntropyLoss()
    # criterion_val = lovasz_softmax

    # size = (args.train_patch_height, args.train_patch_width)  # 图像尺寸
    # criterion_train = bicon_loss(size)
    train_loader, val_loader = get_dataloader(args)
    ###图片加载
    st.markdown("预处理图片:balloon:")
    get_pictures()
    ##训练
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    best = {'epoch': 0, 'AUC_roc': 0.5}  # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter

    train_list = []
    train_chart = st.line_chart()
    for epoch in range(args.start_epoch, args.N_epochs + 1):

        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
              (epoch, args.N_epochs, optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        # train stage
        if epoch != 0 and epoch % 50 == 0:
            for param in optimizer.param_groups:
                param['lr'] *= 0.1
        train_log = train(train_loader, net, criterion_val, optimizer, device, args)
        train_list.append(train_log["train_loss"])
        train_chart.line_chart(train_list)
        # val stage
        val_log = val(val_loader, net, criterion_val, device)
        log.update(epoch, train_log, val_log)  # Add log information
        lr_scheduler.step()
        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_auc_roc'] > best['AUC_roc']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'], best['AUC_roc']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        message(epoch, args, val_log)
        torch.cuda.empty_cache()


def get_pictures():
    imgs = {"p1": r"D:\Pytouch\Vessel_segmentation\Vessel_segmentation\result\Unet_vessel_seg\sample_input_imgs.png",
            "p2": r"D:\Pytouch\Vessel_segmentation\Vessel_segmentation\result\Unet_vessel_seg\sample_input_masks.png"}
    p1 = Image.open(imgs["p1"])
    p2 = Image.open(imgs["p2"])
    st.image(p1, caption='sample_input_imgs')
    st.image(p2, caption='sample_input_masks')


def message(epoch, args, val_log):
    container = st.empty()
    container.write(f"""现在训练轮数{epoch}/{args.N_epochs} || 测试分数{val_log}""")

# if __name__ == '__main__':
#     train_and_test()
