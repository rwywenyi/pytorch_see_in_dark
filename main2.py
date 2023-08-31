import os
import time
import numpy as np
import glob
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from skimage.metrics import structural_similarity as ssimfunc
from skimage.metrics import peak_signal_noise_ratio as psnrfunc
import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import SeeInDark
from EDSR_model import EDSR
from util import reduce_mean, run_test, save_checkpoint, load_checkpoint
from dataset_sony import CustomDataset, CustomDatasetMemory
torch.distributed.init_process_group(backend='nccl')
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    opt = {'base_lr': 1e-4, 'batch_size': 2, 'start_epoch':1, 'epochs': 6001, 'save_frequency': 100,
            'test_frequency': 10000, 'patch_size': 512, 'load_weigths': False, 'debug':True}

    print(f"debug mode: {opt['debug']}")
    print(f"batch_size: {opt['batch_size']}")
    print(f"start_epoch: {opt['start_epoch']}")
    print(f"all train epochs: {opt['epochs']}")
    print(f"save_frequency: {opt['save_frequency']}")
    print(f"test_frequency: {opt['test_frequency']}")
    epoch = opt['start_epoch']

    writer = SummaryWriter(log_dir='./logs', filename_suffix='EDSR')

    # 保存模型、日志地址
    metric_average_file = 'metric_test_filename.csv'
    save_file_path = './EDSR_No_Ratio_BS_8'
    file_name = 'train_result'
    log_epoch_file = os.path.join(save_file_path, file_name, 'log')
    save_weights_file = os.path.join(save_file_path, file_name, 'weights')
    save_weights_file2 = os.path.join(save_file_path, file_name, 'weights2')
    save_images_file = os.path.join(save_file_path, file_name, 'images')
    save_csv_file = os.path.join(save_file_path, file_name, 'csv_files')
    csv_filename = os.path.join(save_file_path, file_name, 'test_result')

    if not os.path.exists(log_epoch_file):
        os.makedirs(log_epoch_file)
    if not os.path.exists(save_weights_file):
        os.makedirs(save_weights_file)
    if not os.path.exists(save_weights_file2):
        os.makedirs(save_weights_file2)
    if not os.path.exists(save_images_file):
        os.makedirs(save_images_file)
    if not os.path.exists(csv_filename):
        os.makedirs(csv_filename)
    if not os.path.exists(save_csv_file):
        os.makedirs(save_csv_file)

    # TODO 训练数据地址, 尽量使用相对路径
    # TODO glob.glob的结果是list！！！
    input_dir = "/home/xly/dms/ISP_test_data/SID/Sony/Sony/short/"
    gt_dir = "/home/xly/dms/ISP_test_data/SID/Sony/Sony/long/"
    train_gt_paths = glob.glob(gt_dir + '0*.ARW')
    train_ids = []
    for i in range(len(train_gt_paths)):
        if opt['debug']:
            if i > 6:
                break
        _, train_fn = os.path.split(train_gt_paths[i])
        train_ids.append(int(train_fn[0:5]))

    test_path = glob.glob(input_dir + '1*_00_*.ARW')
    test_ids = []
    for i in range(len(test_path)):
        if opt['debug']:
            if i > 6:
                break
        _, test_fn = os.path.split(test_path[i])
        test_ids.append(int(test_fn[0:5]))

    # Raw data takes long time to load. Keep them in memory after loaded.
    gt_images=[None]*6000
    input_images = {}
    input_images['300'] = [None]*len(train_ids)
    input_images['250'] = [None]*len(train_ids)
    input_images['100'] = [None]*len(train_ids)

    # 指定训练设备,构建dataloader
    # train_dataset = CustomDataset(train_ids, input_dir, gt_dir, patch_size=opt['patch_size'])
    train_dataset = CustomDatasetMemory(train_ids, input_dir, gt_dir, input_images, gt_images, patch_size=opt['patch_size'])
    # test_dataset = CustomDataset(test_ids, input_dir, gt_dir, training=False)
    dataloader_train = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=4, pin_memory=True, sampler=DistributedSampler(train_dataset))
    # dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    import argparse
    # Argument for EDSR
    parser = argparse.ArgumentParser(description='EDSR')
    parser.add_argument('--n_resblocks', type=int, default=32,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--scale', type=str, default=2,
                        help='super resolution scale')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='output patch size')
    parser.add_argument('--n_colors', type=int, default=4,
                        help='number of input color channels to use')
    parser.add_argument('--o_colors', type=int, default=3,
                        help='number of output color channels to use')
    args = parser.parse_args()

    # model = SeeInDark()
    model = EDSR(args)
    model.to(device)
    if opt['load_weigths']:
        model_weights = torch.load('/hy-tmp/learning_to_see_in_the_dark/saved_model/checkpoint_sony_e0500.pth')
        model.load_state_dict(model_weights)

        # load_checkpoint(path=, default_epoch=, modules=, optimizers=)

    # 如果是SeeInDark,则使用初始化权重
    if model.name == 'SeeInDark' and not opt['load_weigths']:
        model.initialize_weights()
        print('--'*8, '已使用initialize函数对model进行初始化', '--'*8)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank) # device_ids will include all GPU devices by default

    # model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=opt['base_lr'])
    optimizer.zero_grad()

    g_loss = np.zeros((5000,1))
    loss_list = ['loss_rgb']
    metrics = ['PSNR_rgb, SSIM_rgb']
    epoch_list = ['Iteration']
    epoch_LR = ['Iter_LR']

    print(f'model: {model.name}')
    print(f"Device: {device}")
    print(f'训练图像： {len(train_ids)}张')
    print(f'测试图像： {len(test_ids)}张')

    while epoch < opt['epochs']:
        print(f'epoch:{epoch:04}...........................')
        if epoch > 2000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
                print('降低学习率..............')
        st = time.time()
        for x, img in tqdm.tqdm(enumerate(dataloader_train)):
            input_raw = img[0].to(device)
            gt_rgb = img[1].to(device)
            ratio = img[2].cpu().data.numpy()
            train_id = img[3].cpu().data.numpy()

            pred_rgb = model(input_raw)
            loss = reduce_mean(pred_rgb, gt_rgb)
            g_loss[train_id]=loss.data.cpu()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # writer.add_scalar('train_loss', loss.item(), epoch)

            # 调整维度顺序，BCHW -》 BHWC，放cpu上并且转为numpy
            pred_rgb = pred_rgb.permute(0, 2, 3, 1).cpu().data.numpy()
            gt_rgb = gt_rgb.permute(0, 2, 3, 1).cpu().data.numpy()
            batch_size = gt_rgb.shape[0]

            if epoch % opt['save_frequency'] == 0:
                loss_list.append('{:.5f}'.format(loss.item()))

                epoch_result_dir = os.path.join(save_images_file, f'{epoch:04}/')
                if not os.path.isdir(epoch_result_dir):
                    os.makedirs(epoch_result_dir)

                for i in range(batch_size):
                    # TODO 对模型生成的RGB图像的上下边界进行压缩！！！
                    pred_rgb[i, :, :, :] = np.minimum(np.maximum(pred_rgb[i, :, :, :], 0), 1)
                    temp = np.concatenate((gt_rgb[i, :, :, :], pred_rgb[i, :, :, :]), axis=1)
                    Image.fromarray((temp * 255).astype('uint8')).save(epoch_result_dir + f'{train_id[i]:05}_00_{ratio[i]}.jpg')

        if epoch % opt['save_frequency'] == 0:
            mean_loss = np.mean(g_loss[np.where(g_loss)])
            loss_txt = os.path.join(log_epoch_file, 'loss_epoch.txt')
            with open(loss_txt,"a+",encoding="utf-8",newline="") as f:
                f.write(str(mean_loss))
                f.write("\n")

            torch.save({'model': model.state_dict()},
                       os.path.join(save_weights_file, 'weights_{}.pth'.format(epoch)))
            
            save_checkpoint(os.path.join(save_weights_file2, 'weights_{}.pth'.format(epoch)), epoch, model, optimizer)
            print('model saved......')

        print(f'epoch:{epoch:04}耗时:{time.time() - st:.3}s')
        
        if epoch % opt['test_frequency'] == 0:
            test_start_time = time.time()
            # run_test(model, dataloader_test, save_test_file, metric_average_file, device)
            run_test(model, test_ids, input_dir, gt_dir, epoch, csv_filename)
            print(f'test耗时: {time.time() - test_start_time:.3}')
        
        epoch += 1
