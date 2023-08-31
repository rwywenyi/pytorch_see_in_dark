import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from util import save_checkpoint, load_checkpoint
from EDSR_model import EDSR
from model import SeeInDark
from torch.utils.tensorboard import SummaryWriter


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out


torch.set_float32_matmul_precision('high')

# 数据地址、保存地址
writer = SummaryWriter(log_dir='./logs', filename_suffix='EDSR')
input_dir = "/home/xly/dms/ISP_test_data/SID/Sony/Sony/short/"
gt_dir = "/home/xly/dms/ISP_test_data/SID/Sony/Sony/long/"
result_dir = './EDSR_PS1024_Ratio/train_image/' # EDSR_PS1024_Ratio
model_dir = './EDSR_PS1024_Ratio/saved_model/'
model_dir2 = './EDSR_PS1024_Ratio/saved_model2/'
log_sample_file = './EDSR_PS1024_Ratio/loss_sample.txt'
log_epoch_file = './EDSR_PS1024_Ratio/loss_epoch.txt'
debug = True

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(model_dir2):
    os.makedirs(model_dir2)

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    if debug:
        if i >= 6:
            break
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

# 训练参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_ratio = False
ps = 1024
save_freq = 100
g_loss = np.zeros((5000,1))
lastepoch = 0
total_epoch = 6001
learning_rate = 1e-4

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

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

model = EDSR(args).to(device)
# model = SeeInDark().to(device)
# torch.compile加速训练，mode参数：'reduce-overhead':适合加速大模型
# 'default':适合加速小模型，需要额外存储空间 'max-autotune':编译速度最耗时，但提供最快的加速
# model = torch.compile(model, mode='max-autotune') # use compile GPU memory-usage 18495MiB
# model.initialize_weights()

# 加载权重，恢复训练
# check_point = torch.load('/hy-tmp/hy-tmp/learning_to_see_in_the_dark/result/result_EDSR_32_SE/weights/weights_3500.pth')
# model.load_state_dict(check_point['model'])

opt = optim.Adam(model.parameters(), lr = learning_rate)

print(f"Device: {device}")
print(f'is_ratio:{is_ratio}')
print(f'debug mode:{debug}')
print(f'train dataset number:{len(train_ids)}')
print(f'lastepoch:{lastepoch}') 
print(f'total_epoch:{total_epoch}') 
print(f'save_freq:{save_freq}')
print(f'patch size:{ps}')

for epoch in range(lastepoch, total_epoch): 
    cnt=0
    if epoch > 2000:
        for g in opt.param_groups:
            g['lr'] = 1e-5

    start_time_epoch = time.time()
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        start_time_sample = time.time()
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        in_path = in_files[np.random.randint(0,len(in_files))]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/in_exposure,300)
          
        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            if is_ratio:
                input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) * ratio
            else:
                input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)

        #crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = gt_images[ind][:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]
       
        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))
        
        input_patch = np.minimum(input_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2).to(device)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).to(device)

        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()
        opt.step()

        g_loss[ind]=loss.data.cpu()
        loss = g_loss[ind][0]
        with open(log_sample_file,"a+",encoding="utf-8",newline="") as f:
            f.write(str(loss))
            f.write("\n")

        writer.add_scalar('train_loss', loss.item(), epoch)

        print(f"Epoch: {epoch} \t Count: {cnt} \t Loss={loss:.3} \t Time={time.time()-st:.3}")
        print(f'sample{train_id} time consuming: {time.time() - start_time_sample:.3}')
        if epoch%save_freq==0:
            epoch_result_dir = os.path.join(result_dir, f'{epoch:04}')
            if not os.path.isdir(epoch_result_dir):
                os.makedirs(epoch_result_dir)

            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output,0),1)
            
            temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]),axis=1)
            Image.fromarray((temp*255).astype('uint8')).save(os.path.join(epoch_result_dir, f'{train_id:05}_00_train_{ratio}.jpg'))
            torch.save({'model': model.state_dict()}, model_dir+'EDSR_sony_e%04d.pth'%epoch)
            save_checkpoint(os.path.join(model_dir2, 'EDSR_sony_e%04d.pth'%epoch), epoch, model, opt)

    mean_loss = np.mean(g_loss[np.where(g_loss)])
    with open(log_epoch_file,"a+",encoding="utf-8",newline="") as f:
        f.write(str(mean_loss))
        f.write("\n")
    print(f'epoch{epoch} time consuming: {time.time() - start_time_epoch:.3}')