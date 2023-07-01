import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.best import Network
from utils.data_val import test_dataset
from time import time


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/home/rec/Desktop/SInet/SINet-V2-main/snapshot/Ours/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['CAMO','COD10K']:
    data_path = './Dataset/TestDataset/{}/'.format(_data_name)
    save_path = './res/Best/{}/'.format(_data_name)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    model = Network(channel=32)
    model=torch.nn.DataParallel(model,device_ids=[0,1,2])
    # model = torch.nn.DataParallel(model, device_ids=[0])
    model.to(device)


    model.load_state_dict(torch.load(opt.pth_path))
 
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    edge_root = '{}/Edge/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, edge_root,opt.testsize)

    # print(model)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # res5, res4, res3, res2 ,_= model(image)
        res=model(image)
        res = res[3]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        # misc.imsave(save_path+name, res)
        # If `mics` not works in your environment, please comment it and then use CV2
        cv2.imwrite(save_path+name,res*255)
