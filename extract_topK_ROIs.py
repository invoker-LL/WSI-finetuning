import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP_SAVE
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline

import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--patch_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--topk_num', type=int, default=512)

args = parser.parse_args()

if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError
    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.data_h5_dir,exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    print('loading model checkpoint')

    from models.vib_clam_wsi import CLAM_SB
    model_dict = {'n_classes': 2}
    model_wsi = CLAM_SB(**model_dict)
    ckpt_path = 'clam_camelyon16_ostu_res50_pretrain_vib_s2021'
    ckpt_path = os.path.join('./results',ckpt_path,'s_0_checkpoint.pt')
    ckpt = torch.load(ckpt_path)
    # load weights
    model_wsi.load_state_dict(ckpt,strict=False)
    model_wsi.eval()
    model_wsi.cuda()

    total = len(bags_dataset)
    time_start = time.time()

    for bag_candidate_idx in range(total):
        try:
            slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
            bag_name = slide_id + '.h5'
            feat_file_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
            slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(slide_id)
            output_path = os.path.join(args.data_h5_dir, slide_id + '.pt')
            if os.path.exists(output_path):continue
            try:
                wsi = openslide.open_slide(slide_file_path)
            except:
                print('load error', slide_file_path)
                continue
            with h5py.File(feat_file_path, "r") as file:
                features = file['features'][:]
                coords = file['coords'][:]
            features = torch.as_tensor(features).cuda()
            logits, Y_prob, Y_hat, A, results_dict = model_wsi(features, testing=True)
            topk_indexes = torch.topk(A.squeeze(), min(args.topk_num,A.shape[0]), sorted=True)[1].cpu()

            patch_file_path = os.path.join(args.patch_dir, 'patches', slide_id+'.h5')
            dataset = Whole_Slide_Bag_FP_SAVE(file_path=patch_file_path,wsi=wsi,select_idx=topk_indexes)

            # coords_ok = coords[topk_indexes] == dataset.coords_new
            kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
            loader = DataLoader(dataset=dataset, batch_size=args.topk_num, **kwargs, collate_fn=collate_features)
            for count, (batch, coords) in enumerate(loader):
                print('count', count)
                img_batch= torch.as_tensor(batch*255,dtype=torch.uint8)
                dict = {'data':img_batch,'coords':coords}
                torch.save(dict,output_path)
        except:
            continue