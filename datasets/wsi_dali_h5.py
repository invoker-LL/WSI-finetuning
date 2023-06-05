import pdb
import random
import numpy as np
import os, json

from random import shuffle
def shuffle_list(*ls,seed=0):
    random.seed(seed)
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


import h5py
import pandas as pd


def filter_df(df, filter_dict):
    if len(filter_dict) > 0:
        filter_mask = np.full(len(df), True, bool)
        # assert 'label' not in filter_dict.keys()
        for key, val in filter_dict.items():
            mask = df[key].isin(val)
            filter_mask = np.logical_and(filter_mask, mask)
        df = df[filter_mask]
    return df

def df_prep(data, label_dict, ignore, label_col):
    if label_col != 'label':
        data['label'] = data[label_col].copy()

    mask = data['label'].isin(ignore)
    data = data[~mask]
    data.reset_index(drop=True, inplace=True)
    for i in data.index:
        key = data.loc[i, 'label']
        data.at[i, 'label'] = label_dict[key]

    return data

def get_split_from_df(slide_data, all_splits, split_key='train'):
    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)

    if len(split) > 0:
        # pdb.set_trace()
        mask = slide_data['slide_id'].isin(split.tolist())
        df_slice = slide_data[mask].reset_index(drop=True)
        # split = Generic_IMG_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, train_eval=split_key)
        return df_slice

    else:
        return None
label_dict={0: 0, 1: 1, 2: 2,
            'IDC': 0, 'ILC': 1}
def get_data_list(data_dir, split_num, train_val_test, splitter_path, label_csv_path):
    # load
    slide_data = pd.read_csv(label_csv_path)
    slide_data = filter_df(slide_data, {})
    slide_data = df_prep(slide_data, label_dict=label_dict, ignore=[], label_col='label')
    all_splits = pd.read_csv(splitter_path, dtype=slide_data['slide_id'].dtype)
    #
    df_slice = get_split_from_df(slide_data, all_splits, split_key=train_val_test)
    slide_id_list = df_slice['slide_id'].tolist()
    label_list_ori = df_slice['label'].tolist()
    wsi_list, label_list, roi_rankings = [], [], []

    for i, slide in enumerate(slide_id_list):
        wsi_path = os.path.join(data_dir, slide + '.pt')
        if os.path.exists(wsi_path):
            wsi_list.append(wsi_path)
            label_list.append(label_list_ori[i])
        wsi_roi_ranking_path = os.path.join(data_dir, 'top_roi_split'+str(split_num), slide + '.npy')
        if os.path.exists(wsi_roi_ranking_path):
            roi_rankings.append(wsi_roi_ranking_path)
        else:
            roi_rankings.append(False)
    return wsi_list, label_list, roi_rankings

import torch

class ExternalInputCallable:
    def __init__(self, data_dir, batch_size, split_num,splitter_path, shuffle=False, device_id=0, num_gpus=1, train_eval_test='val',
                 bag_size=1024,label_csv_path='/data1/lhl_workspace/CLAM-master/dataset_csv/camelyon16.csv'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_val_test = train_eval_test
        self.bag_size = bag_size
        self.num_gpus = num_gpus
        self.device_id = device_id
        # pdb.set_trace()
        self.wsi_list_all, self.label_list_all, self.roi_rankings_all = get_data_list(data_dir,split_num, self.train_val_test, splitter_path,label_csv_path=label_csv_path)
        if self.shuffle:
            self.wsi_list_all, self.label_list_all, self.roi_rankings_all = shuffle_list(self.wsi_list_all, self.label_list_all, self.roi_rankings_all)

        temp_dataset_len = len(self.wsi_list_all)

        self.wsi_list = self.wsi_list_all[temp_dataset_len // num_gpus * device_id:
                                 temp_dataset_len // num_gpus * (device_id + 1)]

        self.label_list = self.label_list_all[temp_dataset_len // num_gpus * device_id:
                                     temp_dataset_len // num_gpus * (device_id + 1)]
        self.roi_rankings = self.roi_rankings_all[temp_dataset_len // num_gpus * device_id:
                                     temp_dataset_len // num_gpus * (device_id + 1)]
        self.data_set_len = len(self.wsi_list)
        self.index_list = range(self.data_set_len)

        self.n = len(self.wsi_list)
        self.full_iterations = len(self.wsi_list) // batch_size

        self.perm = None  # permutation of indices
        self.last_seen_epoch = None  # so that we don't have to recompute the `self.perm` for every sample
        self.i = 0

    def __call__(self, sample_info):
        if sample_info >= self.full_iterations:
            if self.shuffle:
                self.wsi_list_all, self.label_list_all, self.roi_rankings_all = shuffle_list(self.wsi_list_all,
                                                                                             self.label_list_all,
                                                                                             self.roi_rankings_all)
                self.wsi_list, self.label_list, self.roi_rankings = self.wsi_list_all, self.label_list_all, self.roi_rankings_all
            raise StopIteration

        batch, labels = [], []
        for _ in range(self.batch_size):
            sample_idx = sample_info
            full_path = self.wsi_list[sample_idx]
            data_dict = torch.load(full_path)
            img_data = data_dict['data'].permute(0,2,3,1).cpu()
            instance_num = len(img_data)
            for j in range(self.bag_size):
                if j < instance_num:
                    batch.append(np.ascontiguousarray(img_data[j]))
                    # batch.append(img_data[j])
                    labels.append(np.array([self.label_list[sample_idx], 0], dtype=np.uint8))
                else:
                    batch.append(batch[-1])
                    labels.append(np.array([self.label_list[sample_idx], 1], dtype=np.uint8))
        # pdb.set_trace()
        return (batch, labels,)
    @property
    def size(self, ):
        return self.data_set_len

def TrainPipeline(eii, batch_size, num_threads, device_id, seed, img_size=256, use_h5=True):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed)
    device = 'gpu'
    with pipe:
        if use_h5:
            images, labels = fn.external_source(source=eii, num_outputs=2)
        else:
            jpegs, labels = fn.external_source(source=eii, num_outputs=2)
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if device == 'gpu':
            images=images.gpu()
        if img_size!=256:
            images = fn.resize(images, size=img_size)
        # hsv
        # random_prob = fn.random.coin_flip(probability=0.2)
        # fn.cast(random_prob, dtype=types.FLOAT)
        # if True:
        #     images = fn.hsv(images.gpu(), hue=random_prob * fn.random.uniform(range=(0.0, 359.0)),
        #                                 saturation=random_prob * fn.random.uniform(range=(0.8, 1.2)),
        #                                 value=random_prob * fn.random.uniform(range=(0.8, 1.2)),device='gpu')
        # brightness
        images = fn.brightness_contrast(images, contrast=fn.random.uniform(range=(0.8, 1.2)),
                                        brightness=fn.random.uniform(range=(0.8, 1.2)),device=device)
        # rotate
        images = fn.rotate(images, keep_size=True, angle = fn.random.uniform(range=(-180.0, 180.0)), fill_value=0, device=device)

        # random resized_crop
        images = fn.random_resized_crop(images, size=(img_size, img_size), random_area=[0.7,1.0], device=device)
        # random flip
        images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5),
                         vertical=fn.random.coin_flip(probability=0.5), device=device)

        # scale to 0~1.0
        images = images / 255.0
        # normalize
        images = fn.normalize(images, device=device, axes=[0,1],mean=np.array([0.485, 0.456, 0.406]).reshape((1,1,3)),
                              stddev=np.array([0.229, 0.224, 0.225]).reshape((1,1,3)))

        images = fn.transpose(images, device=device, perm=[2, 0, 1])
        pipe.set_outputs(images, labels)
    return pipe

def ValPipeline(eii, batch_size, num_threads, device_id, seed, img_size=256, use_h5=True, use_gpu=True):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed)
    if use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'
    with pipe:
        if use_h5:
            images, labels = fn.external_source(source=eii, num_outputs=2)
        else:
            jpegs, labels = fn.external_source(source=eii, num_outputs=2)
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        # scale to 0~1.0
        if device == 'gpu':
            images=images.gpu()
        if img_size!=256:
            images = fn.resize(images, size=img_size)
        images = images / 255.0
        # normalize
        images = fn.normalize(images, device=device, axes=[0, 1],
                              mean=np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),
                              stddev=np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))
        images = fn.transpose(images, device=device, perm=[2, 0, 1])
        pipe.set_outputs(images, labels)
    return pipe

def get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='val',
                   splitter_path='./', device_id=0, num_gpus=1, seed=1, bag_size=1024,label_csv_path='./',split_num=0,):
    eii = ExternalInputCallable(data_dir=data_dir, batch_size=batch_size,split_num=split_num,
                                splitter_path=splitter_path, shuffle=shuffle,
                                device_id=device_id, num_gpus=num_gpus, train_eval_test=train_eval_test, bag_size=bag_size,
                                label_csv_path=label_csv_path)
    img_size=224
    if train_eval_test=='train':
        pipe = TrainPipeline(batch_size=batch_size * bag_size, eii=eii, num_threads=num_threads, device_id=device_id,
                                seed=seed + device_id,img_size=img_size)
    elif train_eval_test=='val':
        pipe = ValPipeline(batch_size=batch_size * bag_size, eii=eii, num_threads=num_threads, device_id=device_id,
                              seed=seed + device_id, img_size=img_size,)
    else:
        pipe = ValPipeline(batch_size=batch_size * bag_size, eii=eii, num_threads=num_threads, device_id=device_id,
                           seed=seed + device_id,use_gpu=True,img_size=img_size,)
    pipe.build()
    loader = DALIClassificationIterator(pipe, size=eii.size * bag_size,auto_reset=True,
                                        last_batch_padded=True,prepare_first_batch=False)

    return loader

def get_train_val_test_loaders(data_dir,splitter_path, bag_size=256, label_csv_path='./',split_num=0,args=None):
    train_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=True, num_threads=2, train_eval_test='train',split_num=split_num,
                                  splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=bag_size,label_csv_path=label_csv_path)
    val_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='val',split_num=split_num,
                                splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=bag_size,label_csv_path=label_csv_path)
    test_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='test',split_num=split_num,
                                splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=bag_size,label_csv_path=label_csv_path)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data_dir = '/data1/lhl_workspace/CLAM-master/topk_rois/v2/'
    train_val_test= 'train'
    splitter_path = "/data5/kww/CLAM/splits/task_1_tumor_vs_normal_100/splits_0.csv"
    label_csv_path = '/data1/lhl_workspace/CLAM-master/dataset_csv/camelyon16.csv'

    # x = get_data_list(data_dir,train_val_test, splitter_path,label_csv_path)
    # print(x,len(x[0]))
    train_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=True, num_threads=2, train_eval_test='train',
                   splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=1024)
    print(len(train_loader))

    val_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='val',
                                  splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=1024)
    test_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='test',
                                splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=1024)
    # print(train_loader)
    # test_loader.reset()
    pdb.set_trace()
