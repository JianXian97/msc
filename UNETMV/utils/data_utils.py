#Code adapted from https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV/main.py

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
from monai.utils import set_determinism

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def generate_splits(total_length, num_splits, split_num):
    total = np.arange(total_length)
 
    splits = {}
    for i in range(num_splits):
        splits[i] = total[i:total_length:num_splits]
        
    val_indices = splits[split_num]
    train_indices = np.array(list(set(total) - set(val_indices)))
    
    return train_indices, val_indices
    
    
def get_loader(args):
    if "BTCV" in args.data_dir or "ACDC" in args.data_dir:
        return get_loader_BTCV(args)
    elif "AMOS" in args.data_dir:
        return get_loader_AMOS(args)
    else:
        raise ValueError("Dataset not found")

def get_loader_BTCV(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    if not args.test_mode:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
 

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.kfold:
            train_indices, val_indices = generate_splits(len(datalist), args.num_kfold, args.kfold_split)
            val_files = np.array(datalist)[val_indices]
            datalist = np.array(datalist)[train_indices]
            
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        if not args.kfold: #if it's kfold, it has been generated in the prev lines
            val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader

from monai.config import KeysCollection
from monai.transforms.intensity.array import ScaleIntensityRange, NormalizeIntensity
from monai.utils.enums import TransformBackends

class dynamicScaled(transforms.transform.MapTransform):
    def __init__(self, 
                 keys: KeysCollection, 
                 img_key: str, 
                 type_key:str, 
                 a_min:float, 
                 a_max:float, 
                 b_min:float, 
                 b_max:float): 
        
        super().__init__(keys, allow_missing_keys = False)
        self.normalizer = NormalizeIntensity(subtrahend = None, divisor = None, nonzero = False, channel_wise = False, dtype = np.float32)
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip = True, dtype = np.float32)
        self.img_key = img_key
        self.type_key = type_key

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, data):
        d = dict(data)       
        if d[self.type_key] == 'CT': 
            d[self.img_key] = self.scaler(d[self.img_key])
        else:
            d[self.img_key] = self.normalizer(d[self.img_key])            
        return d

class CustomRandCropByPosNegLabeld(transforms.RandCropByPosNegLabeld):
    def __init__(self, keys, label_key, type_key, spatial_size_A, spatial_size_B, pos, neg, num_samples, image_key, image_threshold):
        _keys = [image_key, label_key]
        self.CT = transforms.RandCropByPosNegLabeld(_keys, label_key, spatial_size_A, pos, neg, num_samples, image_key, image_threshold)
        self.MRI = transforms.RandCropByPosNegLabeld(_keys, label_key, spatial_size_B, pos, neg, num_samples, image_key, image_threshold)
        self.type_key = type_key

    def __call__(self, data):
        d = dict(data)
        if d[self.type_key] == 'CT':
            return self.CT(data)
        else:
            return self.MRI(data)
        
def get_loader_AMOS(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    if not args.test_mode:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                ),
                # transforms.ScaleIntensityRanged(
                #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                # ),
                dynamicScaled(
                    keys=["image", "type"], img_key="image", type_key="type", a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # CustomRandCropByPosNegLabeld(
                #     keys=["image", "label", "type"],
                #     label_key="label",
                #     type_key="type",
                #     spatial_size_A=(160,160,64),
                #     spatial_size_B=(224,160,48),
                #     pos=1,
                #     neg=1,
                #     num_samples=4,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),             
            ]
        )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            # transforms.ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            dynamicScaled(
                keys=["image", "type"], img_key="image", type_key="type", a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
 

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.kfold:
            train_indices, val_indices = generate_splits(len(datalist), args.num_kfold, args.kfold_split)
            val_files = np.array(datalist)[val_indices]
            datalist = np.array(datalist)[train_indices]
            
        
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        if not args.kfold: #if it's kfold, it has been generated in the prev lines
            val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, val_loader]

    return loader
