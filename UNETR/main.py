# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import gc
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import torch.nn.functional as F
import test
from networks.unetr import UNETR
from networks.unetmv import UNETMV
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

import itertools
import joblib

import optuna
from optuna.trial._state import TrialState

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--checkpoint_filename", default="model.pt", type=str, help="checkpoint name")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--save_every", default=1000, type=int, help="save checkpoint frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--patch_x", default=16, type=int, help="roi size in x direction")
parser.add_argument("--patch_y", default=16, type=int, help="roi size in y direction")
parser.add_argument("--patch_z", default=16, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--decode_mode", default='simple', type=str, help="Decoder mode, simple or CA")
parser.add_argument("--cft_mode", default='channel', type=str, help="CFT mode, channel, patch or all")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--tune", action="store_true", help="Run tuning for ablation study")
parser.add_argument("--tune_load_dir", default=None, type=str, help="Resume tuning from file")
parser.add_argument("--tune_file_name", default="Tune Expt Results.pkl", type=str, help="File name of tuning results.")
parser.add_argument("--optuna", action="store_true", help="Run optuna, hyperparameter tuning")
parser.add_argument("--optuna_load_dir", default=None, type=str, help="Resume optuna optimisation from file")
parser.add_argument("--optuna_expt_file_name", default="OPTUNA Expt Results.pkl", type=str, help="File name of optuna experimental results.")
parser.add_argument("--optuna_study_file_name", default="OPTUNA study.pkl", type=str, help="File name of optuna study.")
parser.add_argument("--optuna_add_trials", default=None, type=str, help="File path to import optuna study.")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    # print("REMEMBER TO CHANGE THESE LATER!!!")
    # args.data_dir = "../../datasets/AMOS"
    # args.model_name = "unetmv"
    # args.logdir = "./runs/" + args.logdir
    # args.distributed = True
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        manager = mp.Manager()
        args.shared_list = manager.list()
                
    assert not (args.tune and args.optuna), "optuna and tune cannot be run simultaneously!"
    args.kfold = False 
    if args.optuna:
        mp.spawn(main_worker_optimise, nprocs=args.ngpus_per_node, args=(args,))
    elif args.tune:
        mp.spawn(main_worker_tune, nprocs=args.ngpus_per_node, args=(args,))
    else:            
        if args.distributed:            
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
        else:
            main_worker(gpu=0, args=args)

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    if (args.model_name is None) or args.model_name == "unetr" or args.model_name == "unetmv":
        if args.model_name == "unetr": 
            model = UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
            )
        elif args.model_name == "unetmv":
            model = UNETMV(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                patch_size=(args.patch_x, args.patch_y, args.patch_z),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate,
                decode_mode=args.decode_mode,
                cft_mode=args.cft_mode                
            )

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")

        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))
 
    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )
    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    hd_acc = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True, percentile=95)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight, momentum=args.momentum
        )
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        hd_func=hd_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
 
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # print("Indiv Print GPU " + str(torch.distributed.get_rank()))
    # print(torch.cuda.mem_get_info(device=torch.distributed.get_rank()))

    return accuracy

def main_worker_optimise(gpu, args):
    def objective(trial):
        # if args.rank == 0:
        #     args.shared_list.append(trial)
        objs = [trial]
        dist.broadcast_object_list(objs, src=0) #blocking call
        trial = objs[0]
        
        print("Trial Number " + str(trial.number))
        args.test_mode = False
        
        args.dropout_rate = trial.suggest_categorical("Dropout", np.arange(0,0.5,0.1))        
        # lr_list = [1e-6,1e-5,1e-4,1e-3]
        # if args.distributed:            
        #     lr_list = [x*args.ngpus_per_node for x in lr_list]
        args.hidden_size = trial.suggest_categorical("Hidden size, E", [18,36,54,72,90])
        args.feature_size = trial.suggest_categorical("Model feature size, F", [4,8,12,16,20])
        args.decode_mode = trial.suggest_categorical("Decode mode", ['CA', 'simple'])
        args.cft_mode =  trial.suggest_categorical("Cft mode", ['channel', 'patch', 'all'])
        
        args.mlp_dim = 4 * args.hidden_size
        
        model = UNETMV(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=(args.patch_x, args.patch_y, args.patch_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
            decode_mode=args.decode_mode,
            cft_mode=args.cft_mode                
        )
     
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
        post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
        post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
        dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
        hd_acc = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True, percentile=95)
        
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )
    
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters count", pytorch_total_params)
    
        start_epoch = 0
        model.cuda(args.gpu)
        if args.distributed:
            torch.cuda.set_device(args.gpu)
            if args.norm_name == "batch":
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
            )
        if args.optim_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight, momentum=args.momentum
            )
        elif args.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
            )
        else:
            raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    
        if args.lrschedule == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
            )
        elif args.lrschedule == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
            if args.checkpoint is not None:
                scheduler.step(epoch=start_epoch)
        else:
            scheduler = None
        accuracy = run_training(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            hd_func=hd_acc,
            args=args,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
        )
         
        test_accuracy = 0
        if args.rank == 0:
            args.train_distributed = args.distributed
            args.distributed = False #dont use distributed testing
            test_accuracy = test.test_model(args, model)
            args.distributed = args.train_distributed
            
            gc.collect()
            torch.cuda.empty_cache()
            print("Trial " + str(trial.number) + " TEST Accuracy " + str(test_accuracy))
            path = os.path.join(args.logdir, args.optuna_expt_file_name)
            study.trials_dataframe().to_pickle(path)            
            path = os.path.join(args.logdir, args.optuna_study_file_name)
            joblib.dump(study, path)     
            
             
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return test_accuracy #only gpu 0 will have a non 0 value. Other gpus will not update optuna
    
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]  
    print(args.rank, " gpu", args.gpu)
    n_trials = 30
    n_trials_rand = 20
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)   
        print("Running Optuna")
        if args.optuna_load_dir is not None:
            path = os.path.join(args.optuna_load_dir, args.optuna_study_file_name)
            try:
                study = joblib.load(path)
                print("loaded optuna study")
            except:
                study = optuna.create_study(study_name="optimise 100G", direction='maximize', sampler=optuna.samplers.RandomSampler())
                if args.optuna_add_trials is None:
                    study = add_default(study, args)
                print("Created optuna study!")
        else:   
            study = optuna.create_study(study_name="optimise 100G", direction='maximize', sampler=optuna.samplers.RandomSampler())
            if args.optuna_add_trials is None:
                study = add_default(study, args)
            print("Created optuna study")
        
        if args.optuna_add_trials is not None:
            study2 = joblib.load(args.optuna_add_trials)
            study.add_trials(study2.trials)
            print("Trials added!")
            print("Num of trials: " + str(len(study.trials)))
        
        if len(study.trials) < n_trials_rand:
            study.optimize(objective, n_trials=n_trials_rand)
               
        study.sampler = optuna.samplers.TPESampler()
        study.optimize(objective, n_trials= (n_trials - n_trials_rand))
    
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
    
        print("Best trial:")
        trial = study.best_trial
    
        print("  Value: ", trial.value)
    
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        path = os.path.join(args.logdir, args.optuna_expt_file_name)
        study.trials_dataframe().to_pickle(path)
    else:
        for i in range(n_trials):
            trial = None #this will be updated using broadcast 
            objective(trial) 
   
def add_default(study, args):
    params = {'Dropout': 0, 
              'Hidden size, E': 72, 
              'Model feature size, F': 16, 
              'Decode mode': 'simple', 
              'Cft mode': 'channel', 
              'lr': 1e-04}  
    if args.distributed:
        params['lr'] *=  args.ngpus_per_node
    study.enqueue_trial(params)
    return study
    
import pandas as pd
def main_worker_tune(gpu, args):
    def objective(combi, iteration):
        print("Combi Number " + str(iteration))
        args.test_mode = False
        
        args.hidden_size = combi['E']
        args.feature_size = combi['F']
        args.decode_mode = combi['decode_mode'] 
        args.cft_mode = combi['cft_mode']   
        
        args.mlp_dim = 4 * args.hidden_size
        
        model = UNETMV(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            patch_size=(args.patch_x, args.patch_y, args.patch_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
            decode_mode=args.decode_mode,
            cft_mode=args.cft_mode                
        )
     
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
        post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
        post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
        dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
        hd_acc = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True, percentile=95)
        
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )
    
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters count", pytorch_total_params)
    
        start_epoch = 0
        model.cuda(args.gpu)
        if args.distributed:
            torch.cuda.set_device(args.gpu)
            if args.norm_name == "batch":
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
            )
        if args.optim_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight, momentum=args.momentum
            )
        elif args.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
            )
        else:
            raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    
        if args.lrschedule == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
            )
        elif args.lrschedule == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
            if args.checkpoint is not None:
                scheduler.step(epoch=start_epoch)
        else:
            scheduler = None
        accuracy = run_training(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            hd_func=hd_acc,
            args=args,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
        )
         
        dice_scores, hd_scores = 0, 0
        if args.rank == 0:
            args.train_distributed = args.distributed
            args.distributed = False #dont use distributed testing
            dice_scores, hd_scores = test.test_model(args, model, return_details = True)
            args.distributed = args.train_distributed
            
            gc.collect()
            torch.cuda.empty_cache()
            test_accuracy = np.mean(dice_scores)
            print("Tune " + str(iteration) + " TEST Accuracy " + str(test_accuracy))
             
             
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return dice_scores, hd_scores #only gpu 0 will have a non 0 value. Other gpus will not update the results
    
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]  
    print(args.rank, " gpu", args.gpu)
    combinations = get_tune_comb()
    progress = 0
    prev_results = None
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)   
        print("Running Tune")
        if args.tune_load_dir is not None:
            path = os.path.join(args.tune_load_dir, args.tune_file_name)            
            try:
                prev_results = pd.read_pickle(path)
                progress = len(prev_results)
                print("loaded tune progress")
            except:      
                print("progress file not found! restart from 0")    
    
    objs = [progress]
    dist.broadcast_object_list(objs, src=0) #blocking call, synchronise across all gpus
    progress = objs[0]
    
    new_results = []
    for i, combi in enumerate(list(combinations.values())[progress:]):
        torch.distributed.barrier() #block until all gpus have reached here
        dice_scores, hd_scores = objective(combi, i + progress) 
        
        if args.rank == 0:
            combi['dice'] = dice_scores
            combi['dice_avg'] = np.mean(dice_scores)
            combi['hd'] = hd_scores
            combi['hd_avg'] = np.nanmean(hd_scores)
            new_results.append(combi)
            if prev_results is not None:
                temp = pd.concat([prev_results, pd.DataFrame.from_dict(new_results)])    
            else:
                temp = pd.DataFrame.from_dict(new_results)                       
            path = os.path.join(args.logdir, args.tune_file_name)
            temp.to_pickle(path)
            print("Saved progress")
        
            
def get_tune_comb():
    default = {'E': 72,
               'F': 16,
               'decode_mode': 'simple',
               'cft_mode': 'channel'
        }
    
    combinations = {}
    
    options = {'E' : [18,36,54,72,90],
              'F' : [4,8,12,16,20],
              'decode_mode': ['CA', 'simple'],
              'cft_mode': ['channel', 'patch', 'all']
              }
    count = 0
    for key in ['E', 'F']:
        new_set = default.copy()
        for value in options[key]:
            new_set[key] = value
            combinations[count] = new_set.copy()
            count += 1
    
    for dec in options['decode_mode']:
        for cft in options['cft_mode']:
            new_set = default.copy()
            new_set['decode_mode'] = dec
            new_set['cft_mode'] = cft
            combinations[count] = new_set.copy()
            count += 1
            
    return combinations
        
if __name__ == "__main__":
    main()
