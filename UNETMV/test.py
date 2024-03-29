#Code adapted from https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV/test.py

import argparse
import os

import numpy as np
import torch
from networks.unetr import UNETR
from networks.unetmv import UNETMV
from trainer import dice
from utils.data_utils import get_loader

import nibabel as nib

from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--save_dir", default="./predictions/", type=str, help="directory to save the predicted segmentations")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
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
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")


def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    pretrained_model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, pretrained_model_name)
    # pretrained_pth = pretrained_dir.strip('\'') + pretrained_model_name
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
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
        model_dict = torch.load(pretrained_pth)
        try:
            model.load_state_dict(model_dict)
        except:
            model.load_state_dict(model_dict["state_dict"])
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        dice_list_case_org = []
        hd_list_case = []
        hd_list_case_org = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            val_inputs = val_inputs.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            hd_list_sub = []
            
            #save cropped ground truth, input image and image predictions             
            nib.save(nib.Nifti1Image(val_labels[0], np.eye(4)), os.path.join(args.save_dir, "ground_" + img_name))
            nib.save(nib.Nifti1Image(val_inputs[0], np.eye(4)), os.path.join(args.save_dir, "img_" + img_name))
            nib.save(nib.Nifti1Image(val_outputs[0], np.eye(4)), os.path.join(args.save_dir, "pred_" + img_name))

            for i in range(1, args.out_channels):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                if (val_labels[0] == i).sum() == 0:
                    organ_Dice = np.nan
                    
                hd = compute_hausdorff_distance(np.expand_dims(val_outputs, 0) == i, np.expand_dims(val_labels, 0) == i, percentile=95)[0][0]
                dice_list_sub.append(organ_Dice)
                hd_list_sub.append(hd)
                    
            mean_dice = np.nanmean(dice_list_sub)
            mean_hd = np.nanmean(hd_list_sub)
            dice_list_case_org.append(dice_list_sub)
            hd_list_case_org.append(hd_list_sub)
            print("Organ Dice: {}".format(["%0.2f" % i for i in dice_list_sub]))
            print("Mean Organ Dice: {}".format(mean_dice))            
            print("Organ 95HD: {}".format(["%0.2f" % i for i in hd_list_sub]))
            print("Mean 95HD: {}".format(mean_hd))
            dice_list_case.append(mean_dice)
            hd_list_case.append(mean_hd)

        print("Overall Organ Dice: {}".format(["%0.2f" % i for i in 100*np.nanmean(dice_list_case_org, axis = 0)]))
        print("Overall Mean Dice: {}".format(np.nanmean(dice_list_case)))
        print("Overall Organ 95HD: {}".format(["%0.2f" % i for i in np.nanmean(hd_list_case_org, axis = 0)]))
        print("Overall Mean 95HD: {}".format(np.nanmean(hd_list_case)))
        

def test_model(args, model, return_details = False):
    args.test_mode = True
    val_loader = get_loader(args)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        dice_list_case_org = []
        hd_list_case = []
        hd_list_case_org = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            val_inputs = val_inputs.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            hd_list_sub = []
  
            for i in range(1, args.out_channels):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                if (val_labels[0] == i).sum() == 0:
                    organ_Dice = np.nan
    
                hd = compute_hausdorff_distance(np.expand_dims(val_outputs, 0) == i, np.expand_dims(val_labels, 0) == i, percentile=95)[0][0]
                dice_list_sub.append(organ_Dice)
                hd_list_sub.append(hd)
                    
            mean_dice = np.nanmean(dice_list_sub)
            mean_hd = np.nanmean(hd_list_sub)
            dice_list_case_org.append(dice_list_sub)
            hd_list_case_org.append(hd_list_sub)
            print("Organ Dice: {}".format(["%0.2f" % i for i in dice_list_sub]))
            print("Mean Organ Dice: {}".format(mean_dice))            
            print("Organ 95HD: {}".format(["%0.2f" % i for i in hd_list_sub]))
            print("Mean 95HD: {}".format(mean_hd))
            dice_list_case.append(mean_dice)
            hd_list_case.append(mean_hd)

        print("Overall Organ Dice: {}".format(["%0.2f" % i for i in 100*np.mean(dice_list_case_org, axis = 0)]))
        print("Overall Mean Dice: {}".format(np.nanmean(dice_list_case)))
        print("Overall Organ 95HD: {}".format(["%0.2f" % i for i in np.nanmean(hd_list_case_org, axis = 0)]))
        print("Overall Mean 95HD: {}".format(np.nanmean(hd_list_case)))
        
    if return_details:
        return dice_list_case, hd_list_case
    return np.mean(dice_list_case)


if __name__ == "__main__":
    main()
