import json
import os
import glob
import torch
import numpy as np
import nibabel as nib
import torchio as tio
import torch.nn.functional as F

from medpy import metric
from Utils.utils import read_n_permute
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def eval(pred_img, pred_label, ref_img, ref_label):
    label_result = eval_results_labels(pred_label, ref_label)
    img_result = eval_results_imgs(pred_img, ref_img)
    return label_result, img_result


def eval_results_imgs(pred, ref, channel_axis=0):
    ssim_scores = ssim(pred, ref, channel_axis=channel_axis)
    psnr_scores = psnr(pred, ref, data_range=255)
    return {
        "ssim": str(ssim_scores),
        "psnr": str(psnr_scores)
    }


def eval_results_labels(pred: np.ndarray, ref: np.ndarray):
    dsc_list, assd_list = list(), list()
    for i in range(12):
        dsc_list.append(str(metric.dc(pred[:, i, ...], ref[:, i, ...])))
        assd_list.append(str(metric.assd(pred[:, i, ...],
                                     ref[:, i, ...], voxelspacing=1)))
    verror_list = find_vol_error(pred, ref, 12, True).tolist()
    verror_list = [str(item) for item in verror_list]
    return {
        "dsc": dsc_list,
        "assd": assd_list,
        "verror": verror_list
    }


def find_vol_error(pred, gt, num_class, is_3d):
    def get_vol_count(pred_class: torch.Tensor,
                      gt_class: torch.Tensor):
        """
            returns the element count of the predicted muscle label elements,
            and that of the ground truth.

            :return: The count of the predicted muscle labels, the count of that of the GT.
        """
        return torch.count_nonzero(pred_class).item(), torch.count_nonzero(gt_class).item()

    phy_vol = 1 * 1 * 1
    vol_e = np.zeros(num_class)
    for idx in range(num_class):
        if is_3d:
            pred_this_c, gt_this_c = pred[:, idx, ...], gt[:, idx, ...]
        else:
            pred_this_c, gt_this_c = pred[:, idx, ...], gt[:, idx, ...]
        pred_count, gt_count = get_vol_count(torch.from_numpy(pred_this_c),
                                             torch.from_numpy(gt_this_c))
        pred_vol, gt_vol = (pred_count * phy_vol) / 1000, (gt_count * phy_vol) / 1000
        vol_e[idx] = abs(pred_vol - gt_vol)
    return vol_e


def eval_results(moved_data_path, ref_img_path, ref_label_path, output_path):
    rescale = tio.RescaleIntensity(out_min_max=(0, 255))
    data = sorted(glob.glob(os.path.join(moved_data_path, "*.nii.gz")))
    img_list = [img for img in data if "mdixon_r.nii.gz" in img]
    label_list = [label for label in data if "segmentation_r.nii.gz" in label]
    img_ref_list = [os.path.join(ref_img_path, os.path.basename(img)) for img in img_list]
    label_ref_list = [os.path.join(ref_label_path, os.path.basename(label)) for label in label_list]
    img_metric_dict, label_metric_dict = dict(), dict()
    for img_pred, label_pred, img_ref, label_ref in zip(img_list, label_list, img_ref_list, label_ref_list):
        img_ref_data = rescale(tio.Subject(
            image=tio.ScalarImage(tensor=read_n_permute(img_ref))))["image"].data.numpy()
        img_pred_data = rescale(tio.Subject(
            image=tio.ScalarImage(tensor=read_n_permute(img_pred))))["image"].data.numpy()
        img_metrics = eval_results_imgs(img_pred_data, img_ref_data)
        label_pred_data = F.one_hot(torch.from_numpy(nib.load(label_pred).get_fdata()).to(torch.int64),
                                    num_classes=12).permute(2, 3, 0, 1).numpy()
        label_ref_data = F.one_hot(torch.from_numpy(nib.load(label_ref).get_fdata()).to(torch.int64),
                                    num_classes=12).permute(2, 3, 0, 1).numpy()
        label_metrics = eval_results_labels(label_pred_data, label_ref_data)
        img_metric_dict[img_ref] = img_metrics
        label_metric_dict[label_ref] = label_metrics
    with open(os.path.join(output_path, "img_results.json"), "w+") as f:
        json.dump(img_metric_dict, f)
    with open(os.path.join(output_path, "label_results.json"), "w+") as f:
        json.dump(label_metric_dict, f)


