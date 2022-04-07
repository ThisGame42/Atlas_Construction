import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

from medpy import metric
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def eval(pred_img, pred_label, ref_img, ref_label):
    label_result = eval_results_labels(pred_label, ref_label)
    img_result = eval_results_imgs(pred_img, ref_img)
    return label_result, img_result


def eval_results_imgs(pred, ref):
    pred = np.transpose(pred, (1, 2, 3, 0))
    ref = np.transpose(ref, (1, 2, 3, 0))
    return ssim(pred, ref, multichannel=True), psnr(pred, ref)


def eval_results_labels(pred: np.ndarray, ref: np.ndarray):
    dsc_list, assd_list = list(), list()
    for i in range(12):
        dsc_list.append(metric.dc(pred[:, i, ...], ref[:, i, ...]))
        assd_list.append(metric.assd(pred[:, i, ...], ref[:, i, ...], voxelspacing=1))
    verror_list = find_vol_error(pred, ref, 12, True).tolist()
    return dsc_list, assd_list, verror_list


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
            pred_this_c, gt_this_c = pred[idx, ...], gt[idx, ...]
        else:
            pred_this_c, gt_this_c = pred[:, idx, ...], gt[:, idx, ...]
        pred_count, gt_count = get_vol_count(torch.from_numpy(pred_this_c),
                                             torch.from_numpy(gt_this_c))
        pred_vol, gt_vol = (pred_count * phy_vol) / 1000, (gt_count * phy_vol) / 1000
        vol_e[idx] = abs(pred_vol - gt_vol)
    return vol_e
