import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

from medpy import metric
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def eval(pred_img, pred_label, img_name, label_name):
    ref_img = torch.from_numpy(nib.load(img_name).get_fdata()).permute(3, 2, 0, 1).numpy()
    ref_label = F.one_hot(torch.from_numpy(nib.load(label_name).get_fdata()).permute(2, 0, 1).to(torch.int64),
                          num_classes=12).permute(3, 0, 1, 2).numpy()

    label_result = eval_results_labels(pred_label, ref_label)
    img_result = eval_results_imgs(pred_img, ref_img)
    return label_result, img_result


def eval_results_imgs(pred, ref):
    ssim_list, psnr_list = list(), list()
    ssim_list.append(ssim(pred, ref))
    psnr_list.append(psnr(pred, ref))
    return psnr_list, ssim_list


def eval_results_labels(pred: np.ndarray, ref: np.ndarray):
    dsc_list, assd_list, verror_list = list(), list(), list()
    for i in range(12):
        dsc_list.append(metric.dc(pred[:, i, ...], ref[:, i, ...]))
        assd_list.append(metric.assd(pred[:, i, ...], ref[:, i, ...], voxelspacing=1))
        verror_list.append(find_vol_error(pred, ref, 12, True))
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

    phy_vol = 0.4 * 0.4 * 5
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
