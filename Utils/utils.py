import os
import glob
import pathlib

import torch
import subprocess
import numpy as np
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy import ndimage
from sklearn.model_selection import KFold


def spatial_transform_to(in_path, out_path, target, interp_method="cubic"):
    command = ["mrtransform", "-template", target, in_path, out_path, "-force", "-interp", interp_method]
    subprocess.run(command)


def resample_to(source_img, target_img, output_img):
    mdixon_shape = nib.load(target_img).get_fdata().shape
    command = ["c4d", "-interpolation", "Cubic", source_img, "-resample",
               f"{mdixon_shape[0]}x{mdixon_shape[1]}x{mdixon_shape[2]}x3", "-o", output_img]
    subprocess.run(command)


def resample_dti(path):
    dti_files = sorted(glob.glob(os.path.join(path, f'*.nii.gz')))
    dti_files = sorted([file for file in dti_files if "clean_EV1.nii.gz" in file])

    mdixon_files = sorted(glob.glob(os.path.join(path, f'*.nii.gz')))
    mdixon_files = sorted([file for file in mdixon_files if "mdixon.nii.gz" in file])
    print(f"{len(mdixon_files)} mdixon files found and {len(dti_files)} DTI files found.")
    for d_f, m_f in zip(dti_files, mdixon_files):
        d_base_name = os.path.basename(d_f)
        m_base_name = os.path.basename(m_f)
        assert d_base_name[:4] == m_base_name[:4]
        output_f = d_f.replace("clean_EV1.nii.gz", "resampled_EV1.nii.gz")
        resample_to(source_img=d_f, target_img=m_f, output_img=output_f)


def resample_mdixon(path):
    target = "/media/jiayi/Data/Dataset/MUGGLE_DTI_data/0051_01_mdixon.nii.gz"
    files = sorted(glob.glob(os.path.join(path, f'*.nii.gz')))
    files = [file for file in files if "mdixon.nii.gz" in file and "resampled_mdixon.nii.gz" not in file]
    print(f"{len(files)} mdixon files found.")

    for f in files:
        print(nib.load(f).get_fdata().shape)
        # output_f = f.replace("mdixon.nii.gz", "resampled_mdixon.nii.gz")
        # resample_to(f, output_f, target)


def reset_orient(path):
    files = sorted(glob.glob(os.path.join(path, f'*.nii.gz')))
    mdixons = list()
    eigen_vecs = list()
    labels = list()
    for f in files:
        if "mdixon_0.nii.gz" in f or "mdixon_1.nii.gz" in f or "mdixon_2.nii.gz" in f or "mdixon_3.nii.gz" in f:
            mdixons.append(f)
        if "_EV1_0.nii.gz" in f or "_EV1_1.nii.gz" in f or "_EV1_2.nii.gz" in f or "_EV1_3.nii.gz" in f:
            eigen_vecs.append(f)
        if "segmentation_0.nii.gz" in f or "segmentation_1.nii.gz" in f or "segmentation_2.nii.gz" in f or "segmentation_3.nii.gz" in f:
            labels.append(f)
    assert len(mdixons) == len(eigen_vecs) == len(labels) == 80
    # for f in mdixons:
    #     print(f"Doing {f}.")
    #     do_re_orient(f)
    # for f in eigen_vecs:
    #     print(f"Doing {f}.")
    #     do_re_orient(f)
    for f in labels:
        print(f"Doing {f}.")
        do_re_orient(f)


def do_re_orient(file):
    command = ["c3d", file, "-orient", "RPI", "-o", file]
    subprocess.run(command)


def resample_mdixon_labels(path):
    target = "/media/jiayi/Data/Dataset/MUGGLE_DTI_data/0051_01_mdixon_segmentation.nii.gz"
    files = sorted(glob.glob(os.path.join(path, f'*.nii.gz')))
    files = [file for file in files if "mdixon_segmentation.nii.gz" in file]
    print(f"{len(files)} mdixon files found.")
    assert len(files) == 20
    for f in files:
        output_f = f.replace("mdixon_segmentation.nii.gz", "resampled_mdixon_segmentation.nii.gz")
        spatial_transform_to(f, output_f, target, interp_method="nearest")


def labels_randomized(real_data_batch, real_data_labels, fake_data_batch, fake_data_labels):
    cls_labels = torch.cat([real_data_labels, fake_data_labels], dim=0)
    data_batch_mixed = torch.cat([real_data_batch, fake_data_batch], dim=0)
    rand_idx = torch.randperm(real_data_batch.size()[0] + fake_data_batch.size()[0])
    cls_labels = cls_labels[rand_idx]
    data_batch_mixed = data_batch_mixed[rand_idx]
    return cls_labels, data_batch_mixed


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    dydx = dydx.view(dydx.size()[0], -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def loss_filter(mask, device="cuda"):
    _list = []
    for i, m in enumerate(mask):
        if torch.any(m == 1):
            _list.append(i)
    index = torch.tensor(_list, dtype=torch.long).to(device)
    return index


def read_n_permute(img):
    img = torch.from_numpy(nib.load(img).get_fdata())
    if len(img.size()) == 3:
        img = torch.unsqueeze(img, dim=-1)
    return do_permute(img)


def do_permute(img):
    return img.permute(3, 0, 1, 2)


def load_from_f(file_path: str):
    data = list()
    with open(file_path, "r") as f:
        for line in f:
            data.append(line.rstrip())
    return sorted(data)


def get_affine():
    return tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.1),
                         degrees=2,
                         translation=2,
                         isotropic=True)
    ])


def fix_labels(label_map):
    label_map[label_map == 4] = 3
    for i in range(5, 15):
        label_map[label_map == i] = i - 1
    return label_map


def fix_labels_all(path):
    labels = sorted(glob.glob(os.path.join(path, "*.nii.gz")))
    for label in labels:
        label_obj = nib.load(label)
        label_data = label_obj.get_fdata()
        label_data = fix_labels(label_data)
        label_data[label_data >= 12] = 0
        nib.save(nib.Nifti1Image(label_data, label_obj.affine), label)


def flip_data(data_path, label_path):
    ident_list = ["0046",
                  "0047",
                  "0007",
                  "0036",
                  "0037",
                  "0063",
                  "0072",
                  "0105",
                  "0057"]
    images = sorted(glob.glob(os.path.join(data_path, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))
    for img, label in zip(images, labels):
        to_flip = False
        for ident in ident_list:
            if ident in img:
                print(f"Found {img} to flip")
                to_flip = True
                break
        if to_flip:
            # img_data = nib.load(img).get_fdata()
            # img_data = np.flip(img_data, axis=0)
            # nib.save(nib.Nifti1Image(img_data, nib.load(img).affine),
            #          img)

            label_data = nib.load(label).get_fdata()
            label_data = np.flip(label_data, axis=0)
            nib.save(nib.Nifti1Image(label_data, nib.load(label).affine),
                     label)


def augment_data(img_path, label_path, new_img_path, new_label_path):
    images = sorted(glob.glob(os.path.join(img_path, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))
    images = [f for f in images if "_r.nii.gz" in f]
    labels = [f for f in labels if "_r.nii.gz" in f]
    num_augmentation_iter = 4
    assert len(images) == len(labels) == 20
    for i in range(num_augmentation_iter):
        t_fn = get_affine()
        for img, label in zip(images, labels):
            if i == 0:
                print(f"Augmenting {img}")
            new_img_name = os.path.join(new_img_path, os.path.basename(img.replace(".nii.gz", f"_{i}.nii.gz")))
            new_label_name = os.path.join(new_label_path, os.path.basename(label.replace(".nii.gz", f"_{i}.nii.gz")))
            img_data = tio.Subject(
                mdixon=tio.ScalarImage(tensor=read_n_permute(img)),
                mdixon_label=tio.LabelMap(tensor=read_n_permute(label))
            )
            aug_data = t_fn(img_data)

            mdixon_data, mdixon_affine = torch.squeeze(aug_data["mdixon"].data, dim=0).numpy(), \
                                         aug_data["mdixon"].affine
            mdixon_data = np.flip(np.transpose(mdixon_data, (1, 2, 3, 0)), axis=0)
            label_data, label_affine = torch.squeeze(aug_data["mdixon_label"].data, dim=0).numpy(), \
                                       aug_data["mdixon_label"].affine
            label_data = np.flip(label_data, axis=0)

            nib.save(nib.Nifti1Image(mdixon_data, mdixon_affine), new_img_name)
            nib.save(nib.Nifti1Image(label_data, label_affine), new_label_name)


def binarize_scans(label_root):
    labels = sorted(glob.glob(os.path.join(label_root, f'*.nii.gz')))
    assert len(labels) == 20

    for label in labels:
        label_b_name = label.replace("_mask.nii.gz", ".nii.gz")
        label_data = nib.load(label)
        # label_m = label_data.get_fdata()
        # label_m[label_m >= 13] = 0
        # label_m[label_m >= 1] = 1
        assert np.all(np.equal(np.unique(label_data.get_fdata()), np.array([0, 1])))
        nib.save(nib.Nifti1Image(label_data.get_fdata(), label_data.affine), label_b_name)


def produce_fibre_orientations(root_path):
    files = sorted(glob.glob(os.path.join(root_path, "*.nii.gz")))
    dwi_scans = [f for f in files if "DTI_regrid" in f]
    dwi_scans = [f for f in dwi_scans if "DTI_regrid.nii.gz" not in f]
    for dwi_scan in dwi_scans:
        basename = os.path.basename(dwi_scan)
        prefix = basename[:7]
        bvec = os.path.join(root_path, f"{prefix}_DTI_clean.bvec")
        bval = os.path.join(root_path, f"{prefix}_DTI_clean.bval")
        tensor_file = dwi_scan.replace(".nii.gz", "_tensor.mif")
        mask = dwi_scan.replace("DTI_regrid", "mdixon_segmentation_binary")
        vector = dwi_scan.replace("DTI_regrid", "ev1_matrix")
        subprocess.run(["dwi2tensor", dwi_scan, "-mask",
                        mask,
                        "-fslgrad", bvec, bval,
                        tensor_file])
        subprocess.run(["tensor2metric", "-vector", vector, "-num", "1",
                        "-modulate", "none", tensor_file])


def produce_fibre_orientations_resampled(root_path):
    files = sorted(glob.glob(os.path.join(root_path, f'*.nii.gz')))
    dwi_scans = [f for f in files if "DTI_clean.nii.gz" in f]
    print(len(dwi_scans))
    assert len(dwi_scans) == 20
    for dwi_scan in dwi_scans:
        subprocess.run(["mrgrid", dwi_scan, "regrid", "-template", dwi_scan.replace("DTI_clean", "mdixon"),
                        dwi_scan.replace("DTI_clean", "DTI_regrid"), "-force"])
        subprocess.run(["dwi2tensor", dwi_scan.replace("DTI_clean", "DTI_regrid"), "-mask",
                        dwi_scan.replace("DTI_clean", "mdixon_segmentation_binary"),
                        "-fslgrad", dwi_scan.replace("nii.gz", "bvec"), dwi_scan.replace("nii.gz", "bval"),
                        dwi_scan.replace("DTI_clean.nii.gz", "Tensor.mif"), "-force"])
        subprocess.run(["tensor2metric", "-vector", dwi_scan.replace("DTI_clean", "ev1_matrix"), "-num", "1",
                        "-modulate", "none", dwi_scan.replace("DTI_clean.nii.gz", "Tensor.mif"), "-force"])


def write_f(file_name: str, data: list, aug_path: str, num_iter: int = 4, is_test=False):
    def collect_aug_data():
        img_aug_ = list()
        for f_ in data:
            basename = os.path.basename(f_)
            img_aug_.append(os.path.join(aug_path, basename))
            if is_test:
                continue
            for i in range(num_iter):
                b_name = basename.replace("_r.nii.gz", f"_r_{i}.nii.gz")
                img_aug_.append(os.path.join(aug_path, b_name))
        return img_aug_

    img_aug = collect_aug_data()
    with open(file_name, "w+") as f:
        print(*img_aug, sep="\n", file=f)


def make_dir_if_none(tar_dir) -> str:
    path = pathlib.Path(tar_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return tar_dir


def get_val_from(train_x, train_y, num_items):
    val_x_list, val_y_list = list(), list()
    for _ in range(num_items):
        idx = np.random.randint(low=0, high=len(train_x) - 1)
        val_x_list.append(train_x.pop(idx))
        val_y_list.append(train_y.pop(idx))
    return val_x_list, val_y_list, train_x, train_y


def load_data(img_path, label_path):
    img_files = sorted(glob.glob(os.path.join(img_path, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))

    return np.array(img_files), np.array(label_files)


def generate_kcv_folds(img_path,
                       label_path,
                       aug_img_path,
                       aug_label_path,
                       f_path,
                       num_folds=5):
    images, labels = load_data(img_path, label_path)
    kf = KFold(n_splits=num_folds, random_state=42,
               shuffle=True)
    kcv_path = make_dir_if_none(os.path.join(f_path, "kcv"))

    for i, (train_idx, test_idx) in enumerate(kf.split(images)):
        train_x, test_x = images[train_idx], images[test_idx]
        train_y, test_y = labels[train_idx], labels[test_idx]
        assert len(train_x) == len(train_y)
        num_files = len(train_x)
        num_val = int(0.1 * num_files)
        val_x, val_y, train_x, train_y = get_val_from(train_x.tolist(), train_y.tolist(),
                                                      num_val)
        train_x_file = os.path.join(kcv_path, f"partitions_train_x_{i}_kcv.txt")
        write_f(train_x_file, train_x, aug_img_path)
        train_y_file = os.path.join(kcv_path, f"partitions_train_y_{i}_kcv.txt")
        write_f(train_y_file, train_y, aug_label_path)
        test_x_file = os.path.join(kcv_path, f"partitions_test_x_{i}_kcv.txt")
        write_f(test_x_file, test_x, aug_img_path, is_test=True)
        test_y_file = os.path.join(kcv_path, f"partitions_test_y_{i}_kcv.txt")
        write_f(test_y_file, test_y, aug_label_path, is_test=True)
        val_x_file = os.path.join(kcv_path, f"partitions_val_x_{i}_kcv.txt")
        write_f(val_x_file, val_x, aug_img_path)
        val_y_file = os.path.join(kcv_path, f"partitions_val_y_{i}_kcv.txt")
        write_f(val_y_file, val_y, aug_label_path)


def save_model_plot_loss(model, output_path,
                         loss_t,
                         loss_v,
                         recon_t,
                         recon_v,
                         seg_t,
                         seg_v,
                         flow_t,
                         flow_v,
                         num_epochs,
                         kcv_round,
                         save_periodic=False,
                         save_weights=True,
                         weights_name=None,
                         netD_model=None,
                         netG_t=None,
                         netG_v=None,
                         netD_real_t=None,
                         netD_real_v=None,
                         net_D_fake_t=None,
                         net_D_fake_v=None):
    model_name = type(model).__name__
    ident = "" if not save_periodic else f"epoch_{num_epochs}"
    weights_name = f"{model_name}_kcv_{ident}_{kcv_round}.pth" if weights_name is None else weights_name

    if save_weights:
        torch.save(model.state_dict(), os.path.join(output_path, weights_name))
        if netD_model is not None:
            torch.save(netD_model.state_dict(), os.path.join(output_path,
                                                             type(netD_model).__name__))
        print(f"Model weights saved at epoch {num_epochs}")

    torch.save(model.state_dict(), os.path.join(output_path, weights_name))
    print(f"Model weights saved to {output_path} with the name: {weights_name}")

    plot_graph = f"{model_name}_kcv_{kcv_round}.png"
    plot_loss_multiple(loss_t=loss_t,
                       loss_v=loss_v,
                       recon_t=recon_t,
                       recon_v=recon_v,
                       seg_t=seg_t,
                       seg_v=seg_v,
                       flow_t=flow_t,
                       flow_v=flow_v,
                       num_epochs=num_epochs,
                       path_plot=os.path.join(output_path, plot_graph),
                       netG_t=netG_t,
                       netG_v=netG_v,
                       netD_real_t=netD_real_t,
                       netD_real_v=netD_real_v,
                       net_D_fake_t=net_D_fake_t,
                       net_D_fake_v=net_D_fake_v)
    print(f"The loss graph was saved to {output_path} with the name: {plot_graph}")


def plot_loss_multiple(loss_t,
                       loss_v,
                       recon_t,
                       recon_v,
                       seg_t,
                       seg_v,
                       flow_t,
                       flow_v,
                       num_epochs,
                       path_plot,
                       netG_t=None,
                       netG_v=None,
                       netD_real_t=None,
                       netD_real_v=None,
                       net_D_fake_t=None,
                       net_D_fake_v=None):
    x_ticks = np.arange(1, num_epochs + 1, 1)
    plt.figure()
    plt.title("Loss Values")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(x_ticks, loss_t, label="Training loss (Overall)")
    plt.plot(x_ticks, loss_v, label="Val loss (Overall)")
    plt.plot(x_ticks, recon_t, label="Recon (Training)")
    plt.plot(x_ticks, recon_v, label="Recon (Val)")
    plt.plot(x_ticks, seg_t, label="Seg (Training)")
    plt.plot(x_ticks, seg_v, label="Seg (Val)")
    plt.plot(x_ticks, flow_t, label="Flow (Training)")
    plt.plot(x_ticks, flow_v, label="Flow (Val)")
    if netG_t is not None:
        plt.plot(x_ticks, netG_t, label="NetG (Training)")
        plt.plot(x_ticks, netG_v, label="NetG (Val)")
        plt.plot(x_ticks, netD_real_t, label="NetD (Training, real)")
        plt.plot(x_ticks, netD_real_v, label="NetD (Val, real)")
        plt.plot(x_ticks, net_D_fake_t, label="NetD (Training, fake)")
        plt.plot(x_ticks, net_D_fake_v, label="NetD (Val, fake)")
    plt.legend()
    plt.savefig(path_plot)
    plt.close()


def plot_loss_one(angle_v,
                  num_epochs,
                  path_plot):
    x_ticks = np.arange(1, num_epochs + 1, 1)
    plt.figure()
    plt.title("Loss Values")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(x_ticks, angle_v, label="Angle loss (Validation)")
    plt.legend()
    plt.savefig(path_plot)
    plt.close()


def make_switch_to_Gadi(root_path, to_gadi=True):
    files = glob.glob(os.path.join(root_path, '*.txt'))
    for file in files:
        old_data = load_from_f(file)
        new_data = list()
        for d in old_data:
            if to_gadi:
                new_data.append(d.replace("/media/jiayi/Data/Dataset/dixon_template",
                                          "/g/data/nk53/jz6401/dixon_template"))
            else:
                new_data.append(d.replace("/g/data/nk53/jz6401/dixon_template",
                                          "/media/jiayi/Data/Dataset/dixon_template"))
        with open(file, "w+") as f:
            print(*new_data, sep="\n", file=f)


def dot_products_vectors_abs(a, b, epsilon, permuted_dim):
    a = a.permute(*permuted_dim)
    b = b.permute(*permuted_dim)
    assert a.size()[-1] == b.size()[-1] == 3

    a = torch.flatten(a, start_dim=0, end_dim=2)
    b = torch.flatten(b, start_dim=0, end_dim=2)
    non_zero_rows_b = torch.abs(b).sum(dim=1) != -100
    # remove zero entries from both lists of vectors
    # deliberately using the indices from the ref vectors
    # this is because our predicted vectors may contain more zero'd entries caused by wrong predictions
    a = a[non_zero_rows_b]
    b = b[non_zero_rows_b]
    assert a.size() == b.size()
    a = a.reshape(a.size()[0], 1, 3)
    b = b.reshape(b.size()[0], 3, 1)
    dot_products = torch.abs(torch.matmul(a, b).squeeze(1).squeeze(-1))
    return dot_products


def resize_scans(path, interp_method="cubic"):
    files = sorted(glob.glob(os.path.join(path, "*.nii.gz")))
    for f in files:
        f_basename = os.path.basename(f)
        new_name = f_basename.replace(".nii.gz", "_r.nii.gz")
        cmd = ["mrgrid", "-size", "240,240,512", "-interp", interp_method, f, "regrid", os.path.join(path, new_name),
               "-force"]
        subprocess.run(cmd)


def align_labels(img_path, label_path, new_label_path, transform_path):
    labels = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))
    for seg_file in labels:
        basename = os.path.basename(seg_file)
        output_file = os.path.join(new_label_path, basename)
        img_file = os.path.join(img_path, basename.replace("_segmentation", ""))
        transform_file = os.path.join(transform_path, basename.replace("_segmentation.nii.gz", ".txt"))
        cmd = ["mrtransform", seg_file, output_file, "-linear", transform_file, "-template",
               img_file, "-interp", "nearest", "-force"]
        subprocess.run(cmd)
        cmd = ["c3d", output_file, "-split", "-foreach", "-smooth", "3mm", "-endfor", "-merge",
               "-o", output_file]
        subprocess.run(cmd)


def average_labels(label_path, template_path):
    labels = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))
    labels = [f for f in labels if "_r.nii.gz" in f]
    print(len(labels))
    affine = nib.load(template_path).affine
    all_labels = np.zeros((20, 240, 240, 512, 12))
    for i, f in enumerate(labels):
        f_data = nib.load(f).get_fdata()
        f_data = F.one_hot(torch.from_numpy(f_data).to(torch.int64), num_classes=12)
        all_labels[i] = f_data.numpy()
    average_label = np.mean(all_labels, axis=0)
    nib.save(nib.Nifti1Image(average_label, affine), "/media/jiayi/Data/Dataset/dixon_template/template_label.nii.gz")
    average_label_view = torch.argmax(torch.from_numpy(average_label), dim=-1)
    nib.save(nib.Nifti1Image(average_label_view.numpy().astype(np.int8), affine),
             "/media/jiayi/Data/Dataset/dixon_template/template_view.nii.gz")


def majority_voting(label_path, template_path):
    labels = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))
    affine = nib.load(template_path).affine
    all_labels = np.zeros((20, 230, 238, 519))
    result_label = np.zeros((230, 238, 519))
    for i, f in enumerate(labels):
        f_data = nib.load(f).get_fdata()
        all_labels[i] = f_data

    for i in range(230):
        for j in range(238):
            for k in range(519):
                pixels = all_labels[:, i, j, k]
                values, counts = np.unique(pixels, return_counts=True)
                ind = np.argmax(counts)
                result_label[i, j, k] = pixels[ind]

    nib.save(nib.Nifti1Image(result_label, affine),
             os.path.join(label_path, "new_test.nii.gz"))


def get_largest_cc(label):
    label_obj = nib.load(label)
    label_data = torch.from_numpy(label_obj.get_fdata())
    label_data_out = torch.zeros(label_obj.get_fdata().shape)
    label_data = fix_labels(label_data)
    for i in range(12):
        label_data_copy = torch.clone(label_data)
        label_data_copy[label_data_copy != i] = 0
        label_data_copy[label_data_copy == i] = 1
        label_data_copy = label_data_copy.numpy()
        label_im, nb_labels = ndimage.label(label_data_copy)
        sizes = ndimage.sum(label_data_copy, label_im, range(nb_labels + 1))
        mask = sizes == max(sizes)
        label_data_copy = mask[label_im]
        label_data_copy = label_data_copy.astype(np.int32)
        label_data_copy[label_data_copy == 1] = i
        label_data_out += torch.from_numpy(label_data_copy.astype(np.int32))
    nib.save(nib.Nifti1Image(label_data_out.numpy().astype(np.int32), label_obj.affine),
             "/media/jiayi/Data/Dataset/dixon_template/template_label_test.nii.gz")


def resave_files(path):
    data = sorted(glob.glob(os.path.join(path, "*.nii.gz")))
    for f in data:
        f_obj = nib.load(f)
        f_data = np.nan_to_num(f_obj.get_fdata(), nan=0.0)
        nib.save(nib.Nifti1Image(f_data, f_obj.affine), f)


def fix_data(img_path, label_path):
    imgs = sorted(glob.glob(os.path.join(img_path, f'*.nii.gz')))
    labels = sorted(glob.glob(os.path.join(label_path, f'*.nii.gz')))

    for img, label in zip(imgs, labels):
        f_obj = nib.load(img)
        f_img = f_obj.get_fdata()

        f_obj = nib.load(label)
        f_label = f_obj.get_fdata()
        print(np.unique(f_label))


if __name__ == "__main__":
    resave_files("/g/data/nk53/jz6401/dixon_template/transformed_aug")
    resave_files("/g/data/nk53/jz6401/dixon_template/transformed_label_aug")
