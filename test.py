import os
import glob
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from medpy import metric

# nn.MSELoss()
# img = nib.load("/media/jiayi/Data/Dataset/dixon_template/transformed_aug/0001_01_mdixon_r.nii.gz")
# data = torch.from_numpy(img.get_fdata()).permute(3, 0, 1, 2)
# rescale = tio.RescaleIntensity(out_min_max=(-1, 1))
# data = rescale(tio.Subject(image=tio.ScalarImage(tensor=data)))["image"].data.permute(1, 2, 3, 0).numpy()
# nib.save(nib.Nifti1Image(data, img.affine),
#          "/media/jiayi/Data/Dataset/dixon_template/test.nii.gz")


all_dsc = list()
all_assd = list()

labels = sorted(glob.glob(os.path.join("/media/jiayi/Data/Dataset/dixon_template/transformed_label", "*.nii.gz")))
labels = [label for label in labels if "_r.nii.gz" in label]
label_2 = nib.load("/media/jiayi/Data/Dataset/dixon_template/results/test_outputGAN/template_label_updated_15.nii.gz").get_fdata()
label_2 = torch.unsqueeze(F.one_hot(torch.from_numpy(label_2).to(torch.int64), num_classes=12).permute(3, 2, 0, 1), dim=0)

for label in labels:
    label_1 = nib.load(label).get_fdata()
    label_1 = torch.from_numpy(label_1).to(torch.int64)
    label_1 = torch.unsqueeze(F.one_hot(label_1, num_classes=12).permute(3, 2, 0, 1), dim=0)
    dsc, assd = list(), list()
    for i in range(12):
        d_score = metric.dc(label_2[:, i, ...].numpy(), label_1[:, i, ...].numpy())
        assd_score = metric.assd(label_2[:, i, ...].numpy(), label_1[:, i, ...].numpy(), voxelspacing=1)
        dsc.append(d_score)
        assd.append(assd_score)
    all_dsc.append(np.array(dsc))
    all_assd.append(np.array(assd))
avg_dsc = np.mean(np.array(all_dsc), axis=0)
avg_assd = np.mean(np.array(all_assd), axis=0)
for i, (d, a) in enumerate(zip(avg_dsc, avg_assd)):
    print(f"The DSC for class {i} is {d: .3f}")
    print(f"The ASSD for class {i} is {a: .3f}")
    print()