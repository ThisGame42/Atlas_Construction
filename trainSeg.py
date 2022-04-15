import time
import torch
import numpy as np
import torch.nn as nn

from Utils.trainer_seg import Trainer
from Model.segmentation import Segmentor
from Loss.loss import WeightedDiceLoss
from voxelmorph.torch.losses import Grad
from torch.utils.data import DataLoader
from Dataset.dataset3d import Dataset3D
from torch.optim.lr_scheduler import ExponentialLR


batch_size = 1
device = "cuda:0"
epoch_num = 1
kcv_round = 0
path_prefix = "/media/jiayi/Data/Dataset/dixon_template"
dataset_t = Dataset3D(img_path=f"{path_prefix}/kcv/partitions_train_x_{kcv_round}_kcv.txt",
                      label_path=f"{path_prefix}/kcv/partitions_train_y_{kcv_round}_kcv.txt",
                      template_file=f"{path_prefix}/template_updated.nii.gz",
                      template_file_label=f"{path_prefix}/template_label_updated.nii.gz")
loader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True)
dataset_v = Dataset3D(img_path=f"{path_prefix}/kcv/partitions_val_x_{kcv_round}_kcv.txt",
                      label_path=f"{path_prefix}/kcv/partitions_val_y_{kcv_round}_kcv.txt",
                      template_file=f"{path_prefix}/template_updated.nii.gz",
                      template_file_label=f"{path_prefix}/template_label_updated.nii.gz")

loader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=True)
file_t, file_v = dataset_t.get_files(), dataset_v.get_files()
assert len(np.intersect1d(file_t, file_v)) == 0
for f in file_t:
    print(f)
print()
for f in file_v:
    print(f)
print("*" * 100)
output_path = f"{path_prefix}/test_outputGAN"
# input image has 4 channels, 2 for classification
modelS = Segmentor(in_dim= 4 * 2, f_dim_encoder=1024, patch_size=16,
                   n_dim=3, depth=8, heads=32, in_shape=(128, 240, 240)).cuda()
optimiser = torch.optim.Adam(modelS.parameters(), lr=1e-3, betas=(0.5, 0.9))

scheduler = ExponentialLR(optimiser, gamma=0.9)

seg_lossfn = WeightedDiceLoss(num_classes=12, num_outputs=1, weight=0.1).to(device)
grad_loss = Grad(penalty="l2")

start_time = time.time()
trainer = Trainer(dice_lossfn=seg_lossfn, netS=modelS, optimiserS=optimiser, scheduler=scheduler,
                  epoch_num=epoch_num, grad_loss=grad_loss, loader_t=loader_t, loader_v=loader_v,
                  device=device, kcv_round=kcv_round, lambda_val=800,
                  output_path=output_path)

for epoch in range(epoch_num):
    trainer.train(epoch)
    trainer.eval(epoch)
    if epoch % 5 == 0:
        trainer.save_plot(epoch, save_periodic=True, save_weights=True)
trainer.save_plot(epoch_num, save_periodic=False, save_weights=True)
