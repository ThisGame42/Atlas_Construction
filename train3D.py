import time
import torch
import numpy as np
import torch.nn as nn

from Utils.trainer import Trainer
from Model.model2 import VxmDenseV3, Discriminator
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
                      template_file=f"{path_prefix}/template.nii.gz",
                      template_file_label=f"{path_prefix}/template_label.nii.gz")
dataset_seq = Dataset3D(img_path=f"{path_prefix}/kcv/partitions_train_x_{kcv_round}_kcv.txt",
                        label_path=f"{path_prefix}/kcv/partitions_train_y_{kcv_round}_kcv.txt",
                        template_file=f"{path_prefix}/template.nii.gz",
                        template_file_label=f"{path_prefix}/template_label.nii.gz",
                        for_alignment=True)
loader_seq = DataLoader(dataset_seq, batch_size=1, shuffle=False)
loader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True)
dataset_v = Dataset3D(img_path=f"{path_prefix}/kcv/partitions_val_x_{kcv_round}_kcv.txt",
                      label_path=f"{path_prefix}/kcv/partitions_val_y_{kcv_round}_kcv.txt",
                      template_file=f"{path_prefix}/template.nii.gz",
                      template_file_label=f"{path_prefix}/template_label.nii.gz")

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
modelG = VxmDenseV3(3, ndims_c=4, inshape=(64, 240, 240),
                    nb_unet_features=[
                       [32, 32, 32, 32],
                       [32, 32, 32, 32, 32, 16],
                   ], bidir=True).cuda()
modelD = Discriminator(in_channel=4, feature_dim=32).cuda()
optimiserG = torch.optim.Adam(modelG.parameters(), lr=1e-3, betas=(0.5, 0.9))
optimiserD = torch.optim.Adam(modelD.parameters(), lr=1e-3, betas=(0.5, 0.9))

schedulerG = ExponentialLR(optimiserG, gamma=0.9)
schedulerD = ExponentialLR(optimiserD, gamma=0.9)

seg_lossfn = WeightedDiceLoss(num_classes=12, num_outputs=1, weight=0.1).to(device)
intensity_lossfn = nn.MSELoss().to(device)
grad_loss = Grad(penalty="l2")

start_time = time.time()
trainer = Trainer(netG=modelG, netD=modelD, optimiserG=optimiserG,
                  optimiserD=optimiserD, dice_lossfn=seg_lossfn, intensity_lossfn=intensity_lossfn,
                  epoch_num=epoch_num, grad_loss=grad_loss, loader_t=loader_t, loader_v=loader_v,
                  device=device, kcv_round=kcv_round, dataset_seq=dataset_seq, loader_seq=loader_seq,
                  output_path=output_path)

for epoch in range(epoch_num):
    trainer.train(epoch)
    trainer.eval(epoch)
    trainer.update_template()
    if epoch % 5 == 0:
        trainer.save_plot(epoch, save_periodic=True, save_weights=True)
        trainer.save_templates(epoch)
trainer.save_plot(epoch_num, save_periodic=False, save_weights=True)
