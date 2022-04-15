import time
import torch
import numpy as np
import torch.nn as nn

from Utils.tester import Tester
from Model.model2 import VxmDenseV3
from torch.utils.data import DataLoader
from Dataset.dataset3d import Dataset3D


batch_size = 1
device = "cuda:0"
kcv_round = 0
path_prefix = "/g/data/nk53/jz6401/dixon_template"
output_path = f"{path_prefix}/Warped_pairs"
dataset_t = Dataset3D(img_path=f"{path_prefix}/kcv/partitions_test_x_{kcv_round}_kcv.txt",
                      label_path=f"{path_prefix}/kcv/partitions_test_y_{kcv_round}_kcv.txt",
                      template_file=f"{path_prefix}/template_updated.nii.gz",
                      template_file_label=f"{path_prefix}/template_label_updated.nii.gz")
loader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=False)
modelG = VxmDenseV3(3, ndims_c=4, inshape=(128, 240, 240),
                    nb_unet_features=[
                       [32, 32, 32, 32],
                       [32, 32, 32, 32, 32, 16],
                   ], bidir=True).cuda()
modelG.load_state_dict(torch.load(f"{path_prefix}/test_outputExp/VxmDenseV3_kcv_epoch_40_0.pth"))
tester = Tester(netG=modelG, output_path=output_path, loader_t=loader_t,
                dataset_t=dataset_t, device=device)
tester.test_run()
tester.save_results()
