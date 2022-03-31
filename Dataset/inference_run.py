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
epoch_num = 1
kcv_round = 0
path_prefix = "/media/jiayi/Data/Dataset/dixon_template"
output_path = f"{path_prefix}/test_output"
dataset_t = Dataset3D(img_path=f"{path_prefix}/kcv/partitions_test_x_{kcv_round}_kcv.txt",
                      label_path=f"{path_prefix}/kcv/partitions_test_y_{kcv_round}_kcv.txt",
                      # change the template below
                      template_file=f"{path_prefix}/template.nii.gz",
                      template_file_label=f"{path_prefix}/template_label.nii.gz")
loader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=False)
modelG = VxmDenseV3(3, ndims_c=4, inshape=(64, 240, 240),
                    nb_unet_features=[
                       [32, 32, 32, 32],
                       [32, 32, 32, 32, 32, 16],
                   ], bidir=True).cuda()
tester = Tester(netG=modelG, output_path=output_path, loader_t=loader_t,
                dataset_t=dataset_t, device="cuda:0")