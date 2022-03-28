import time
import torch
import numpy as np
import torch.nn as nn
import nibabel as nib

from Model.model2 import VxmDenseV3
from torch.utils.data import DataLoader
from Dataset.dataset2d import Dataset2D
from Loss.loss import Grad2D
from torch.optim.lr_scheduler import ExponentialLR
from Loss.loss import WeightedDiceLoss

from Utils.utils import save_model_plot_loss

batch_size = 128
device = "cuda:0"
epoch_num = 40
kcv_round = 0
path_prefix = "/g/data/nk53/jz6401/dixon_template"
dataset_t = Dataset2D(img_path=f"{path_prefix}/kcv/partitions_train_x_{kcv_round}_kcv.txt",
                      label_path=f"{path_prefix}/kcv/partitions_train_y_{kcv_round}_kcv.txt",
                      template_file=f"{path_prefix}/template.nii.gz",
                      template_file_label=f"{path_prefix}/template_label.nii.gz")
dataset_seq = Dataset2D(img_path=f"{path_prefix}/kcv/partitions_train_x_{kcv_round}_kcv.txt",
                        label_path=f"{path_prefix}/kcv/partitions_train_y_{kcv_round}_kcv.txt",
                        template_file=f"{path_prefix}/template.nii.gz",
                        template_file_label=f"{path_prefix}/template_label.nii.gz",
                        for_alignment=True)
loader_seq = DataLoader(dataset_seq, batch_size=batch_size, shuffle=False)
loader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True)
dataset_v = Dataset2D(img_path=f"{path_prefix}/kcv/partitions_val_x_{kcv_round}_kcv.txt",
                      label_path=f"{path_prefix}/kcv/partitions_val_y_{kcv_round}_kcv.txt",
                      template_file=f"{path_prefix}/template.nii.gz",
                      template_file_label=f"{path_prefix}/template_label.nii.gz")

file_t, file_v = dataset_t.get_files(), dataset_v.get_files()
assert len(np.intersect1d(file_t, file_v)) == 0
for f in file_t:
    print(f)
print()
for f in file_v:
    print(f)
print("*" * 100)
output_path = f"{path_prefix}/test_output2D"
loader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=True)
# input image has 4 channels, 2 for classification
model = VxmDenseV3(2, ndims_c=4, inshape=(240, 240),
                   nb_unet_features=[
                       [32, 32, 32, 32],
                       [32, 32, 32, 32, 32, 16],
                   ], bidir=True).cuda()

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
scheduler = ExponentialLR(optimiser, gamma=0.9)

seg_lossfn = WeightedDiceLoss(num_classes=12, num_outputs=1, weight=0.1).to(device)
intensity_lossfn = nn.MSELoss().to(device)
grad_loss = Grad2D(penalty="l2")

start_time = time.time()
loss_t, loss_v = list(), list()
recon_t_list, recon_v_list = list(), list()
seg_t_list, seg_v_list = list(), list()
flow_t_list, flow_v_list = list(), list()

print('start training...')

best_val_loss = np.inf
for epoch in range(epoch_num):
    loss_t_val, loss_v_val = 0., 0.
    recon_t, recon_v = 0., 0.
    seg_t, seg_v = 0., 0.
    flow_img_t, flow_img_v = 0., 0.
    index = 0
    model.train()
    for index, batch_data in enumerate(loader_t):
        optimiser.zero_grad()
        moving_atlas, fixed_img, moving_atlas_label, fixed_labels = batch_data["moving_atlas"].to(device), \
                                                                    batch_data["fixed_img"].to(device), \
                                                                    batch_data["moving_atlas_label"].to(device), \
                                                                    batch_data["fixed_labels"].to(device)
        results = model(moving_img=moving_atlas,
                        fixed_img=fixed_img,
                        moving_label=moving_atlas_label,
                        fixed_label=fixed_labels)

        recon_loss_forward = intensity_lossfn(results["moved_img"], fixed_img)
        recon_loss_backward = intensity_lossfn(results["fixed_img_moved"], moving_atlas)
        recon_loss = (recon_loss_forward + recon_loss_backward) / 2
        seg_to_atlas = intensity_lossfn(results["moved_label"], fixed_labels)
        atlas_to_seg = intensity_lossfn(results["fixed_label_moved"], moving_atlas_label)
        seg_dice = seg_lossfn(torch.clamp(results["moved_label"], min=0, max=1), fixed_labels, do_act=False)
        atlas_dice = seg_lossfn(torch.clamp(results["fixed_label_moved"], min=0, max=1),
                                moving_atlas_label, do_act=False)
        dice_loss = (seg_dice + atlas_dice) / 2
        seg_loss = (seg_to_atlas + atlas_to_seg) / 2 + dice_loss
        flow_img_loss = grad_loss.loss(results["img_flow"], results["img_flow"])
        loss_val = recon_loss + seg_loss + flow_img_loss
        loss_t_val += loss_val.item()
        recon_t += recon_loss.item()
        seg_t += seg_loss.item()
        flow_img_t += flow_img_loss.item()
        loss_val.backward()
        optimiser.step()
    loss_t.append(loss_t_val / (index + 1))
    recon_t_list.append(recon_t / (index + 1))
    seg_t_list.append(seg_t / (index + 1))
    flow_t_list.append(flow_img_t / (index + 1))
    print(f"Loss value break-down for epoch: {epoch + 1}\n"
          f"Total -> {loss_t[-1] : .3f}\n"
          f"Recon -> {recon_t_list[-1]: .3f}\n"
          f"Seg -> {seg_t_list[-1]: .3f}\n"
          f"Flow -> {flow_t_list[-1]: .3f}\n")

    # eval
    with torch.no_grad():
        model.eval()
        for index, batch_data in enumerate(loader_v):
            moving_atlas, fixed_img, moving_atlas_label, fixed_labels = batch_data["moving_atlas"].to(device), \
                                                                        batch_data["fixed_img"].to(device), \
                                                                        batch_data["moving_atlas_label"].to(device), \
                                                                        batch_data["fixed_labels"].to(device)
            results = model(moving_img=moving_atlas,
                            fixed_img=fixed_img,
                            moving_label=moving_atlas_label,
                            fixed_label=fixed_labels)

            recon_loss_forward = intensity_lossfn(results["moved_img"], fixed_img)
            recon_loss_backward = intensity_lossfn(results["fixed_img_moved"], moving_atlas)
            recon_loss = (recon_loss_forward + recon_loss_backward) / 2
            seg_to_atlas = intensity_lossfn(results["moved_label"], fixed_labels)
            atlas_to_seg = intensity_lossfn(results["fixed_label_moved"], moving_atlas_label)
            seg_dice = seg_lossfn(torch.clamp(results["moved_label"], min=0, max=1), fixed_labels, do_act=False)
            atlas_dice = seg_lossfn(torch.clamp(results["fixed_label_moved"], min=0, max=1),
                                    moving_atlas_label, do_act=False)
            dice_loss = (seg_dice + atlas_dice) / 2
            seg_loss = (seg_to_atlas + atlas_to_seg) / 2 + dice_loss
            flow_img_loss = grad_loss.loss(results["img_flow"], results["img_flow"])
            loss_val = recon_loss + seg_loss + flow_img_loss
            loss_v_val += loss_val.item()
            recon_v += recon_loss.item()
            seg_v += seg_loss.item()
            flow_img_v += flow_img_loss.item()
    loss_v.append(loss_v_val / (index + 1))
    recon_v_list.append(recon_v / (index + 1))
    seg_v_list.append(seg_v / (index + 1))
    flow_v_list.append(flow_img_v / (index + 1))
    print(f"Loss value break-down for epoch: {epoch + 1} (val)\n"
          f"Total -> {loss_v[-1] : .3f}\n"
          f"Recon -> {recon_v_list[-1]: .3f}\n"
          f"Seg -> {seg_v_list[-1]: .3f}\n"
          f"Flow -> {flow_v_list[-1]: .3f}\n")

    # update the template
    with torch.no_grad():
        model.eval()
        for batch_data in loader_seq:
            moving_atlas, fixed_img, moving_atlas_label, fixed_labels = batch_data["moving_atlas"].to(device), \
                                                                        batch_data["fixed_img"].to(device), \
                                                                        batch_data["moving_atlas_label"].to(device), \
                                                                        batch_data["fixed_labels"].to(device)
            index = batch_data["index"]
            results = model(moving_img=moving_atlas,
                            fixed_img=fixed_img,
                            moving_label=moving_atlas_label,
                            fixed_label=fixed_labels)

            dataset_seq.insert_warped_img(results["fixed_img_moved"])
            dataset_seq.insert_warped_label(results["fixed_label_moved"])
        dataset_seq.update_template_img(alpha=0.9)
        dataset_seq.update_template_label(alpha=0.9)

        if epoch % 5 == 0 and epoch > 0 or (epoch + 1) == epoch_num:
            template, template_affine = dataset_seq.retrieve_template()
            template = template.permute(1, 2, 3, 0)
            template_label, template_label_affine = dataset_seq.retrieve_template_label()
            template_label = torch.argmax(template_label, dim=0).permute(1, 2, 0)
            nib.save(nib.Nifti1Image(template.numpy(), template_affine),
                     f"{output_path}/template_updated_{epoch}.nii.gz")
            nib.save(nib.Nifti1Image(template_label.numpy().astype(np.int8), template_label_affine),
                     f"{output_path}/template_label_updated_{epoch}.nii.gz")

            save_model_plot_loss(model=model,
                                 output_path=output_path,
                                 loss_t=loss_t,
                                 loss_v=loss_v,
                                 recon_t=recon_t_list,
                                 recon_v=recon_v_list,
                                 seg_t=seg_t_list,
                                 seg_v=seg_v_list,
                                 flow_t=flow_t_list,
                                 flow_v=flow_v_list,
                                 num_epochs=epoch + 1,
                                 kcv_round=kcv_round,
                                 save_periodic=True,
                                 save_weights=True)

save_model_plot_loss(model=model,
                     output_path=output_path,
                     loss_t=loss_t,
                     loss_v=loss_v,
                     recon_t=recon_t_list,
                     recon_v=recon_v_list,
                     seg_t=seg_t_list,
                     seg_v=seg_v_list,
                     flow_t=flow_t_list,
                     flow_v=flow_v_list,
                     num_epochs=epoch_num,
                     kcv_round=kcv_round,
                     save_periodic=False,
                     save_weights=True)
