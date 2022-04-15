import torch
import numpy as np
import nibabel as nib

from Utils.utils import save_model_plot_loss


class Trainer(object):
    def __init__(self, netS, optimiserS, output_path, kcv_round, lambda_val,
                 epoch_num, dice_lossfn, grad_loss, loader_t, loader_v, device, scheduler):
        self.loss_t, self.loss_v = list(), list()
        self.seg_t_list, self.seg_v_list = list(), list()
        self.flow_t_list, self.flow_v_list = list(), list()

        self.output_path = output_path
        self.kcv_round = kcv_round
        self.netS = netS
        self.optimiserS = optimiserS
        self.dice_lossfn = dice_lossfn
        self.grad_loss = grad_loss
        self.loader_t = loader_t
        self.loader_v = loader_v
        self.device = device
        self.epoch_num = epoch_num
        self.lambda_val = lambda_val
        self.scheduler = scheduler

    def segment(self, batch_data):
        moving_atlas, fixed_img, moving_atlas_label = batch_data["moving_atlas"].to(self.device), \
                                                      batch_data["fixed_img"].to(self.device), \
                                                      batch_data["moving_atlas_label"].to(self.device)
        results = self.netS(x=torch.cat([moving_atlas, fixed_img], dim=1),
                            atlas_label=moving_atlas_label)
        return results

    def calculate_seg_loss(self, results, batch_data):
        dice_loss = self.dice_lossfn(results["label"], batch_data["fixed_labels"].to(self.device))
        flow_img_loss = self.grad_loss.loss(results["flow"], results["flow"])
        loss_val = dice_loss + self.lambda_val * flow_img_loss
        return {
            "loss_val": loss_val,
            "flow_loss": flow_img_loss,
            "dice_loss": dice_loss
        }

    def train_generator_one_step(self, batch_data):
        self.optimiserS.zero_grad()
        self.netS.train()
        results = self.segment(batch_data)
        loss_vals = self.calculate_seg_loss(results, batch_data)
        loss_vals["loss_val"].backward()
        self.optimiserS.step()
        return loss_vals

    def eval_generator_one_step(self, batch_data):
        with torch.no_grad():
            self.netS.eval()
            results = self.segment(batch_data)
            loss_vals = self.calculate_seg_loss(results, batch_data)
        self.scheduler.step()
        return loss_vals

    def train(self, epoch):
        loss_t_val = 0.
        seg_t = 0.
        flow_t = 0.
        index = 0
        for index, batch_data in enumerate(self.loader_t):
            loss_vals = self.train_generator_one_step(batch_data)

            loss_t_val += loss_vals["loss_val"].item()
            seg_t += loss_vals["dice_loss"].item()
            flow_t += loss_vals["flow_loss"].item()

        self.loss_t.append(loss_t_val / (index + 1))
        self.seg_t_list.append(seg_t / (index + 1))
        self.flow_t_list.append(flow_t / (index + 1))

        print(f"Loss value break-down for epoch: {epoch + 1}\n"
              f"Total -> {self.loss_t[-1] : .3f}\n"
              f"Seg -> {self.seg_t_list[-1]: .3f}\n"
              f"Flow -> {self.flow_t_list[-1]: .3f}\n")

    def eval(self, epoch):
        loss_v_val = 0.
        seg_v = 0.
        flow_v = 0.
        index = 0

        for index, batch_data in enumerate(self.loader_t):
            loss_vals = self.eval_generator_one_step(batch_data)

            loss_v_val += loss_vals["loss_val"].item()
            seg_v += loss_vals["seg_loss"].item()
            flow_v += loss_vals["flow_img_loss"].item()

        self.loss_v.append(loss_v_val / (index + 1))
        self.seg_v_list.append(seg_v / (index + 1))
        self.flow_v_list.append(flow_v / (index + 1))

        print(f"Loss value break-down for epoch: {epoch + 1} (eval)\n"
              f"Total -> {self.loss_t[-1] : .3f}\n"
              f"Seg -> {self.seg_t_list[-1]: .3f}\n"
              f"Flow -> {self.flow_t_list[-1]: .3f}\n")


    def save_plot(self, epoch, save_periodic, save_weights):
        save_model_plot_loss(model=self.netS,
                             netD_model=self.netD,
                             output_path=self.output_path,
                             loss_t=self.loss_t,
                             loss_v=self.loss_v,
                             recon_t=self.recon_t_list,
                             recon_v=self.recon_v_list,
                             seg_t=self.seg_t_list,
                             seg_v=self.seg_v_list,
                             flow_t=self.flow_t_list,
                             flow_v=self.flow_v_list,
                             num_epochs=epoch + 1 if epoch != self.epoch_num else self.epoch_num,
                             kcv_round=self.kcv_round,
                             save_periodic=save_periodic,
                             save_weights=save_weights,
                             netG_t=self.gan_g_t_list,
                             netG_v=self.gan_g_v_list,
                             net_D_fake_v=self.gan_d_fake_v_list,
                             net_D_fake_t=self.gan_d_fake_t_list,
                             netD_real_t=self.gan_d_real_t_list,
                             netD_real_v=self.gan_d_real_v_list)