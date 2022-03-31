import torch
import numpy as np
import nibabel as nib

from Loss.gan_loss import GAN_Loss
from Utils.utils import save_model_plot_loss


class Trainer(object):
    def __init__(self, netG, netD, optimiserG, optimiserD, intensity_lossfn, output_path, kcv_round,
                 epoch_num, dice_lossfn, grad_loss, loader_t, loader_v, loader_seq, dataset_seq, device):
        self.loss_t, self.loss_v = list(), list()
        self.recon_t_list, self.recon_v_list = list(), list()
        self.seg_t_list, self.seg_v_list = list(), list()
        self.flow_t_list, self.flow_v_list = list(), list()
        self.gan_g_t_list, self.gan_g_v_list = list(), list()
        self.gan_d_fake_t_list, self.gan_d_fake_v_list = list(), list()
        self.gan_d_real_t_list, self.gan_d_real_v_list = list(), list()

        self.output_path = output_path
        self.kcv_round = kcv_round
        self.netG = netG
        self.netD = netD
        self.optimiserG = optimiserG
        self.optimiserD = optimiserD
        self.intensity_lossfn = intensity_lossfn
        self.dice_lossfn = dice_lossfn
        self.grad_loss = grad_loss
        self.loader_t = loader_t
        self.loader_v = loader_v
        self.loader_seq = loader_seq
        self.dataset_seq = dataset_seq
        self.device = device
        self.gan_loss = GAN_Loss()
        self.epoch_num = epoch_num

    def generate(self, batch_data):
        moving_atlas, fixed_img, moving_atlas_label, fixed_labels = batch_data["moving_atlas"].to(self.device), \
                                                                    batch_data["fixed_img"].to(self.device), \
                                                                    batch_data["moving_atlas_label"].to(self.device), \
                                                                    batch_data["fixed_labels"].to(self.device)
        results = self.netG(moving_img=moving_atlas,
                            fixed_img=fixed_img,
                            moving_label=moving_atlas_label,
                            fixed_label=fixed_labels)
        return results

    def calculate_netG_loss(self, results, batch_data):
        recon_loss_forward = self.intensity_lossfn(results["moved_img"], batch_data["fixed_img"].to(self.device))
        recon_loss_backward = self.intensity_lossfn(results["fixed_img_moved"],
                                                    batch_data["moving_atlas"].to(self.device))
        recon_loss = (recon_loss_forward + recon_loss_backward) / 2
        seg_to_atlas = self.intensity_lossfn(results["moved_label"], batch_data["fixed_labels"].to(self.device))
        atlas_to_seg = self.intensity_lossfn(results["fixed_label_moved"],
                                             batch_data["moving_atlas_label"].to(self.device))
        seg_dice = self.dice_lossfn(torch.clamp(results["moved_label"], min=0, max=1),
                                    batch_data["fixed_labels"].to(self.device),
                                    do_act=False)
        atlas_dice = self.dice_lossfn(torch.clamp(results["fixed_label_moved"], min=0, max=1),
                                      batch_data["moving_atlas_label"].to(self.device), do_act=False)
        dice_loss = (seg_dice + atlas_dice) / 2
        seg_loss = (seg_to_atlas + atlas_to_seg) / 2 + dice_loss
        flow_img_loss = self.grad_loss.loss(results["img_flow"], results["img_flow"])

        fake_probs, real_probs = self.discriminate(results["moved_img"], batch_data["fixed_img"].to(self.device))
        # training generator, treating the fake as real
        gan_loss = self.gan_loss(fake_probs, is_real=True)
        loss_val = recon_loss + seg_loss + flow_img_loss + gan_loss
        return {
            "loss_val": loss_val,
            "recon_loss": recon_loss,
            "seg_loss": seg_loss,
            "flow_img_loss": flow_img_loss,
            "gan_loss": gan_loss
        }

    def discriminate(self, fake_image, real_image):
        fake_real = torch.cat([fake_image, real_image], dim=0)
        probs = self.netD(fake_real)
        fake_probs, real_probs = probs[:probs.size()[0] // 2, ...], probs[probs.size()[0] // 2:, ...]
        return fake_probs, real_probs

    def train_generator_one_step(self, batch_data):
        self.optimiserG.zero_grad()
        self.netG.train()
        results = self.generate(batch_data)
        loss_vals = self.calculate_netG_loss(results, batch_data)
        loss_vals["loss_val"].backward()
        self.optimiserG.step()
        return loss_vals

    def eval_generator_one_step(self, batch_data):
        with torch.no_grad():
            self.netG.eval()
            results = self.generate(batch_data)
            loss_vals = self.calculate_netG_loss(results, batch_data)
        return loss_vals

    def train_discriminator_one_step(self, batch_data):
        self.optimiserD.zero_grad()
        self.netD.train()
        with torch.no_grad():
            results = self.generate(batch_data)
            fake_img = results["moved_img"].detach()
            fake_img.requires_grad_()
        fake_probs, real_probs = self.discriminate(fake_img, batch_data["fixed_img"].to(self.device))
        fake_loss = self.gan_loss(fake_probs, is_real=False)
        real_loss = self.gan_loss(real_probs, is_real=True)
        loss_val = (fake_loss + real_loss) / 2
        loss_val.backward()
        self.optimiserD.step()
        return fake_loss, real_loss

    def eval_discriminator_one_step(self, batch_data):
        with torch.no_grad():
            self.netD.eval()
            self.netG.eval()
            results = self.generate(batch_data)
            fake_img = results["moved_img"].detach()
            fake_probs, real_probs = self.discriminate(fake_img, batch_data["fixed_img"].to(self.device))
            fake_loss = self.gan_loss(fake_probs, is_real=False)
            real_loss = self.gan_loss(real_probs, is_real=True)
        return fake_loss, real_loss

    def train(self, epoch):
        loss_t_val = 0.
        recon_t = 0.
        seg_t = 0.
        flow_img_t = 0.
        gan_g_t = 0.
        gan_d_fake_t, gan_d_real_t = 0., 0.
        index = 0
        for index, batch_data in enumerate(self.loader_t):
            loss_vals = self.train_generator_one_step(batch_data)

            loss_t_val += loss_vals["loss_val"].item()
            recon_t += loss_vals["recon_loss"].item()
            seg_t += loss_vals["seg_loss"].item()
            flow_img_t += loss_vals["flow_img_loss"].item()
            gan_g_t += loss_vals["gan_loss"].item()

            # train discriminator
            fake, real = self.train_discriminator_one_step(batch_data)
            gan_d_fake_t += fake.item()
            gan_d_real_t += real.item()

        self.loss_t.append(loss_t_val / (index + 1))
        self.recon_t_list.append(recon_t / (index + 1))
        self.seg_t_list.append(seg_t / (index + 1))
        self.flow_t_list.append(flow_img_t / (index + 1))
        self.gan_g_t_list.append(gan_g_t / (index + 1))
        self.gan_d_real_t_list.append(gan_d_real_t / (index + 1))
        self.gan_d_fake_t_list.append(gan_d_fake_t / (index + 1))

        print(f"Loss value break-down for epoch: {epoch + 1}\n"
              f"Total -> {self.loss_t[-1] : .3f}\n"
              f"Recon -> {self.recon_t_list[-1]: .3f}\n"
              f"Seg -> {self.seg_t_list[-1]: .3f}\n"
              f"Flow -> {self.flow_t_list[-1]: .3f}\n"
              f"Gan (netG) -> {self.gan_g_t_list[-1]: .3f}\n"
              f"Gan (netD, real) -> {self.gan_d_real_t_list[-1]: .3f}\n"
              f"Gan (netD, fake) -> {self.gan_d_fake_t_list[-1]: .3f}")

    def eval(self, epoch):
        loss_v_val = 0.
        recon_v = 0.
        seg_v = 0.
        flow_img_v = 0.
        gan_g_v = 0.
        index = 0
        gan_d_fake_v, gan_d_real_v = 0., 0.

        for index, batch_data in enumerate(self.loader_t):
            loss_vals = self.eval_generator_one_step(batch_data)

            loss_v_val += loss_vals["loss_val"].item()
            recon_v += loss_vals["recon_loss"].item()
            seg_v += loss_vals["seg_loss"].item()
            flow_img_v += loss_vals["flow_img_loss"].item()
            gan_g_v += loss_vals["gan_loss"].item()

            # train discriminator
            fake, real = self.eval_discriminator_one_step(batch_data)
            gan_d_fake_v += fake.item()
            gan_d_real_v += real.item()

        self.loss_v.append(loss_v_val / (index + 1))
        self.recon_v_list.append(recon_v / (index + 1))
        self.seg_v_list.append(seg_v / (index + 1))
        self.flow_v_list.append(flow_img_v / (index + 1))
        self.gan_g_v_list.append(gan_g_v / (index + 1))
        self.gan_d_real_v_list.append(gan_d_real_v / (index + 1))
        self.gan_d_fake_v_list.append(gan_d_fake_v / (index + 1))

        print(f"Loss value break-down for epoch: {epoch + 1} (eval)\n"
              f"Total -> {self.loss_t[-1] : .3f}\n"
              f"Recon -> {self.recon_t_list[-1]: .3f}\n"
              f"Seg -> {self.seg_t_list[-1]: .3f}\n"
              f"Flow -> {self.flow_t_list[-1]: .3f}\n"
              f"Gan (netG) -> {self.gan_g_t_list[-1]: .3f}\n"
              f"Gan (netD, real) -> {self.gan_d_real_v_list[-1]: .3f}\n"
              f"Gan (netD, fake) -> {self.gan_d_fake_v_list[-1]: .3f}")

    def update_template(self):
        with torch.no_grad():
            for batch_data in self.loader_seq:
                results = self.generate(batch_data)
                self.dataset_seq.insert_warped_img(results["fixed_img_moved"])
                self.dataset_seq.insert_warped_label(results["fixed_label_moved"])
            self.dataset_seq.update_template_img(alpha=1)
            self.dataset_seq.update_template_label(alpha=1)

    def save_templates(self, epoch):
        template, template_affine = self.dataset_seq.retrieve_template()
        template = template.permute(1, 2, 3, 0)
        template_label, template_label_affine = self.dataset_seq.retrieve_template_label()
        template_label = torch.argmax(template_label, dim=0).permute(1, 2, 0)
        nib.save(nib.Nifti1Image(template.numpy(), template_affine),
                 f"{self.output_path}/template_updated_{epoch}.nii.gz")
        nib.save(nib.Nifti1Image(template_label.numpy().astype(np.int8), template_label_affine),
                 f"{self.output_path}/template_label_updated_{epoch}.nii.gz")

    def save_plot(self, epoch, save_periodic, save_weights):
        save_model_plot_loss(model=self.netG,
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