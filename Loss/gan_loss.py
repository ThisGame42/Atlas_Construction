import torch
import torch.nn as nn
import torch.nn.functional as F


class GAN_Loss(nn.Module):
    def __init__(self):
        super(GAN_Loss, self).__init__()
        real_label = 1.
        fake_label = 0.
        self.fake_label_tensor = torch.FloatTensor(1).fill_(fake_label)
        self.fake_label_tensor.requires_grad_(False)
        self.real_label_tensor = torch.FloatTensor(1).fill_(real_label)
        self.real_label_tensor.requires_grad_(False)

    def get_target_tensor(self, input_data, is_real):
        if is_real:
            return self.real_label_tensor.expand_as(input_data)
        else:
            return self.fake_label_tensor.expand_as(input_data)

    def __call__(self, input_data, is_real):
        # when training G
        # loss(fake_data, is_real=True)
        # when training D
        # loss(fake_data, is_real=False)
        # loss(real_data, is_real=True)
        target_tensor = self.get_target_tensor(input_data, is_real)
        loss_val = F.binary_cross_entropy_with_logits(input_data, target_tensor.cuda())
        return loss_val
