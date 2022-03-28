import torch
import numpy as np
import torch.nn as nn


class WeightedDiceLoss(nn.Module):
    """
        Dice loss function that optimises directly against the DSC score.
    """

    def __init__(self,
                 num_classes: int = 14,
                 num_outputs: int = 1,
                 weight: float = 1.):

        super(WeightedDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.probs_fn = nn.Softmax(dim=1)
        self.dice_outputs = dict()
        self.weight = weight
        for i in range(num_outputs):
            self.dice_outputs[i] = list()

    def forward(self,
                raw_pred: torch.Tensor,
                ref_labels: torch.Tensor,
                dsc_id=0,
                do_act=False,
                ignore_check=True):
        probs = self.probs_fn(raw_pred) if do_act else raw_pred
        dsc_scores = compute_DSC(predictions=probs,
                                 ref_labels=ref_labels,
                                 ignore_check=ignore_check)
        self.dice_outputs[dsc_id].append(dsc_scores.detach().cpu().numpy())

        loss_val = 1 - torch.mean(dsc_scores)
        return self.weight * loss_val

    def show_dsc(self):
        for k, v in self.dice_outputs.items():
            dsc = np.mean(np.array(v), axis=0)
            print(f"The mean DSC of all classes except for the BACKGROUND is {np.mean(dsc[1:])}, ID => {k}.")

    def clear_dsc(self):
        for k in self.dice_outputs.keys():
            self.dice_outputs[k] = list()


def compute_DSC(predictions: torch.Tensor,
                ref_labels: torch.Tensor,
                epsilon=1.e-6,
                use_vnet_dice=True,
                ignore_check=False) -> torch.Tensor:
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given a multi channel input and target.
    Assumes the input is a normalized probability, e.g. the result of Sigmoid or Softmax function.

        This function computes the VNet Dice if use_vnet_dice is True, and computes the standard Dice otherwise.

    :param predictions: Output of a model. Assumed to be in probabilities (i.e. in [0, 1], not in logits!)
    :param ref_labels: One-hot encoded reference labels
    :param epsilon: Prevents division by zero error
    :param use_vnet_dice: See above
    :param ignore_check: True to perform NO checks on the validity of the inputs.
    :return: A torch Tensor containing average (over the batch) Dice coefficients for each class.
    """
    if not ignore_check:
        print("We are checking if the dimensions of the input data match, are you sure you want to do this?")
        assert predictions.size() == ref_labels.size(), \
            "The predictions and the reference labels are not in the same shape"

        assert predictions.dim() == 5 or predictions.dim() == 4, \
            f"Only 4D or 5D predictions are supported"

        assert torch.max(predictions) <= 1, "Invalid values in predictions detected"
        assert torch.min(predictions) >= 0, "Invalid values in predictions detected"
        assert torch.max(ref_labels) <= 1 and torch.min(ref_labels) >= 0, \
            f"Invalid values in reference labels detected"

    prob_flatten = flatten(predictions)
    ref_flatten = flatten(ref_labels).float()

    # compute per channel Dice Coefficient
    intersect = (prob_flatten * ref_flatten).sum(-1)

    # here we can use standard dice (input + target).sum(-1)
    # or extension (see V-Net) (input^2 + target^2).sum(-1)
    if use_vnet_dice:
        denominator = (prob_flatten * prob_flatten).sum(-1) + (ref_flatten * ref_flatten).sum(-1)
    else:
        denominator = prob_flatten.sum(-1) + ref_flatten.sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flattens a given tensor into a two-dimensional tensor. Class channel becomes the first dimension and
    other dimensions are squashed into one.

       3D: (Batch, Class, Depth, Height, Width) -> (C, B * D * H * W)\n
       2D: (B, C, H, W) -> (C, B * H * W)
    """

    num_classes = tensor.size()[1]
    # new dimension order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(num_classes, -1)


class NormalizedCrossCorrelationLoss(nn.Module):
    """
    The ncc loss: 1- NCC
    """

    def __init__(self, epsilon=1.e-6):
        super(NormalizedCrossCorrelationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_x: torch.tensor, target_x: torch.tensor):
        input_x = input_x.view(input_x.shape[0], -1)
        target_x = target_x.view(target_x.shape[0], -1)
        input_minus_mean = input_x - torch.mean(input_x, 1, keepdim=True)
        target_minus_mean = target_x - torch.mean(target_x, 1, keepdim=True)
        nccSqr = (input_minus_mean * target_minus_mean).mean(1) / (
                torch.sqrt(torch.clamp((input_minus_mean ** 2).mean(1), min=self.epsilon)) * torch.sqrt(
            torch.clamp((target_minus_mean ** 2).mean(1), min=self.epsilon)
        ))
        nccSqr = nccSqr.mean()
        loss_val = 1 - nccSqr
        return loss_val


class BendingEnergyLoss(nn.Module):
    """
    regularization loss of bending energy of a 3d deformation field
    """

    def __init__(self, norm='L2', spacing=(1, 1, 1), normalize=True,
                 device="cuda:0"):
        super(BendingEnergyLoss, self).__init__()
        self.norm = norm
        self.spacing = torch.tensor(spacing).float().to(device)
        self.normalize = normalize
        if self.normalize:
            self.spacing /= self.spacing.min()

    def forward(self, input_x):
        """
        :param input_x: Nx3xDxHxW
        :return:
        """
        spatial_dims = torch.tensor(input_x.shape[2:]).float().to(input_x.device)
        if self.normalize:
            spatial_dims /= spatial_dims.min()

        # according to
        # f''(x) = [f(x+h) + f(x-h) - 2f(x)] / h^2
        # f_{x, y}(x, y) = [df(x+h, y+k) + df(x-h, y-k) - df(x+h, y-k) - df(x-h, y+k)] / 2hk

        ddx = torch.abs(
            input_x[:, :, 2:, 1:-1, 1:-1] + input_x[:, :, :-2, 1:-1, 1:-1] - 2 * input_x[:, :, 1:-1, 1:-1, 1:-1]) \
            .view(input_x.shape[0], input_x.shape[1], -1)

        ddy = torch.abs(
            input_x[:, :, 1:-1, 2:, 1:-1] + input_x[:, :, 1:-1, :-2, 1:-1] - 2 * input_x[:, :, 1:-1, 1:-1, 1:-1]) \
            .view(input_x.shape[0], input_x.shape[1], -1)

        ddz = torch.abs(
            input_x[:, :, 1:-1, 1:-1, 2:] + input_x[:, :, 1:-1, 1:-1, :-2] - 2 * input_x[:, :, 1:-1, 1:-1, 1:-1]) \
            .view(input_x.shape[0], input_x.shape[1], -1)

        dxdy = torch.abs(input_x[:, :, 2:, 2:, 1:-1] + input_x[:, :, :-2, :-2, 1:-1] -
                         input_x[:, :, 2:, :-2, 1:-1] - input_x[:, :, :-2, 2:, 1:-1]).view(input_x.shape[0],
                                                                                           input_x.shape[1], -1)

        dydz = torch.abs(input_x[:, :, 1:-1, 2:, 2:] + input_x[:, :, 1:-1, :-2, :-2] -
                         input_x[:, :, 1:-1, 2:, :-2] - input_x[:, :, 1:-1, :-2, 2:]).view(input_x.shape[0],
                                                                                           input_x.shape[1], -1)

        dxdz = torch.abs(input_x[:, :, 2:, 1:-1, 2:] + input_x[:, :, :-2, 1:-1, :-2] -
                         input_x[:, :, 2:, 1:-1, :-2] - input_x[:, :, :-2, 1:-1, 2:]).view(input_x.shape[0],
                                                                                           input_x.shape[1], -1)

        if self.norm == 'L2':
            ddx = (ddx ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0] ** 2)) ** 2
            ddy = (ddy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1] ** 2)) ** 2
            ddz = (ddz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2] ** 2)) ** 2
            dxdy = (dxdy ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[0] * self.spacing[1])) ** 2
            dydz = (dydz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[1] * self.spacing[2])) ** 2
            dxdz = (dxdz ** 2).mean(2) * (spatial_dims * self.spacing / (self.spacing[2] * self.spacing[0])) ** 2

        d = (ddx.mean() + ddy.mean() + ddz.mean() + 2 * dxdy.mean() + 2 * dydz.mean() + 2 * dxdz.mean()) / 9.0
        return d


class Grad2D:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
