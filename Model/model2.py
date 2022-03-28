import torch
import numpy as np
import torch.nn as nn
import voxelmorph.torch.layers as layers

from torch.distributions.normal import Normal
from voxelmorph.py.utils import default_unet_features
from voxelmorph.torch.modelio import LoadableModel, store_config_args


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, ndims, ndims_c, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = ndims_c * 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += ndims_c * 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class VxmDenseV3(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 ndims,
                 ndims_c,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.features = Unet(
            ndims=ndims,
            ndims_c=ndims_c,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow_img_fw = Conv(self.features.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow_img_fw.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_img_fw.weight.shape))
        self.flow_img_fw.bias = nn.Parameter(torch.zeros(self.flow_img_fw.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # flow field for image
        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize_img = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize_img = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate_img = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def register_tensor(self, flow_field, moving_tensor, fixed_tensor, registration):
        # resize flow for integration
        pos_flow = flow_field
        if self.resize_img:
            pos_flow = self.resize_img(pos_flow)

        preint_flow = pos_flow.clone()

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate_img:
            pos_flow = self.integrate_img(pos_flow)
            neg_flow = self.integrate_img(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize_img:
                pos_flow = self.fullsize_img(pos_flow)
                neg_flow = self.fullsize_img(neg_flow) if self.bidir else None

        # warp image with flow field
        moved_tensor = self.transformer(moving_tensor, pos_flow)
        moved_fixed_tensor = self.transformer(fixed_tensor, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (moved_tensor, moved_fixed_tensor, preint_flow) if self.bidir else \
                (moved_tensor, preint_flow)
        else:
            return moved_tensor, preint_flow  # pos_flow

    def forward(self, moving_img, fixed_img, moving_label, fixed_label, registration=False):
        """
        Parameters:
            moving_img: Source image tensor.
            fixed_img: Target image tensor.
            fixed_label: Source label tensor.
            moving_label: Target label tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        x_fw = torch.cat([moving_img, fixed_img], dim=1)
        features_fw = self.features(x_fw)

        # transform into flow field
        flow_field_fw = self.flow_img_fw(features_fw)
        moved_img, moved_fixed_img, img_flow_fw = self.register_tensor(flow_field_fw,
                                                                       moving_tensor=moving_img,
                                                                       fixed_tensor=fixed_img,
                                                                       registration=registration)
        flow_field_label_fw = flow_field_fw
        moved_label, moved_fixed_label, label_flow_fw = self.register_tensor(flow_field_label_fw,
                                                                             moving_tensor=moving_label,
                                                                             fixed_tensor=fixed_label,
                                                                             registration=registration)

        return {
            "moved_img": moved_img,
            "fixed_img_moved": moved_fixed_img,
            "moved_label": moved_label,
            "fixed_label_moved": moved_fixed_label,
            "img_flow": img_flow_fw
        }


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, do_norm=False):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.InstanceNorm3d(out_channels) if do_norm else None
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self, in_channel, feature_dim):
        super(Discriminator, self).__init__()

        self.down_path = nn.Sequential(
            ConvBlock(3, in_channel, feature_dim, stride=1),
            nn.MaxPool3d(2),
            ConvBlock(3, feature_dim, feature_dim * 2, stride=1, do_norm=True),
            nn.MaxPool3d(2),
            ConvBlock(3, feature_dim * 2, feature_dim * 4, stride=1, do_norm=True),
            nn.MaxPool3d(2),
            ConvBlock(3, feature_dim * 4, feature_dim * 8, stride=1, do_norm=True),
            nn.MaxPool3d(2),
            ConvBlock(3, feature_dim * 8, 1, stride=1, do_norm=True),
            nn.AvgPool3d((4, 15, 15)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.down_path(x)


if __name__ == "__main__":
    model = Discriminator(in_channel=4, feature_dim=32).cuda()
    x = torch.randn(1, 4, 64, 240, 240).cuda()
    results = model(x)
    print(results.size())
