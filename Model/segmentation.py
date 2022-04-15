import voxelmorph.torch.layers as layers

from torch.distributions import Normal
from axial_attention.axial_attention import *


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class InLayer(nn.Module):
    def __init__(self, img_dim, dim, patch_size):
        super(InLayer, self).__init__()
        self.ops = nn.Sequential(
            nn.Conv3d(img_dim, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU()
        )

    def forward(self, x):
        return self.ops(x)


class AxialBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, downsample=None, heads=8,
                 base_width=64, norm_layer=None):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_dim * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(in_dim, width)
        self.bn1 = norm_layer(width)
        self.att_block = AxialAttention(dim=width,
                                        dim_index=1,
                                        num_dimensions=3,
                                        heads=heads,
                                        sum_axial_out=True)
        self.conv_up = conv1x1(width, out_dim)
        self.bn2 = norm_layer(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.att_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channel):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=in_channel,
                               padding=1, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=in_channel, out_channels=in_channel,
                               padding=1, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=in_channel)
        self.norm2 = nn.InstanceNorm3d(num_features=in_channel)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x + identity)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderBlock, self).__init__()
        self.conv = DoubleConv(in_dim)
        self.up = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )

    def forward(self, x, f=None):
        if f is not None:
            x = torch.cat([x, f], dim=1)
        x = self.conv(x)
        return self.up(x)


class Decoder(nn.Module):
    def __init__(self, f_dim):
        super(Decoder, self).__init__()

        self.de1 = DecoderBlock(f_dim, f_dim // 2)
        self.de2 = DecoderBlock(f_dim, f_dim // 4)
        self.de3 = DecoderBlock(f_dim // 2, f_dim // 8)
        self.de4 = DecoderBlock(f_dim // 4, f_dim // 16)

        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(f_dim, f_dim // 2, kernel_size=1),
            nn.ReLU(True)
        )

        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="trilinear", align_corners=False),
            nn.Conv3d(f_dim, f_dim // 4, kernel_size=1),
            nn.ReLU(True)
        )

        self.up_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="trilinear", align_corners=False),
            nn.Conv3d(f_dim, f_dim // 8, kernel_size=1),
            nn.ReLU(True)
        )

        self.up_conv4 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="trilinear", align_corners=False),
            nn.Conv3d(f_dim, f_dim // 16, kernel_size=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        up1 = self.de1(x)
        up1_f = self.up_conv1(x)
        up2 = self.de2(up1, up1_f)
        up2_f = self.up_conv2(x)
        up3 = self.de3(up2, up2_f)
        up3_f = self.up_conv3(x)
        up4 = self.de4(up3, up3_f)
        up4_f = self.up_conv4(x)
        return up4, up4_f


class Flow(nn.Module):
    def __init__(self, f_dim, in_shape, ndims, int_steps=7, int_downsize=2):
        super(Flow, self).__init__()

        self.flow_generator = nn.Conv3d(f_dim, 3, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow_generator.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_generator.weight.shape))
        self.flow_generator.bias = nn.Parameter(torch.zeros(self.flow_generator.bias.shape))

        resize = int_steps > 0 and int_downsize > 1
        self.resize_img = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize_img = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None


        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in in_shape]
        self.integrate_img = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(in_shape)

    def forward(self, features, moving_label):
        pos_flow = self.flow_generator(features)
        if self.resize_img:
            pos_flow = self.resize_img(pos_flow)

        # integrate to produce diffeomorphic warp
        if self.integrate_img:
            pos_flow = self.integrate_img(pos_flow)

            # resize to final resolution
            if self.fullsize_img:
                pos_flow = self.fullsize_img(pos_flow)

        # warp image with flow field
        moved_label = self.transformer(moving_label, pos_flow)

        return moved_label, flow


class Segmentor(nn.Module):
    def __init__(self, in_dim, f_dim_encoder, patch_size, in_shape, n_dim,
                 depth, heads, int_steps=7, int_downsize=2):
        super(Segmentor, self).__init__()

        self.input_layer = InLayer(img_dim=in_dim, dim=f_dim_encoder, patch_size=patch_size)

        self.pos_emb = nn.Identity()
        # use pos embedding later
        # AxialPositionalEmbedding(f_dim_encoder, shape=shape)

        encoder_layers = list()
        for _ in range(depth):
            encoder_layers.append(AxialBlock(f_dim_encoder, f_dim_encoder,
                                             norm_layer=nn.InstanceNorm3d, heads=heads))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = Decoder(f_dim_encoder)

        self.flow = Flow(f_dim_encoder // 8, in_shape=in_shape, ndims=n_dim, int_downsize=int_downsize,
                         int_steps=int_steps)



    def forward(self, x, atlas_label):
        x = self.input_layer(x)
        x_pos = self.pos_emb(x)
        x_encoder = self.encoder(x_pos)
        x_decoder, additional_features = self.decoder(x_encoder)
        moved_label, flow = self.flow(torch.cat([x_decoder, additional_features], dim=1), atlas_label)

        return {
            "label": moved_label,
            "flow": flow
        }

