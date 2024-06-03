import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def get_activation(activation):
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "softmax":
        return nn.Softmax(dim=1)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "mish":
        return nn.Mish()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


def make_batch(x):
    """
    Make batch dimension if input is not batched
    :param x: input tensor
    :return: input tensor with batch dimension and boolean indicating whether input was already a batch or not.
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        batch = False
    elif len(x.shape) == 4:
        batch = True
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")
    return x, batch


class ConvolutionBlock(nn.Module):
    """
    Convolution block consisting of two convolutional layers with optional batch normalization and
    given activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        batch_normalization=False,
        activation="relu",
    ):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.batch_norm1 = (
            # nn.BatchNorm2d(out_channels)
            # if batch_normalization
            # else nn.Identity()
            nn.InstanceNorm2d(out_channels)
            if batch_normalization
            else nn.Identity()
        )
        self.activation1 = get_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.batch_norm2 = (
            # nn.BatchNorm2d(out_channels)
            # if batch_normalization
            # else nn.Identity()
            nn.InstanceNorm2d(out_channels)
            if batch_normalization
            else nn.Identity()
        )
        self.activation2 = get_activation(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)

        return x


class ConvolutionBlockResidual(nn.Module):
    """
    Convolution block consisting of two convolutional layers with optional batch normalization and
    given activation function, including a residual connection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        batch_normalization=False,
        activation="relu",
    ):
        super(ConvolutionBlockResidual, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.batch_norm1 = (
            nn.BatchNorm2d(out_channels) if batch_normalization else nn.Identity()
        )
        self.activation1 = get_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.batch_norm2 = (
            nn.BatchNorm2d(out_channels) if batch_normalization else nn.Identity()
        )
        self.activation2 = get_activation(activation)

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)

        x += residual
        return x


class unet1(nn.Module):
    """
    The netowrk architecture is based on the U-Net 1 architecture proposed in the paper:
    Leclerc S, Smistad E, Pedrosa J, Ostvik A, Cervenansky F, Espinosa F, Espeland T, Berg EAR, Jodoin PM, Grenier T,
    Lartizien C, Dhooge J, Lovstakken L, Bernard O. Deep Learning for Segmentation Using an Open Large-Scale Dataset
    in 2D Echocardiography. IEEE Trans Med Imaging. 2019 Sep;38(9):2198-2210. doi: 10.1109/TMI.2019.2900516.
    Epub 2019 Feb 22. PMID: 30802851.
    """

    def __init__(
        self,
        input_shape=(1, 256, 256),
        activation_inter_layer="relu",
        normalize_input=False,
        normalize_inter_layer=False,
        use_deep_supervision=False,
        nb_classes=4,
        final_activation="softmax",
    ):
        super(unet1, self).__init__()

        self.normalize_input = (
            nn.BatchNorm2d(input_shape[0]) if normalize_input else nn.Identity()
        )
        self.use_deep_supervision = use_deep_supervision
        if self.use_deep_supervision:
            print("Using deep supervision.")
            self.ds1 = nn.Conv2d(128, nb_classes, kernel_size=(1, 1))
            self.ds2 = nn.Conv2d(128, nb_classes, kernel_size=(1, 1))
            self.ds3 = nn.Conv2d(64, nb_classes, kernel_size=(1, 1))
            self.ds4 = nn.Conv2d(32, nb_classes, kernel_size=(1, 1))
        else:
            print("Not using deep supervision.")

        self.down1 = ConvolutionBlock(
            input_shape[0],
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down2 = ConvolutionBlock(
            32,
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down3 = ConvolutionBlock(
            32,
            64,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down4 = ConvolutionBlock(
            64,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down5 = ConvolutionBlock(
            128,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down6 = ConvolutionBlock(
            128,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        self.up1 = ConvolutionBlock(
            256,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up2 = ConvolutionBlock(
            256,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up3 = ConvolutionBlock(
            192,
            64,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up4 = ConvolutionBlock(
            96,
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up5 = ConvolutionBlock(
            64,
            16,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.seg = nn.Conv2d(16, nb_classes, kernel_size=(1, 1))
        self.final_activation = get_activation(final_activation)

    def forward(self, x):
        x, batch = make_batch(x)

        x = self.normalize_input(x)

        down1_x = self.down1(x)
        x = F.max_pool2d(down1_x, kernel_size=2)
        down2_x = self.down2(x)
        x = F.max_pool2d(down2_x, kernel_size=2)
        down3_x = self.down3(x)
        x = F.max_pool2d(down3_x, kernel_size=2)
        down4_x = self.down4(x)
        x = F.max_pool2d(down4_x, kernel_size=2)
        down5_x = self.down5(x)
        x = F.max_pool2d(down5_x, kernel_size=2)
        down6_x = self.down6(x)

        x = F.interpolate(down6_x, scale_factor=2, mode="nearest")
        x = torch.cat([down5_x, x], dim=1)
        up1_x = self.up1(x)
        x = F.interpolate(up1_x, scale_factor=2, mode="nearest")
        x = torch.cat([down4_x, x], dim=1)
        up2_x = self.up2(x)
        x = F.interpolate(up2_x, scale_factor=2, mode="nearest")
        x = torch.cat([down3_x, x], dim=1)
        up3_x = self.up3(x)
        x = F.interpolate(up3_x, scale_factor=2, mode="nearest")
        x = torch.cat([down2_x, x], dim=1)
        up4_x = self.up4(x)
        x = F.interpolate(up4_x, scale_factor=2, mode="nearest")
        x = torch.cat([down1_x, x], dim=1)
        up5_x = self.up5(x)

        if self.use_deep_supervision:
            ds1_out = self.ds1(up1_x)
            ds1_out = F.interpolate(ds1_out, scale_factor=16, mode="nearest")
            ds1_out = self.final_activation(ds1_out)

            ds2_out = self.ds2(up2_x)
            ds2_out = F.interpolate(ds2_out, scale_factor=8, mode="nearest")
            ds2_out = self.final_activation(ds2_out)

            ds3_out = self.ds3(up3_x)
            ds3_out = F.interpolate(ds3_out, scale_factor=4, mode="nearest")
            ds3_out = self.final_activation(ds3_out)

            ds4_out = self.ds4(up4_x)
            ds4_out = F.interpolate(ds4_out, scale_factor=2, mode="nearest")
            ds4_out = self.final_activation(ds4_out)

        seg_x = self.seg(up5_x)
        out_x = self.final_activation(seg_x)

        if not batch:
            out_x = out_x.squeeze(0)

        if self.use_deep_supervision:
            return out_x, ds1_out, ds2_out, ds3_out, ds4_out

        return out_x


class unet2(nn.Module):

    def __init__(
        self,
        input_shape=(1, 256, 256),
        activation_inter_layer="relu",
        normalize_input=False,
        normalize_inter_layer=False,
        use_deep_supervision=False,
        nb_classes=4,
        final_activation="softmax",
    ):
        super(unet2, self).__init__()

        self.normalize_input = (
            nn.BatchNorm2d(input_shape[0]) if normalize_input else nn.Identity()
        )

        self.use_deep_supervision = use_deep_supervision
        if self.use_deep_supervision:
            print("Using deep supervision.")
            self.ds1 = nn.Conv2d(256, nb_classes, kernel_size=(1, 1))
            self.ds2 = nn.Conv2d(256, nb_classes, kernel_size=(1, 1))
            self.ds3 = nn.Conv2d(128, nb_classes, kernel_size=(1, 1))
            self.ds4 = nn.Conv2d(64, nb_classes, kernel_size=(1, 1))

        self.down1 = ConvolutionBlock(
            input_shape[0],
            64,  # Increased from 32 to 64
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down2 = ConvolutionBlock(
            64,  # Increased from 32 to 64
            64,  # Increased from 32 to 64
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down3 = ConvolutionBlock(
            64,  # Increased from 32 to 64
            128,  # Increased from 64 to 128
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down4 = ConvolutionBlock(
            128,
            256,  # Increased from 128 to 256
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down5 = ConvolutionBlock(
            256,
            256,  # Increased from 128 to 256
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down6 = ConvolutionBlock(
            256,
            512,  # Increased from 128 to 512
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        # Adjusting the channels in the upward path to match the new channel dimensions
        self.up1 = ConvolutionBlock(
            512 + 256,  # Increased from 256 + 128
            256,  # Increased from 128 to 256
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up2 = ConvolutionBlock(
            256 + 256,  # Increased from 256 + 128
            256,  # Increased from 128 to 256
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up3 = ConvolutionBlock(
            256 + 128,  # Increased from 192 + 64
            128,  # Increased from 64 to 128
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up4 = ConvolutionBlock(
            128 + 64,  # Increased from 96 + 32
            64,  # Increased from 32 to 64
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up5 = ConvolutionBlock(
            64 + 64,  # Increased from 64 + 32
            32,  # Increased from 16 to 32
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        # Adjusting the final convolution to match the new channel dimensions
        self.seg = nn.Conv2d(
            32, nb_classes, kernel_size=(1, 1)
        )  # Input channels increased from 16 to 32
        self.final_activation = get_activation(final_activation)

    def forward(self, x):
        x, batch = make_batch(x)

        x = self.normalize_input(x)

        down1_x = self.down1(x)
        x = F.max_pool2d(down1_x, kernel_size=2)
        down2_x = self.down2(x)
        x = F.max_pool2d(down2_x, kernel_size=2)
        down3_x = self.down3(x)
        x = F.max_pool2d(down3_x, kernel_size=2)
        down4_x = self.down4(x)
        x = F.max_pool2d(down4_x, kernel_size=2)
        down5_x = self.down5(x)
        x = F.max_pool2d(down5_x, kernel_size=2)
        down6_x = self.down6(x)

        x = F.interpolate(down6_x, scale_factor=2, mode="nearest")
        x = torch.cat([down5_x, x], dim=1)
        up1_x = self.up1(x)
        x = F.interpolate(up1_x, scale_factor=2, mode="nearest")
        x = torch.cat([down4_x, x], dim=1)
        up2_x = self.up2(x)
        x = F.interpolate(up2_x, scale_factor=2, mode="nearest")
        x = torch.cat([down3_x, x], dim=1)
        up3_x = self.up3(x)
        x = F.interpolate(up3_x, scale_factor=2, mode="nearest")
        x = torch.cat([down2_x, x], dim=1)
        up4_x = self.up4(x)
        x = F.interpolate(up4_x, scale_factor=2, mode="nearest")
        x = torch.cat([down1_x, x], dim=1)
        up5_x = self.up5(x)

        if self.use_deep_supervision:
            ds1_out = self.ds1(up1_x)
            ds1_out = F.interpolate(
                ds1_out, scale_factor=16, mode="bilinear", align_corners=True
            )
            ds2_out = self.ds2(up2_x)
            ds2_out = F.interpolate(
                ds2_out, scale_factor=8, mode="bilinear", align_corners=True
            )
            ds3_out = self.ds3(up3_x)
            ds3_out = F.interpolate(
                ds3_out, scale_factor=4, mode="bilinear", align_corners=True
            )
            ds4_out = self.ds4(up4_x)
            ds4_out = F.interpolate(
                ds4_out, scale_factor=2, mode="bilinear", align_corners=True
            )

        seg_x = self.seg(up5_x)
        out_x = self.final_activation(seg_x)

        if not batch:
            out_x = out_x.squeeze(0)

        if self.use_deep_supervision:
            return out_x, ds1_out, ds2_out, ds3_out, ds4_out

        return out_x


class unet1_res(nn.Module):
    """
    The netowrk architecture is based on the U-Net 1 architecture proposed in the paper:
    Leclerc S, Smistad E, Pedrosa J, Ostvik A, Cervenansky F, Espinosa F, Espeland T, Berg EAR, Jodoin PM, Grenier T,
    Lartizien C, Dhooge J, Lovstakken L, Bernard O. Deep Learning for Segmentation Using an Open Large-Scale Dataset
    in 2D Echocardiography. IEEE Trans Med Imaging. 2019 Sep;38(9):2198-2210. doi: 10.1109/TMI.2019.2900516.
    Epub 2019 Feb 22. PMID: 30802851.
    """

    def __init__(
        self,
        input_shape=(1, 256, 256),
        activation_inter_layer="relu",
        normalize_input=False,
        normalize_inter_layer=False,
        nb_classes=4,
        final_activation="softmax",
        use_deep_supervision=False,
    ):
        super().__init__()

        self.normalize_input = (
            nn.BatchNorm2d(input_shape[0]) if normalize_input else nn.Identity()
        )

        self.down1 = ConvolutionBlockResidual(
            input_shape[0],
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down2 = ConvolutionBlockResidual(
            32,
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down3 = ConvolutionBlockResidual(
            32,
            64,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down4 = ConvolutionBlockResidual(
            64,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down5 = ConvolutionBlockResidual(
            128,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down6 = ConvolutionBlockResidual(
            128,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        self.up1 = ConvolutionBlockResidual(
            256,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up2 = ConvolutionBlockResidual(
            256,
            128,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up3 = ConvolutionBlockResidual(
            192,
            64,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up4 = ConvolutionBlockResidual(
            96,
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up5 = ConvolutionBlockResidual(
            64,
            16,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.seg = nn.Conv2d(16, nb_classes, kernel_size=(1, 1))
        self.final_activation = get_activation(final_activation)

    def forward(self, x):
        x, batch = make_batch(x)

        x = self.normalize_input(x)

        down1_x = self.down1(x)
        x = F.max_pool2d(down1_x, kernel_size=2)
        down2_x = self.down2(x)
        x = F.max_pool2d(down2_x, kernel_size=2)
        down3_x = self.down3(x)
        x = F.max_pool2d(down3_x, kernel_size=2)
        down4_x = self.down4(x)
        x = F.max_pool2d(down4_x, kernel_size=2)
        down5_x = self.down5(x)
        x = F.max_pool2d(down5_x, kernel_size=2)
        down6_x = self.down6(x)

        x = F.interpolate(down6_x, scale_factor=2, mode="nearest")
        x = torch.cat([down5_x, x], dim=1)
        up1_x = self.up1(x)
        x = F.interpolate(up1_x, scale_factor=2, mode="nearest")
        x = torch.cat([down4_x, x], dim=1)
        up2_x = self.up2(x)
        x = F.interpolate(up2_x, scale_factor=2, mode="nearest")
        x = torch.cat([down3_x, x], dim=1)
        up3_x = self.up3(x)
        x = F.interpolate(up3_x, scale_factor=2, mode="nearest")
        x = torch.cat([down2_x, x], dim=1)
        up4_x = self.up4(x)
        x = F.interpolate(up4_x, scale_factor=2, mode="nearest")
        x = torch.cat([down1_x, x], dim=1)
        up5_x = self.up5(x)

        seg_x = self.seg(up5_x)
        out_x = self.final_activation(seg_x)

        if not batch:
            out_x = out_x.squeeze(0)
        return out_x


class unet1_transconv(nn.Module):
    def __init__(
        self,
        input_shape=(1, 256, 256),
        activation_inter_layer="relu",
        normalize_input=False,
        normalize_inter_layer=False,
        use_deep_supervision=False,
        nb_classes=4,
        final_activation="softmax",
    ):
        super().__init__()

        self.normalize_input = (
            nn.BatchNorm2d(input_shape[0]) if normalize_input else nn.Identity()
        )
        self.use_deep_supervision = use_deep_supervision

        # Deep supervision layers
        if self.use_deep_supervision:
            print("Using deep supervision.")
            self.ds1 = nn.Conv2d(128, nb_classes, kernel_size=(1, 1))
            self.ds2 = nn.Conv2d(64, nb_classes, kernel_size=(1, 1))
            self.ds3 = nn.Conv2d(32, nb_classes, kernel_size=(1, 1))

        # Encoder
        self.down1 = ConvolutionBlock(
            input_shape[0],
            32,
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down2 = ConvolutionBlock(
            32,
            64,
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down3 = ConvolutionBlock(
            64,
            128,
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down4 = ConvolutionBlock(
            128,
            256,
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        # Decoder with transpose convolutions for upscaling
        self.up1_transpose = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = ConvolutionBlock(
            256,
            128,
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up2_transpose = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = ConvolutionBlock(
            128,
            64,
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up3_transpose = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up3 = ConvolutionBlock(
            64,
            32,
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        self.final_conv = nn.Conv2d(32, nb_classes, kernel_size=1)
        self.final_activation = get_activation(final_activation)

    def forward(self, x):
        x, batch = make_batch(x)

        x = self.normalize_input(x)

        # Encoder path
        down1 = self.down1(x)
        x = F.max_pool2d(down1, 2)
        down2 = self.down2(x)
        x = F.max_pool2d(down2, 2)
        down3 = self.down3(x)
        x = F.max_pool2d(down3, 2)
        down4 = self.down4(x)

        # Decoder path
        x = self.up1_transpose(down4)
        x = torch.cat([x, down3], dim=1)
        up1_x = self.up1(x)

        x = self.up2_transpose(up1_x)
        x = torch.cat([x, down2], dim=1)
        up2_x = self.up2(x)

        x = self.up3_transpose(up2_x)
        x = torch.cat([x, down1], dim=1)
        up3_x = self.up3(x)

        x = self.final_conv(up3_x)
        out_x = self.final_activation(x)

        if not batch:
            out_x = out_x.squeeze(0)

        if self.use_deep_supervision:
            ds1_out = self.ds1(up1_x)
            ds1_out = F.interpolate(ds1_out, scale_factor=4, mode="nearest")
            ds1_out = self.final_activation(ds1_out)

            ds2_out = self.ds2(up2_x)
            ds2_out = F.interpolate(ds2_out, scale_factor=2, mode="nearest")
            ds2_out = self.final_activation(ds2_out)

            ds3_out = self.ds3(up3_x)
            ds3_out = self.final_activation(ds3_out)

            return out_x, ds1_out, ds2_out, ds3_out, None

        return out_x


def getDeeplabv3(num_classes=4):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    # change the first layer to accept single channel input
    model.backbone.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    # change the classifier to output the number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model


if __name__ == "__main__":
    input_shape = (
        1,
        256,
        256,
    )  # Assuming input is grayscale image with shape (1, height, width)
    model = unet1(input_shape)
    x = torch.randn((8, 1, 256, 256))
    y = model(x)
    print(y.shape)


class unet3(nn.Module):

    def __init__(
        self,
        input_shape=(1, 256, 256),
        activation_inter_layer="relu",
        normalize_input=False,
        normalize_inter_layer=False,
        use_deep_supervision=False,
        nb_classes=4,
        final_activation="softmax",
    ):
        super().__init__()

        self.normalize_input = (
            nn.BatchNorm2d(input_shape[0]) if normalize_input else nn.Identity()
        )

        self.use_deep_supervision = use_deep_supervision
        if self.use_deep_supervision:
            print("Using deep supervision.")
            self.ds1 = nn.Conv2d(512, nb_classes, kernel_size=(1, 1))
            self.ds2 = nn.Conv2d(256, nb_classes, kernel_size=(1, 1))
            self.ds3 = nn.Conv2d(128, nb_classes, kernel_size=(1, 1))
            self.ds4 = nn.Conv2d(64, nb_classes, kernel_size=(1, 1))

        # Encoder path
        self.down1 = ConvolutionBlock(
            input_shape[0],
            64,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down2 = ConvolutionBlock(
            64,
            128,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down3 = ConvolutionBlock(
            128,
            256,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down4 = ConvolutionBlock(
            256,
            512,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down5 = ConvolutionBlock(
            512,
            1024,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        # Decoder path
        self.up1 = ConvolutionBlock(
            1024 + 512,
            512,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up2 = ConvolutionBlock(
            512 + 256,
            256,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up3 = ConvolutionBlock(
            256 + 128,
            128,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up4 = ConvolutionBlock(
            128 + 64,
            64,
            (3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        self.seg = nn.Conv2d(64, nb_classes, kernel_size=(1, 1))
        self.final_activation = get_activation(final_activation)

    def forward(self, x):
        x, batch = make_batch(x)

        x = self.normalize_input(x)

        down1_x = self.down1(x)
        x = F.max_pool2d(down1_x, kernel_size=2)
        down2_x = self.down2(x)
        x = F.max_pool2d(down2_x, kernel_size=2)
        down3_x = self.down3(x)
        x = F.max_pool2d(down3_x, kernel_size=2)
        down4_x = self.down4(x)
        x = F.max_pool2d(down4_x, kernel_size=2)
        down5_x = self.down5(x)

        x = F.interpolate(down5_x, scale_factor=2, mode="nearest")
        x = torch.cat([down4_x, x], dim=1)
        up1_x = self.up1(x)
        x = F.interpolate(up1_x, scale_factor=2, mode="nearest")
        x = torch.cat([down3_x, x], dim=1)
        up2_x = self.up2(x)
        x = F.interpolate(up2_x, scale_factor=2, mode="nearest")
        x = torch.cat([down2_x, x], dim=1)
        up3_x = self.up3(x)
        x = F.interpolate(up3_x, scale_factor=2, mode="nearest")
        x = torch.cat([down1_x, x], dim=1)
        up4_x = self.up4(x)

        if self.use_deep_supervision:
            ds1_out = self.ds1(up1_x)
            ds1_out = F.interpolate(
                ds1_out, scale_factor=8, mode="bilinear", align_corners=True
            )
            ds2_out = self.ds2(up2_x)
            ds2_out = F.interpolate(
                ds2_out, scale_factor=4, mode="bilinear", align_corners=True
            )
            ds3_out = self.ds3(up3_x)
            ds3_out = F.interpolate(
                ds3_out, scale_factor=2, mode="bilinear", align_corners=True
            )
            ds4_out = self.ds4(up4_x)
            ds4_out = F.interpolate(
                ds4_out, scale_factor=1, mode="bilinear", align_corners=True
            )

        seg_x = self.seg(up4_x)
        out_x = self.final_activation(seg_x)

        if not batch:
            out_x = out_x.squeeze(0)

        if self.use_deep_supervision:
            return out_x, ds1_out, ds2_out, ds3_out, ds4_out

        return out_x


class unet0(nn.Module):
    """
    A smaller version of the U-Net architecture with channel sizes ranging from 16 to 64.
    Adapted for simplified processing tasks with fewer layers and narrower channels.
    """

    def __init__(
        self,
        input_shape=(1, 256, 256),
        activation_inter_layer="relu",
        normalize_input=False,
        normalize_inter_layer=False,
        use_deep_supervision=False,
        nb_classes=4,
        final_activation="softmax",
    ):
        super().__init__()

        self.use_deep_supervision = use_deep_supervision
        if self.use_deep_supervision:
            print("Using deep supervision.")
            self.ds1 = nn.Conv2d(32, nb_classes, kernel_size=(1, 1))

        # Input normalization layer
        self.normalize_input = (
            nn.BatchNorm2d(input_shape[0]) if normalize_input else nn.Identity()
        )

        # Downsampling path
        self.down1 = ConvolutionBlock(
            input_shape[0],
            16,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down2 = ConvolutionBlock(
            16,
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.down3 = ConvolutionBlock(
            32,
            64,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        # Upsampling path
        self.up1 = ConvolutionBlock(
            64 + 32,
            32,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )
        self.up2 = ConvolutionBlock(
            32 + 16,
            16,
            kernel_size=(3, 3),
            activation=activation_inter_layer,
            batch_normalization=normalize_inter_layer,
        )

        # Final segmentation layer
        self.seg = nn.Conv2d(16, nb_classes, kernel_size=(1, 1))
        self.final_activation = get_activation(final_activation)

    def forward(self, x):
        x, batch = make_batch(x)
        x = self.normalize_input(x)

        # Encoding path
        down1_x = self.down1(x)
        x = F.max_pool2d(down1_x, kernel_size=2)
        down2_x = self.down2(x)
        x = F.max_pool2d(down2_x, kernel_size=2)
        down3_x = self.down3(x)

        # Decoding path
        x = F.interpolate(down3_x, scale_factor=2, mode="nearest")
        x = torch.cat([down2_x, x], dim=1)
        up1_x = self.up1(x)
        x = F.interpolate(up1_x, scale_factor=2, mode="nearest")
        x = torch.cat([down1_x, x], dim=1)
        up2_x = self.up2(x)

        if self.use_deep_supervision:
            ds1_out = self.ds1(up1_x)
            ds1_out = F.interpolate(ds1_out, scale_factor=2, mode="nearest")
            ds1_out = self.final_activation(ds1_out)

        seg_x = self.seg(up2_x)
        out_x = self.final_activation(seg_x)

        if not batch:
            out_x = out_x.squeeze(0)

        if self.use_deep_supervision:
            ds2_out, ds3_out, ds4_out = None, None, None
            return out_x, ds1_out, ds2_out, ds3_out, ds4_out

        return out_x
