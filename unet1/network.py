import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)

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
        nb_classes=4,
        final_activation="softmax",
    ):
        super(unet1, self).__init__()

        self.normalize_input = (
            nn.BatchNorm2d(input_shape[0]) if normalize_input else nn.Identity()
        )

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

        seg_x = self.seg(up5_x)
        out_x = self.final_activation(seg_x)

        if not batch:
            out_x = out_x.squeeze(0)
        return out_x


if __name__ == "__main__":
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
