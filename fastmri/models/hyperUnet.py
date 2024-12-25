import torch
from torch import nn
from torch.nn import functional as F
from .hyperConv import PHConv2d

class HyperUnet(nn.Module):
    """
    PyTorch implementation of HyperU-Net model.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        ph_n: int = 2,  # Number of phase harmonics
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        cuda: bool = True,

    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
            ph_n: Number of phase harmonics for PHConv2d.
            kernel_size: Kernel size for PHConv2d.
            padding: Padding for PHConv2d.
            stride: Stride for PHConv2d.
            cuda: Whether to use CUDA.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.ph_n = ph_n
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.cuda = cuda

        self.down_sample_layers = nn.ModuleList(
            [PHConvBlock(in_chans, chans, drop_prob, ph_n, kernel_size, padding, stride, cuda)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(PHConvBlock(ch, ch * 2, drop_prob, ph_n, kernel_size, padding, stride, cuda))
            ch *= 2
        self.conv = PHConvBlock(ch, ch * 2, drop_prob, ph_n, kernel_size, padding, stride, cuda)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, ph_n, kernel_size, padding, stride, cuda))
            self.up_conv.append(PHConvBlock(ch * 2, ch, drop_prob, ph_n, kernel_size, padding, stride, cuda))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, ph_n, kernel_size, padding, stride, cuda))
        self.up_conv.append(
            nn.Sequential(
                PHConvBlock(ch * 2, ch, drop_prob, ph_n, kernel_size, padding, stride, cuda),
                PHConv2d(ph_n, ch, out_chans, kernel_size=1, stride=1, cuda=cuda),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class PHConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, ph_n: int, kernel_size: int, padding: int, stride: int, cuda: bool):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
            ph_n: Number of phase harmonics for PHConv2d.
            kernel_size: Kernel size for PHConv2d.
            padding: Padding for PHConv2d.
            stride: Stride for PHConv2d.
            cuda: Whether to use CUDA.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.ph_n = ph_n
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.cuda = cuda

        self.layers = nn.Sequential(
            PHConv2d(ph_n, in_chans, out_chans, kernel_size, padding, stride, cuda),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            PHConv2d(ph_n, out_chans, out_chans, kernel_size, padding, stride, cuda),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

        # self.phconv1 = PHConv2d(n, in_chans, out_chans, kernel_size=kernel_size, padding=padding, stride=1, cuda=True)
        # self.norm1 = nn.InstanceNorm2d(out_chans)
        # self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.dropout1 = nn.Dropout2d(drop_prob)
        #
        # self.phconv2 = PHConv2d(n, out_chans, out_chans, kernel_size=kernel_size, padding=padding, stride=1, cuda=True)
        # self.norm2 = nn.InstanceNorm2d(out_chans)
        # self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.dropout2 = nn.Dropout2d(drop_prob)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one PHConv2d layer followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int, ph_n: int, kernel_size: int, padding: int, stride: int, cuda: bool):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            ph_n: Number of phase harmonics for PHConv2d.
            kernel_size: Kernel size for PHConv2d.
            padding: Padding for PHConv2d.
            stride: Stride for PHConv2d.
            cuda: Whether to use CUDA.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.ph_n = ph_n
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.cuda = cuda

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
