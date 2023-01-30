import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

__all__ = [
    "ResidualDenseBlock", "ResidualInResidualDenseBlock",
    "ReceptiveFieldBlock", "ReceptiveFieldDenseBlock", "ResidualOfReceptiveFieldDenseBlock",
    "UpsamplingModule",
    "Generator",
]

class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out

class ResidualInResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualInResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out

class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Modules introduced in RFBNet paper.
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """

        super(ReceptiveFieldBlock, self).__init__()
        branch_channels = in_channels // 4

        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0))

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (1, 1), dilation=1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (3, 3), dilation=3),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels // 2, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels // 2, (branch_channels // 4) * 3, (1, 3), (1, 1), (0, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d((branch_channels // 4) * 3, branch_channels, (3, 1), (1, 1), (1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(branch_channels, branch_channels, (3, 3), (1, 1), (5, 5), dilation=5),
        )

        self.conv_linear = nn.Conv2d(4 * branch_channels, out_channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        shortcut = torch.mul(shortcut, 0.2)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        out = self.conv_linear(out)
        out = torch.add(out, shortcut)

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ReceptiveFieldDenseBlock(nn.Module):
    """Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
    RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, channels: int, growth_channels: int):
        """
        Args:
            channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
        """

        super(ReceptiveFieldDenseBlock, self).__init__()
        self.rfb1 = ReceptiveFieldBlock(channels + 0 * growth_channels, growth_channels)
        self.rfb2 = ReceptiveFieldBlock(channels + 1 * growth_channels, growth_channels)
        self.rfb3 = ReceptiveFieldBlock(channels + 2 * growth_channels, growth_channels)
        self.rfb4 = ReceptiveFieldBlock(channels + 3 * growth_channels, growth_channels)
        self.rfb5 = ReceptiveFieldBlock(channels + 4 * growth_channels, channels)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        rfb1 = self.leaky_relu(self.rfb1(x))
        rfb2 = self.leaky_relu(self.rfb2(torch.cat([x, rfb1], 1)))
        rfb3 = self.leaky_relu(self.rfb3(torch.cat([x, rfb1, rfb2], 1)))
        rfb4 = self.leaky_relu(self.rfb4(torch.cat([x, rfb1, rfb2, rfb3], 1)))
        rfb5 = self.identity(self.rfb5(torch.cat([x, rfb1, rfb2, rfb3, rfb4], 1)))
        out = torch.mul(rfb5, 0.2)
        out = torch.add(out, identity)

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    def __init__(self, channels: int, growths: int):
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.rfdb1 = ReceptiveFieldDenseBlock(channels, growths)
        self.rfdb2 = ReceptiveFieldDenseBlock(channels, growths)
        self.rfdb3 = ReceptiveFieldDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rfdb1(x)
        out = self.rfdb2(out)
        out = self.rfdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


class UpsamplingModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpsamplingModule, self).__init__()
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ReceptiveFieldBlock(channels, channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels * 4, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            ReceptiveFieldBlock(channels, channels),
            nn.LeakyReLU(0.2, True),

            # nn.Upsample(scale_factor=2, mode="nearest"),
            # ReceptiveFieldBlock(channels, channels),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(channels, channels * 4, (3, 3), (1, 1), (1, 1)),
            # nn.PixelShuffle(2),
            # ReceptiveFieldBlock(channels, channels),
            # nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsampling(x)

        return out


class RFB_RRDB(nn.Module):
    def __init__(self, in_nc, out_nc) -> None:
        super(RFB_RRDB, self).__init__()
        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_nc, 64, (3, 3), (1, 1), (1, 1))

        # Trunk-a backbone network.
        trunk_a = []
        for _ in range(16):
            trunk_a.append(ResidualInResidualDenseBlock(64, 32))
        self.trunk_a = nn.Sequential(*trunk_a)

        # Trunk-RFB backbone network.
        trunk_rfb = []
        for _ in range(8):
            trunk_rfb.append(ResidualOfReceptiveFieldDenseBlock(64, 32))
        self.trunk_rfb = nn.Sequential(*trunk_rfb)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = ReceptiveFieldBlock(64, 64)

        # Upsampling convolutional layer.
        self.upsampling = UpsamplingModule(64)

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(64, out_nc, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out_a = self.trunk_a(out1)
        out_rfb = self.trunk_rfb(out_a)
        out2 = self.conv2(out_rfb)
        out = torch.add(out1, out2)
        out = self.conv2(out)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
