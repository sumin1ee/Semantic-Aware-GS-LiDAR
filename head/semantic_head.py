import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Sequence, Union


class ConvBlock(nn.Module):
    """Two-layer convolutional block with optional normalization and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "bn",
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            norm_type (str): Type of normalization to apply. Supports ``"bn"`` for batch
                normalization, ``"gn"`` for group normalization, and ``"none"`` to disable it.
            dropout (float): Dropout probability applied after the first activation. Set to
                ``0.0`` to disable dropout.
        """
        super().__init__()
        use_bias = norm_type == "none"

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
        ]
        layers.extend(self._make_norm_layers(out_channels, norm_type))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
        )
        layers.extend(self._make_norm_layers(out_channels, norm_type))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    @staticmethod
    def _make_norm_layers(channels: int, norm_type: str) -> List[nn.Module]:
        """
        Args:
            channels (int): Number of channels for the normalization layer.
            norm_type (str): Normalization type selector.
        Returns:
            List[nn.Module]: A list containing the chosen normalization layer or empty if disabled.
        """
        if norm_type == "bn":
            return [nn.BatchNorm2d(channels)]
        if norm_type == "gn":
            num_groups = min(8, channels)
            return [nn.GroupNorm(num_groups=num_groups, num_channels=channels)]
        return []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape ``[B, C_in, H, W]``.
        Returns:
            torch.Tensor: Output tensor of shape ``[B, C_out, H, W]``.
        """
        return self.block(x)


class SemanticUNetHead(nn.Module):
    """U-Net style decoder that maps latent semantic grids to class logits."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        depth: int = 3,
        norm_type: str = "bn",
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input feature grid.
            num_classes (int): Number of semantic classes to predict.
            base_channels (int): Number of channels in the first encoder stage.
            depth (int): Number of encoder/decoder stages (excluding the bottleneck).
            norm_type (str): Normalization type passed to each convolutional block.
            dropout (float): Dropout probability applied inside convolutional blocks.
        """
        super().__init__()
        if depth < 1:
            raise ValueError("Depth of SemanticUNetHead must be at least 1.")

        self.depth = depth
        self.pool_layers = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        encoder_channels: List[int] = []
        current_channels = in_channels

        for stage_idx in range(depth):
            out_channels = base_channels * (2 ** stage_idx)
            self.down_blocks.append(
                ConvBlock(current_channels, out_channels, norm_type=norm_type, dropout=dropout)
            )
            encoder_channels.append(out_channels)
            current_channels = out_channels
            if stage_idx < depth - 1:
                self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        bottleneck_channels = current_channels * 2
        self.bottleneck = ConvBlock(
            current_channels, bottleneck_channels, norm_type=norm_type, dropout=dropout
        )

        self.up_blocks = nn.ModuleList()
        decoder_in_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels[:-1]):
            self.up_blocks.append(
                ConvBlock(
                    decoder_in_channels + skip_channels,
                    skip_channels,
                    norm_type=norm_type,
                    dropout=dropout,
                )
            )
            decoder_in_channels = skip_channels

        self.classifier = nn.Conv2d(decoder_in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape ``[B, C_in, H, W]`` representing rendered
                semantic feature maps.
        Returns:
            torch.Tensor: Logits of shape ``[B, num_classes, H, W]``.
        """
        skips: List[torch.Tensor] = []
        h = x

        for idx, block in enumerate(self.down_blocks):
            h = block(h)
            if idx < self.depth - 1:
                skips.append(h)
                h = self.pool_layers[idx](h)

        h = self.bottleneck(h)

        for block in self.up_blocks:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        logits = self.classifier(h)
        return logits


class SemanticConvHead(nn.Module):
    """Lightweight 1x1 convolutional head for semantic prediction."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: Union[int, Sequence[int]] = (128, 64),
        norm_type: str = "bn",
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input feature grid.
            num_classes (int): Number of semantic classes to predict.
            hidden_channels (Union[int, Sequence[int]]): Hidden channel sizes for intermediate
                1x1 convolutions. A single integer is treated as a one-element sequence.
            norm_type (str): Normalization type passed to each intermediate convolution.
            dropout (float): Dropout probability applied after intermediate activations.
        """
        super().__init__()

        if isinstance(hidden_channels, int):
            channel_sequence: List[int] = [hidden_channels]
        else:
            channel_sequence = list(hidden_channels)

        layers: List[nn.Module] = []
        current_channels = in_channels
        for idx, hidden_dim in enumerate(channel_sequence):
            layers.append(nn.Conv2d(current_channels, hidden_dim, kernel_size=1, bias=True))
            layers.extend(ConvBlock._make_norm_layers(hidden_dim, norm_type))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(p=dropout))
            current_channels = hidden_dim

        layers.append(nn.Conv2d(current_channels, num_classes, kernel_size=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape ``[B, C_in, H, W]``.
        Returns:
            torch.Tensor: Logits of shape ``[B, num_classes, H, W]``.
        """
        return self.net(x)


def build_semantic_head(
    head_type: str,
    in_channels: int,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Args:
        head_type (str): Semantic head variant. Supports ``"unet"`` and ``"conv"``.
        in_channels (int): Number of channels in the input feature tensor.
        num_classes (int): Number of semantic classes to predict.
        **kwargs: Additional keyword arguments forwarded to the specific head constructor.
    Returns:
        nn.Module: Instantiated semantic head module.
    """
    head_type = head_type.lower()
    if head_type in ("unet", "unet-style", "u-net"):
        return SemanticUNetHead(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=kwargs.get("base_channels", 64),
            depth=kwargs.get("depth", 3),
            norm_type=kwargs.get("norm_type", "bn"),
            dropout=kwargs.get("dropout", 0.0),
        )

    if head_type in ("conv", "1x1", "conv1x1", "mlp"):
        return SemanticConvHead(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=kwargs.get("hidden_channels", (128, 64)),
            norm_type=kwargs.get("norm_type", "bn"),
            dropout=kwargs.get("dropout", 0.0),
        )

    raise ValueError(f"Unsupported semantic head type: {head_type}")
