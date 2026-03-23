"""
DeepSupSMPUNet — U-Net with pretrained encoder, SCSE attention, and deep supervision.

Architecture: ResNet34 encoder (ImageNet) + U-Net decoder with SCSE attention
              + 4 auxiliary heads for deep supervision during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DeepSupSMPUNet(nn.Module):
    """
    U-Net with pretrained ResNet34, SCSE attention, and deep supervision.

    Training:  returns (main_output, [aux_1, aux_2, aux_3, aux_4])
    Inference: returns main_output only
    """

    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet',
                 in_channels=1, num_classes=4):
        super().__init__()

        self.base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_attention_type='scse',
            decoder_channels=(256, 128, 64, 32, 16),
        )

        # Deep supervision: hook intermediate decoder blocks
        self._intermediate = {}
        for i, block in enumerate(self.base.decoder.blocks[:-1]):
            block.register_forward_hook(self._make_hook(i))

        # Auxiliary heads (1x1 conv -> num_classes)
        ds_channels = [256, 128, 64, 32]
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(ch, num_classes, kernel_size=1) for ch in ds_channels
        ])

        # Init aux heads with Kaiming
        for head in self.aux_heads:
            nn.init.kaiming_normal_(head.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(head.bias)

    def _make_hook(self, idx):
        def hook(module, input, output):
            self._intermediate[idx] = output
        return hook

    def forward(self, x):
        H, W = x.shape[2:]
        main_out = self.base(x)

        if self.training:
            aux_outputs = []
            for i, head in enumerate(self.aux_heads):
                feat = self._intermediate[i]
                aux = head(feat)
                aux = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=False)
                aux_outputs.append(aux)
            return main_out, aux_outputs

        return main_out
