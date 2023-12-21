"""This file contains the network architecture of the model.
We will be using Mobilenet V3 Large architecture here.
"""

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet101, resnet152, resnet18, resnet34, resnet50


class ResNet(nn.Module):
    def __init__(self, backbone="resnet50", backbone_path=None, weights="IMAGENET1K_V1"):
        """
        Initializes a ResNet object.

        Args:
            backbone (str): The backbone architecture to use. Default is "resnet50".
            backbone_path (str): The path to the pre-trained backbone weights. Default is None.
            weights (str): The weights to use for the backbone. Default is "IMAGENET1K_V1".
        """
        super().__init__()
        if backbone == "resnet18":
            backbone = resnet18(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == "resnet34":
            backbone = resnet34(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == "resnet50":
            backbone = resnet50(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == "resnet101":
            backbone = resnet101(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.feature_extractor(x)


class _Ssd300(nn.Module):
    """
    Single Shot MultiBox Detector (SSD) network implementation.

    Args:
        backbone (nn.Module): Backbone network for feature extraction. Default is ResNet-50.

    Attributes:
        feature_extractor (nn.Module): Feature extraction backbone network.
        label_num (int): Number of class labels.
        num_defaults (list): List of number of default boxes for each feature map.
        loc (nn.ModuleList): List of convolutional layers for localization predictions.
        conf (nn.ModuleList): List of convolutional layers for class confidence predictions.
        additional_blocks (nn.ModuleList): List of additional convolutional layers for feature extraction.
    """

    def __init__(self, backbone=ResNet("resnet50")):
        super().__init__()

        self.feature_extractor = backbone
        self.label_num = 81
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        """
        Build additional convolutional layers for feature extraction.

        Args:
            input_size (list): List of input channel sizes for each feature map.
        """
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
            zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])
        ):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels, output_size, kernel_size=3, padding=1, stride=2, bias=False
                    ),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        """
        Initialize the weights of the network.
        """
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        """
        Reshape and concatenate the localization and confidence predictions.

        Args:
            src (list): List of feature maps.
            loc (nn.ModuleList): List of convolutional layers for localization predictions.
            conf (nn.ModuleList): List of convolutional layers for class confidence predictions.

        Returns:
            tuple: Tuple containing the reshaped and concatenated localization and confidence predictions.
        """
        ret = [
            (l(s).reshape(s.size(0), 4, -1), c(s).reshape(s.size(0), self.label_num, -1))
            for s, l, c in zip(src, loc, conf)
        ]
        print(ret[0][0].shape, ret[0][1].shape)
        locs, confs = list(zip(*ret))
        print(locs[0].shape, confs[0].shape, len(locs), len(confs))
        locs, confs = (torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous())
        print(locs.shape, confs.shape)
        return (locs, confs)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing the localization and confidence predictions.
        """
        x = self.feature_extractor(x)
        detection_feed = [x]
        for ix, l in enumerate(self.additional_blocks):
            print(ix, x[0].shape, x[1].shape)
            x = l(x)
            print(ix, x[0].shape, x[1].shape)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return (locs, confs)

if __name__ == "__main__":
    model = _Ssd300()
    model(torch.randn(10, 3, 300, 300))

    # '' from torchinfo import summary

    #  '' print(summary(model, (2, 3, 300, 300)))