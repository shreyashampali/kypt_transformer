# Copyright (c) 2020 Graz University of Technology All rights reserved.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
import torch.nn as nn
from torch.nn import functional as F
from main.config import cfg
from common.nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers, make_upsample_layers
from common.nets.resnet import ResNetBackbone
import math

class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)
    
    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        img_feat = self.resnet(img)
        return img_feat

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.resnet_decoder = Decoder()

    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid, heatmap_out = self.resnet_decoder(img_feat, skip_conn_layers)
        return feature_pyramid, heatmap_out

class DecoderNet_big(nn.Module):
    def __init__(self):
        super(DecoderNet_big, self).__init__()
        self.resnet_decoder = Decoder_big()

    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid, heatmap_out = self.resnet_decoder(img_feat, skip_conn_layers)
        return feature_pyramid, heatmap_out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        if cfg.resnet_type >=50:
            self.conv0d = make_conv_layers([2048, 512], kernel=1, padding=0)

            self.conv1d = make_conv_layers([1024, 256], kernel=1, padding=0)
            self.deconv1 = make_deconv_layers([2048, 256])
            self.conv1 = make_conv_layers([512, 256])

            self.conv2d = make_conv_layers([512, 128], kernel=1, padding=0)
            self.deconv2 = make_deconv_layers([256, 128])
            self.conv2 = make_conv_layers([256, 128])

            self.conv3d = make_conv_layers([256, 64], kernel=1, padding=0)
            self.deconv3 = make_deconv_layers([128, 64])
            self.conv3 = make_conv_layers([128, 64])

            self.conv4d = make_conv_layers([64, 32], kernel=1, padding=0)
            self.deconv4 = make_deconv_layers([64, 64])
            self.conv4 = make_conv_layers([64+32, 32])
        else:
            self.conv1d = make_conv_layers([256, 256], kernel=1, padding=0)
            self.deconv1 = make_deconv_layers([512, 256])
            self.conv1 = make_conv_layers([512, 256])

            self.conv2d = make_conv_layers([128, 128], kernel=1, padding=0)
            self.deconv2 = make_deconv_layers([256, 128])
            self.conv2 = make_conv_layers([256, 128])

            self.conv3d = make_conv_layers([64, 64], kernel=1, padding=0)
            self.deconv3 = make_deconv_layers([128, 64])
            self.conv3 = make_conv_layers([128, 64])

            self.conv4d = make_conv_layers([64, 32], kernel=1, padding=0)
            self.deconv4 = make_deconv_layers([64, 64])
            self.conv4 = make_conv_layers([64 + 32, 32])



        if cfg.has_object:
            self.convOut_hm = make_conv_layers([32, 32, 1], kernel=1, padding=0, bnrelu_final=False)
            self.convOut_seg = make_conv_layers([32, 32, 1], kernel=1, padding=0, bnrelu_final=False)
        else:
            self.convOut = make_conv_layers([32, 1], kernel=1, padding=0, bnrelu_final=False)


    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid = {}
        assert isinstance(skip_conn_layers, dict)
        if cfg.resnet_type >= 50:
            feature_pyramid['stride32'] = self.conv0d(img_feat)
        else:
            feature_pyramid['stride32'] = img_feat

        skip_stride16_d = self.conv1d(skip_conn_layers['stride16']) #512
        deconv_img_feat1 = self.deconv1(img_feat)
        deconv_img_feat1_cat = torch.cat((skip_stride16_d, deconv_img_feat1),1)
        deconv_img_feat1_cat_conv = self.conv1(deconv_img_feat1_cat) #256
        feature_pyramid['stride16'] = deconv_img_feat1_cat_conv #256

        skip_stride8_d = self.conv2d(skip_conn_layers['stride8'])  # 256
        deconv_img_feat2 = self.deconv2(deconv_img_feat1_cat_conv)
        deconv_img_feat2_cat = torch.cat((skip_stride8_d, deconv_img_feat2), 1)
        deconv_img_feat2_cat_conv = self.conv2(deconv_img_feat2_cat) # 128
        feature_pyramid['stride8'] = deconv_img_feat2_cat_conv # 128

        skip_stride4_d = self.conv3d(skip_conn_layers['stride4'])  # 128
        deconv_img_feat3 = self.deconv3(deconv_img_feat2_cat_conv)
        deconv_img_feat3_cat = torch.cat((skip_stride4_d, deconv_img_feat3), 1)
        deconv_img_feat3_cat_conv = self.conv3(deconv_img_feat3_cat) # 64
        feature_pyramid['stride4'] = deconv_img_feat3_cat_conv # 64

        skip_stride2_d = self.conv4d(skip_conn_layers['stride2'])  # 32
        deconv_img_feat4 = self.deconv4(deconv_img_feat3_cat_conv)
        deconv_img_feat4_cat = torch.cat((skip_stride2_d, deconv_img_feat4), 1)
        deconv_img_feat4_cat_conv = self.conv4(deconv_img_feat4_cat) # 16x128x128 (featxhxw)
        feature_pyramid['stride2'] = deconv_img_feat4_cat_conv

        if cfg.has_object:
            heatmap_out = self.convOut_hm(deconv_img_feat4_cat_conv) # N x 1 x 128 x 128
            seg_out = self.convOut_seg(deconv_img_feat4_cat_conv)  # N x 1 x 128 x 128
            heatmap_out = torch.cat([heatmap_out, seg_out], dim=1)
        else:
            heatmap_out = self.convOut(deconv_img_feat4_cat_conv)

        return feature_pyramid, heatmap_out

class Decoder_big(nn.Module):
    def __init__(self):
        super(Decoder_big, self).__init__()
        self.deconv1 = make_deconv_layers([2048, 1024])
        self.conv1 = make_conv_layers([2048, 1024])

        self.deconv2 = make_deconv_layers([1024, 512])
        self.conv2 = make_conv_layers([1024, 512])

        self.deconv3 = make_deconv_layers([512, 256])
        self.conv3 = make_conv_layers([512, 256])

        self.deconv4 = make_deconv_layers([256, 128])
        self.conv4 = make_conv_layers([64+128, 128])

        if cfg.has_object:
            self.convOut_hm = make_conv_layers([128, 128, 64, 1], kernel=1, padding=0, bnrelu_final=False)
            self.convOut_seg = make_conv_layers([128, 128, 64, 1], kernel=1, padding=0, bnrelu_final=False)
        else:
            self.convOut = make_conv_layers([128, 1], kernel=1, padding=0, bnrelu_final=False)



    def forward(self, img_feat, skip_conn_layers):
        feature_pyramid = {}
        assert isinstance(skip_conn_layers, dict)
        feature_pyramid['stride32'] = img_feat

        deconv_img_feat1 = self.deconv1(img_feat)
        deconv_img_feat1_cat = torch.cat((skip_conn_layers['stride16'], deconv_img_feat1),1)
        deconv_img_feat1_cat_conv = self.conv1(deconv_img_feat1_cat)
        feature_pyramid['stride16'] = deconv_img_feat1_cat_conv

        deconv_img_feat2 = self.deconv2(deconv_img_feat1_cat_conv)
        deconv_img_feat2_cat = torch.cat((skip_conn_layers['stride8'], deconv_img_feat2), 1)
        deconv_img_feat2_cat_conv = self.conv2(deconv_img_feat2_cat)
        feature_pyramid['stride8'] = deconv_img_feat2_cat_conv

        deconv_img_feat3 = self.deconv3(deconv_img_feat2_cat_conv)
        deconv_img_feat3_cat = torch.cat((skip_conn_layers['stride4'], deconv_img_feat3), 1)
        deconv_img_feat3_cat_conv = self.conv3(deconv_img_feat3_cat)
        feature_pyramid['stride4'] = deconv_img_feat3_cat_conv

        deconv_img_feat4 = self.deconv4(deconv_img_feat3_cat_conv)
        deconv_img_feat4_cat = torch.cat((skip_conn_layers['stride2'], deconv_img_feat4), 1)
        deconv_img_feat4_cat_conv = self.conv4(deconv_img_feat4_cat) # 128x128x128 (featxhxw)
        feature_pyramid['stride2'] = deconv_img_feat4_cat_conv

        if cfg.has_object:
            heatmap_out = self.convOut_hm(deconv_img_feat4_cat_conv) # N x 1 x 128 x 128
            seg_out = self.convOut_seg(deconv_img_feat4_cat_conv)  # N x 1 x 128 x 128
            heatmap_out = torch.cat([heatmap_out, seg_out], dim=1)
        else:
            heatmap_out = self.convOut(deconv_img_feat4_cat_conv)

        return feature_pyramid, heatmap_out

