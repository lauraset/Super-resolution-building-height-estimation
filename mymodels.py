'''
Function: super-resolution height regression

Update:
Aug. 31, 2023: add adaptive bins for regression, refer to https://github.com/shariqfarooq123/AdaBins
'''
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from SR.edsr import EDSR, EDSR_fea, EDSR_feaHR
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet import UnetDecoder, UnetDecoder_noise
from SR.HRfuse import HRfuse, HRfuse_x2, HRfeature, HRfuse_residual, Refine_residual, GeoNet, HRupsample
from torch.distributions.uniform import Uniform

class SRRegress(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet",
                 in_channels=4, classes=1, super_channels=4):
        super().__init__()
        self.super_res = EDSR(n_colors=in_channels, n_out=super_channels)
        self.regress = smp.Unet(encoder_name=encoder_name, encoder_weights = encoder_weights,
                                in_channels=super_channels, classes=classes)
    def forward(self,x):
        super_fea = self.super_res(x)
        pred = self.regress(super_fea)
        return pred

# height regression and land cover classification
class SRRegress_Cls_EDSR(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=4, classes=1, super_channels=4,
                 decoder_channels = (256, 128, 64, 32, 16)):
        super().__init__()
        self.super_res = EDSR(n_colors=in_channels, n_out=super_channels)
        self.super_out = nn.Conv2d(super_channels, 3, kernel_size=3, padding=1)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=super_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.seg1 = nn.Conv2d(decoder_channels[-1], 1, kernel_size=3, padding=1)
        self.seg2 = nn.Conv2d(decoder_channels[-1], 2, kernel_size=3, padding=1)

    def forward(self,x):
        super_fea = self.super_res(x)
        super_out = self.super_out(super_fea)

        encode_fea = self.encoder(super_fea)

        height = self.decoder1(*encode_fea)
        height = self.seg1(height)

        build = self.decoder2(*encode_fea)
        build = self.seg2(build)

        return super_out, height, build

# 2023.9.10: ass
'''
class SRRegress_Cls_Segformer(torch.nn.Module):
    def __init__(self, encoder_name="mit_b1",
                 in_channels=4, num_classes=1,
                 super_channels=4, pretrained_path=r'D:\code\pretrained_weights\mit_b1.pth'):
        super().__init__()
        self.super_res = EDSR(n_colors=in_channels, n_out=super_channels)
        self.super_out = nn.Conv2d(super_channels, 3, kernel_size=3, padding=1)
        # Encoder-Decoder
        self.num_classes = num_classes
        self.embedding_dim = 256
        self.feature_strides = [4, 8, 16, 32]
        self.encoder = getattr(mix_transformer, encoder_name)(stride=(4, 2, 2, 2))
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path)
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )
        # change input channels
        patch_first_conv(self.encoder, super_channels, pretrained=pretrained_path is not None)
        self.decoder1 = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=1) # height
        self.decoder2 = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=2) # build

    def forward(self,x):
        super_fea = self.super_res(x)
        super_out = self.super_out(super_fea)
        h, w = super_out.shape[2:]

        encode_fea = self.encoder(super_fea)

        height = self.decoder1(encode_fea)
        height = torch.nn.functional.interpolate(height, size=(h,w), mode="bilinear", align_corners=False)

        build = self.decoder2(encode_fea)
        build = torch.nn.functional.interpolate(build, size=(h,w), mode="bilinear", align_corners=False)

        return super_out, height, build

    def get_param_groups(self):
        param_groups = [[], [], []]  #
        # encoder
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        # decoder
        for param in list(self.decoder1.parameters()):
            param_groups[2].append(param)
        for param in list(self.decoder2.parameters()):
            param_groups[2].append(param)
        # super resolution
        for param in list(self.super_res.parameters()):
            param_groups[2].append(param)
        for param in list(self.super_out.parameters()):
            param_groups[2].append(param)
        return param_groups
'''

# fusion at decision layer
'''
class SRRegress_Cls_decision_lowfea(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=7, classes=1, super_in=4, super_mid=64, upscale=4):
        super().__init__()
        self.super_in = super_in
        # Super-Resolution branches
        self.super_res = EDSR_fea(n_colors=super_in, n_out=3, n_feats=super_mid)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        dec_in = (256, 128, 64, 32, 16)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        # Fuse the SR HR features and the result of decoder, then upsampler
        self.seg1 = HRfuse(hr_channel=super_mid, lr_channel=dec_in[-1], mid_channel=dec_in[-1],
                           out_channel=1, upscale=upscale)
        self.seg2 = HRfuse(hr_channel=super_mid, lr_channel=dec_in[-1], mid_channel=dec_in[-1],
                           out_channel=2, upscale=upscale)

    def forward(self, x):
        super_out, super_fea = self.super_res(x[:, :self.super_in])

        encode_fea = self.encoder(x) # torch.cat((xo, xs), dim=1)

        height = self.decoder1(*encode_fea)
        height = self.seg1(height, super_fea)

        build = self.decoder2(*encode_fea)
        build = self.seg2(build, super_fea)

        return super_out, height, build
'''

class SRRegress_Cls_decision(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=7, classes=1, super_in=4, super_mid=64, upscale=4):
        super().__init__()
        self.super_in = super_in
        # Super-Resolution branches
        self.super_res = EDSR_feaHR(n_colors=super_in, n_out=3, n_feats=super_mid)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        dec_in = (256, 128, 64, 32, 16)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        # Fuse the SR HR features and the result of decoder, then upsampler
        self.seg1 = HRfuse_x2(hr_channel=super_mid, lr_channel=dec_in[-1], mid_channel=dec_in[-1],
                           out_channel=1, upscale=upscale)
        self.seg2 = HRfuse_x2(hr_channel=super_mid, lr_channel=dec_in[-1], mid_channel=dec_in[-1],
                           out_channel=2, upscale=upscale)

    def forward(self, x):
        '''
        :param xo: optical images, sentinel-2
        :param xs: sar images, sentinel-1
        :return:
        '''
        super_out, super_fea = self.super_res(x[:, :self.super_in])

        encode_fea = self.encoder(x) # torch.cat((xo, xs), dim=1)

        height = self.decoder1(*encode_fea)
        height = self.seg1(height, super_fea)

        build = self.decoder2(*encode_fea)
        build = self.seg2(build, super_fea)

        return super_out, height, build


# add hr feature using fixed networks from GAN
# add feature extraction module (3 blocks) 2023.9.27
class SRRegress_Cls_feature(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=7, classes=1, super_in=4, super_mid=64, upscale=4,
                 isaggre=False, chans_build=2, uniform_range=0.3, isunsup=False):
        super().__init__()
        # self.super_in = super_in
        # Super-Resolution branches
        # self.super_res = EDSR_feaHR(n_colors=super_in, n_out=3, n_feats=super_mid)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        dec_in = (256, 128, 64, 32, 16)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        # Fuse the SR HR features and the result of decoder, then upsampler
        self.reg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1], mid_chans=dec_in[-1],
                           out_chans=1, upscale=upscale)
        self.seg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1], mid_chans=dec_in[-1],
                           out_chans=chans_build, upscale=upscale)
        self.hrfeat = HRfeature(in_chans=super_in, mid_chans=super_mid, out_chans=super_mid)
        # Aggregate
        self.isaggre = isaggre
        if self.isaggre:
            self.aggre_height = nn.Conv2d(super_mid, 1, 3,1,1) # kernel is set to 1

    def forward(self, x, super_fea):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :param super_fea: directly generated from existing one, and may has Internal Covariate Shift
        :return:
        '''
        encode_fea = self.encoder(x)
        super_fea = self.hrfeat(super_fea)

        height_fea = self.decoder1(*encode_fea)
        if self.isaggre:
            height_aggre = self.aggre_height(height_fea)
            # height_aggre = height_aggre.squeeze()

        height = self.reg(height_fea, super_fea)
        # height = height.squeeze()

        build = self.decoder2(*encode_fea)
        build = self.seg(build, super_fea)

        if self.isaggre:
            return height, build, height_aggre

        return height, build

    def forward_unsup(self, x, super_fea):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :param super_fea: directly generated from existing one, and may has Internal Covariate Shift
        :return:
        '''
        encode_fea = self.encoder(x)
        super_fea = self.hrfeat(super_fea)

        height_fea = self.decoder1(*encode_fea)

        height = self.reg(height_fea, super_fea)
        height = height.squeeze()
        #
        # build = self.decoder2(*encode_fea)
        # build = self.seg(build, super_fea)

        return height # , build

    def forward_nobuild(self, x, super_fea):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :param super_fea: directly generated from existing one, and may has Internal Covariate Shift
        :return:
        '''
        encode_fea = self.encoder(x)
        super_fea = self.hrfeat(super_fea)

        height_fea = self.decoder1(*encode_fea)
        if self.isaggre:
            height_aggre = self.aggre_height(height_fea)
            # height_aggre = height_aggre.squeeze()

        height = self.reg(height_fea, super_fea)
        # height = height.squeeze()

        # build = self.decoder2(*encode_fea)
        # build = self.seg(build, super_fea)

        if self.isaggre:
            return height, height_aggre

        return height


# 2023.12.14: remove the super-resolution module
class SRRegress_Cls_nosuper(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=7, upscale=4, isaggre=False, chans_build=2):
        super().__init__()
        # self.super_in = super_in
        # Super-Resolution branches
        # self.super_res = EDSR_feaHR(n_colors=super_in, n_out=3, n_feats=super_mid)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        dec_in = (256, 128, 64, 32, 16)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        # directly upsample the prediction result
        self.reg = HRupsample(lr_chans=dec_in[-1], out_chans=1, upscale=upscale)
        self.seg = HRupsample(lr_chans=dec_in[-1], out_chans=chans_build, upscale=upscale)
        # Aggregate
        self.isaggre = isaggre
        if self.isaggre:
            self.aggre_height = nn.Conv2d(dec_in[-1], 1, 3,1,1) # kernel is set to 1

    def forward(self, x):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :return:
        '''
        encode_fea = self.encoder(x)
        height_fea = self.decoder1(*encode_fea)
        if self.isaggre:
            height_aggre = self.aggre_height(height_fea)

        height = self.reg(height_fea)

        build = self.decoder2(*encode_fea)
        build = self.seg(build)

        if self.isaggre:
            return height, build, height_aggre

        return height, build

    def forward_nobuild(self, x):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :return:
        '''
        encode_fea = self.encoder(x)
        height_fea = self.decoder1(*encode_fea)
        if self.isaggre:
            height_aggre = self.aggre_height(height_fea)

        height = self.reg(height_fea)

        if self.isaggre:
            return height, height_aggre

        return height


# unsupervised version
class SRRegress_Cls_feature_unsup(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=7, classes=1, super_in=4, super_mid=64, upscale=4,
                 isaggre=False, chans_build=2, uniform_range=0.3, isunsup=False):
        super().__init__()
        # self.super_in = super_in
        # Super-Resolution branches
        # self.super_res = EDSR_feaHR(n_colors=super_in, n_out=3, n_feats=super_mid)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        dec_in = (256, 128, 64, 32, 16)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        # Fuse the SR HR features and the result of decoder, then upsampler
        self.reg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1], mid_chans=dec_in[-1],
                           out_chans=1, upscale=upscale)
        self.seg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1], mid_chans=dec_in[-1],
                           out_chans=chans_build, upscale=upscale)
        self.hrfeat = HRfeature(in_chans=super_in, mid_chans=super_mid, out_chans=super_mid)
        # Aggregate
        self.isaggre = isaggre
        if self.isaggre:
            self.aggre_height = nn.Conv2d(super_mid, 1, 3,1,1) # kernel is set to 1

        # feature pertubation: noise, set to 3
        #if isunsup:
        num_aux = 3
        self.decoder_aux = nn.ModuleList([UnetDecoder_noise(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
            uniform_range=uniform_range) for _ in range(num_aux)])
        self.reg_aux = nn.ModuleList([HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1], mid_chans=dec_in[-1],
                           out_chans=1, upscale=upscale) for _ in range(num_aux)])

    def forward(self, x, super_fea):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :param super_fea: directly generated from existing one, and may has Internal Covariate Shift
        :return:
        '''
        encode_fea = self.encoder(x)
        super_fea = self.hrfeat(super_fea)

        height_fea = self.decoder1(*encode_fea)
        if self.isaggre:
            height_aggre = self.aggre_height(height_fea)
            height_aggre = height_aggre.squeeze()

        height = self.reg(height_fea, super_fea)
        height = height.squeeze()

        build = self.decoder2(*encode_fea)
        build = self.seg(build, super_fea)

        if self.isaggre:
            return height, build, height_aggre

        return height, build

    # unsupervised learning
    def forward_unsup(self, x, super_fea):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :param super_fea: directly generated from existing one, and may has Internal Covariate Shift
        :return:
        '''
        encode_fea = self.encoder(x)
        super_fea = self.hrfeat(super_fea)

        height_fea = self.decoder1(*encode_fea)
        height = self.reg(height_fea, super_fea)
        height = height.squeeze()

        # noise
        height_noise = [decoder_aux(*encode_fea) for decoder_aux in self.decoder_aux]
        height_noise = [reg_aux(tmp, super_fea) for reg_aux, tmp in zip(self.reg_aux, height_noise)]
        height_noise = [i.squeeze() for i in height_noise]

        return height, height_noise


# 2023.11.15: add geo prior
class SRRegress_Cls_feature_geo(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=7, classes=1, super_in=4, super_mid=64, upscale=4,
                 isaggre=False, chans_build=2, geo_chans_in=3, geo_chans_mid=16):
        super().__init__()
        # self.super_in = super_in
        # Super-Resolution branches
        # self.super_res = EDSR_feaHR(n_colors=super_in, n_out=3, n_feats=super_mid)
        # Encoder-Decoder
        self.in_chans = in_channels
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        dec_in = (256, 128, 64, 32, 16)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        # Fuse the SR HR features and the result of decoder, then upsampler
        self.reg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1]+geo_chans_mid, mid_chans=dec_in[-1],
                           out_chans=1, upscale=upscale)
        self.seg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1]+geo_chans_mid, mid_chans=dec_in[-1],
                           out_chans=chans_build, upscale=upscale)
        self.hrfeat = HRfeature(in_chans=super_in, mid_chans=super_mid, out_chans=super_mid)
        # Aggregate
        self.isaggre = isaggre
        if self.isaggre:
            self.aggre_height = nn.Conv2d(dec_in[-1]+geo_chans_mid, 1, 3,1,1) # kernel is set to 1

        # TODO: ADD geo-prior net for processing lont, lat, and alt
        self.geoprior = GeoNet(in_chans=geo_chans_in, mid_chans=geo_chans_mid)

    def forward(self, x, super_fea):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :param super_fea: directly generated from existing one, and may has Internal Covariate Shift
        :return:
        '''
        geo_fea = self.geoprior(x[:,self.in_chans:]) #

        encode_fea = self.encoder(x[:, :self.in_chans])
        super_fea = self.hrfeat(super_fea)

        height_fea = self.decoder1(*encode_fea)
        height_fea = torch.cat([height_fea, geo_fea], dim=1) # N C H W
        if self.isaggre:
            height_aggre = self.aggre_height(height_fea)
        height = self.reg(height_fea, super_fea)

        build = self.decoder2(*encode_fea)
        build = torch.cat([build, geo_fea], dim=1) # N C H W
        build = self.seg(build, super_fea)

        if self.isaggre:
            return height, build, height_aggre

        return height, build


# add refine layer
class SRRegress_Cls_feature_refine(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=7, classes=1, super_in=4, super_mid=64, upscale=4,
                 isaggre=False, chans_build=2,
                 hir_value=(0, 1, 4, 7, 10, 20, 30)):
        super().__init__()
        # self.super_in = super_in
        # Super-Resolution branches
        # self.super_res = EDSR_feaHR(n_colors=super_in, n_out=3, n_feats=super_mid)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        dec_in = (256, 128, 64, 32, 16)
        self.decoder1 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        self.decoder2 = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=dec_in,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)
        # Fuse the SR HR features and the result of decoder, then upsampler
        self.reg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1], mid_chans=dec_in[-1],
                           out_chans=1, upscale=upscale)
        self.seg = HRfuse_residual(hr_chans=super_mid, lr_chans=dec_in[-1], mid_chans=dec_in[-1],
                           out_chans=chans_build, upscale=upscale)
        self.hrfeat = HRfeature(in_chans=super_in, mid_chans=super_mid, out_chans=super_mid)
        # Aggregate
        self.isaggre = isaggre
        if self.isaggre:
            self.aggre_height = nn.Conv2d(super_mid, 1, 3,1,1) # kernel is set to 1
        # self.ishir = ishir
        # if self.ishir:
        #     self.hir_conv = nn.Conv2d(chans_build+1, 1, 3,1,1) # kernel is set to 1
            # self.hir_value = torch.tensor(hir*3)
        self.refine = Refine_residual(hr_chans=super_mid,
                                      lr_chans=chans_build+1,
                                      mid_chans=8,
                                      out_chans=1)

    def forward(self, x, super_fea):
        '''
        :param x: optical images, sentinel-2 and sar images, sentinel-1
        :param super_fea: directly generated from existing one, and may has Internal Covariate Shift
        :return:
        '''
        encode_fea = self.encoder(x)
        super_fea = self.hrfeat(super_fea)

        height_fea = self.decoder1(*encode_fea)
        if self.isaggre:
            height_aggre = self.aggre_height(height_fea)

        height = self.reg(height_fea, super_fea)

        build = self.decoder2(*encode_fea)
        build = self.seg(build, super_fea)

        height_refine = self.refine(torch.cat([height, build], dim=1), super_fea)

        if self.isaggre:
            return height, build, height_aggre, height_refine
        return height, build

# adaptive bins
'''
class SRRegress_Adabin(torch.nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", encoder_depth=5,
                 in_channels=4, classes=1, super_channels=4,
                 n_bins=256, min_val=0, max_val=150, norm='linear'):
        super().__init__()
        self.super_res = EDSR(n_colors=in_channels, n_out=super_channels)
        # Encoder-Decoder
        self.encoder = get_encoder(encoder_name, in_channels=super_channels,
                                   depth=encoder_depth, weights=encoder_weights,)
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None)

        # Adaptive bins
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.adaptive_bins_layer = mViT(16, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self,x):
        super_fea = self.super_res(x)
        encode_fea = self.encoder(super_fea)
        unet_out = self.decoder(*encode_fea)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return pred
'''

# def get_param_groups(self):
#     """ keep the same as segformer, except for the bn layer"""
#     param_groups = [[], [], []]  #
#
#     for name, param in list(self.encoder.named_parameters()):
#         if "bn" in name:
#             param_groups[1].append(param)
#         else:
#             param_groups[0].append(param)
#
#     for param in list(self.classifier.parameters()):
#         param_groups[2].append(param)
#     return param_groups


if __name__=="__main__":
    device = 'cuda'
    # model = smp.Unet(encoder_name = "resnet50", encoder_weights = "imagenet",
    #     in_channels=3, classes=2,).to(device)
    # model = SRRegress(encoder_name="efficientnet-b4").to(device)
    # model = SRRegress_Adabin(encoder_name="efficientnet-b4",
    #                          n_bins=256, min_val=0, max_val=150).to(device)
    # model = SRRegress_Cls(encoder_name="efficientnet-b4").to(device)

    # pretrained_path = r'D:\code\pretrained_weights\mit_b2.pth'
    # model = SRRegress_Cls_Segformer(encoder_name="mit_b2", pretrained_path=pretrained_path,
    #     in_channels=4, num_classes=1, super_channels=4).to(device)

    # model = SRRegress_Cls_feature(encoder_name="efficientnet-b4",
    #                                in_channels=7, super_in=4,
    #                                super_mid=16, upscale=4,
    #                               isaggre=True,
    #                             isunsup=True,
    #                               ).to(device)

    model = SRRegress_Cls_nosuper(encoder_name="efficientnet-b4",
                                   in_channels=7, upscale=4, isaggre=True, chans_build=7)

    # nparas = sum([p.numel() if n.startswith("de") else 0 for n, p in model.named_parameters()])
    nparas = sum([p.numel() for p in model.parameters()])
    print('nparams: %.2f'%(nparas/1e+6))

    # datao = torch.ones((2, 4, 64, 64)).to(device)
    # datas = torch.ones((2, 3, 64, 64)).to(device)
    # # datageo = torch.ones((2, 4, 64, 64)).to(device)
    #
    # super_fea = torch.ones((2, 4, 256, 256)).to(device)
    # pred = model.forward_unsup(torch.cat([datao, datas], dim=1), super_fea)
    #
    # criterion  = torch.nn.MSELoss()
    # loss = sum([criterion(i, i) for i in pred[-1]])
    #

    # 2023.12.14: no super-resolution module
    data = torch.ones((2,7,64,64))# .to(device)
    pred = model.forward(data)

    for i in pred:
        if isinstance(i, list):
            for k in i:
                print(k.shape)
        else:
            print(i.shape)

    # encoder: 17.55, decoder: 2.68, mit: 2.5, seg: 0.07, hr: 0.22
    # regress_cls: decoder 5.35, encoder: 17.55, super_res: 6.06
    # regress_segformerb2: decoder 1.05, encoder: 24.2,
    # SRRegress_Cls_decision: 24.49 M,
    # SRRegress_Cls_nosuper: nparams: 22.94