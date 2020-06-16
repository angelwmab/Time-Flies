import torch
import torch.nn as nn
import numpy as np
from torchvision import models

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='sum'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

class Encoder(nn.Module):
    def __init__(self, seq_len):
        super(Encoder, self).__init__()

        self.vid_first = nn.Sequential(
            nn.Conv2d(3 * seq_len, 3 * seq_len, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3 * seq_len, 64, (3, 3))
            )
        
        vid_features = models.vgg19(pretrained=True).features
        self.vid_enc_1 = nn.Sequential()
        self.vid_enc_2 = nn.Sequential()

        # vgg_input -> relu1_1
        for x in range(1, 2):
            self.vid_enc_1.add_module(str(x), vid_features[x])
            
        # relu1_1 -> relu2_1
        for x in range(2, 7):
            self.vid_enc_2.add_module(str(x), vid_features[x])
        
        target_features = models.vgg19(pretrained=True).features
        self.target_enc_1 = nn.Sequential()
        self.target_enc_2_0 = nn.Sequential()
        self.target_enc_2_1 = nn.Sequential()
        self.target_enc_2_2 = nn.Sequential()
        self.target_enc_3_0 = nn.Sequential()
        self.target_enc_3_1 = nn.Sequential()
        self.target_enc_3_2 = nn.Sequential()
        self.target_enc_4_0 = nn.Sequential()
        self.target_enc_4_1 = nn.Sequential()
        self.target_enc_4_2 = nn.Sequential()

        # vgg_input -> relu1_1
        for x in range(0, 2):
            self.target_enc_1.add_module(str(x), target_features[x])
        
        # relu1_1 -> relu2_1
        for x in range(2, 4):
            self.target_enc_2_0.add_module(str(x), target_features[x])
        self.target_enc_2_1.add_module(str(4), WavePool(64))
        for x in range(5, 7):
            self.target_enc_2_2.add_module(str(x), target_features[x])
        
        # relu2_1 -> relu3_1
        for x in range(7, 9):
            self.target_enc_3_0.add_module(str(x), target_features[x])
        self.target_enc_3_1.add_module(str(9), WavePool(128))
        for x in range(10, 12):
            self.target_enc_3_2.add_module(str(x), target_features[x])
        
        # relu3_1 -> relu4_1
        for x in range(12, 17):
            self.target_enc_4_0.add_module(str(x), target_features[x])
        self.target_enc_4_1.add_module(str(18), WavePool(256))
        for x in range(19, 21):
            self.target_enc_4_2.add_module(str(x), target_features[x])
        
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        
        # don't need the gradients, just want the features
        for name in ['target_enc_1', 'target_enc_2_0','target_enc_2_2', 'target_enc_3_0', 'target_enc_3_2', 'target_enc_4_0', 'target_enc_4_2']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1 from video, and relu3_1, relu4_1 from target image
    def encode(self, vid, target):
        skips = list()
        func = getattr(self, 'vid_enc_1')
        results = [func(vid)]
        
        func = getattr(self, 'vid_enc_2')
        results.append(func(results[-1]))
        
        func = getattr(self, 'target_enc_1')
        temp = func(target)
        
        func = getattr(self, 'target_enc_2_0')
        temp = func(temp)
        LL, LH, HL, HH = self.target_enc_2_1(temp)
        skips.append([LH, HL, HH])
        func = getattr(self, 'target_enc_2_2')
        temp = func(LL)
        
        func = getattr(self, 'target_enc_3_0')
        temp = func(temp)
        LL, LH, HL, HH = self.target_enc_3_1(temp)
        skips.append([LH, HL, HH])
        func = getattr(self, 'target_enc_3_2')
        temp = func(LL)
        
        func = getattr(self, 'target_enc_4_0')
        temp = func(temp)
        LL, LH, HL, HH = self.target_enc_4_1(temp)
        skips.append([LH, HL, HH])
        func = getattr(self, 'target_enc_4_2')
        results.append(func(LL))

        return results, skips
    
    def vgg_features(self, input):
        func = getattr(self, 'target_enc_1')
        results = [func(input)]
        
        func = getattr(self, 'target_enc_2_0')
        temp = func(results[-1])
        temp = self.maxpool(temp)
        func = getattr(self, 'target_enc_2_2')
        results.append(func(temp))
        
        func = getattr(self, 'target_enc_3_0')
        temp = func(results[-1])
        temp = self.maxpool(temp)
        func = getattr(self, 'target_enc_3_2')
        results.append(func(temp))
        
        func = getattr(self, 'target_enc_4_0')
        temp = func(results[-1])
        temp = self.maxpool(temp)
        func = getattr(self, 'target_enc_4_2')
        results.append(func(temp))
        
        return results

    def forward(self, target_in, vid_in=None):
        if vid_in is None:
            return self.vgg_features(target_in)
        else:
            vid_feat = self.vid_first(vid_in)
            features, skips = self.encode(vid_feat, target_in)
            return features, skips


class Decoder1(nn.Module):
    """docstring for Decoder1"""
    def __init__(self, GPU):
        super(Decoder1, self).__init__()

        self.GPU = GPU

        self.dec_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
        )
        self.unpool_1 = WaveUnpool(256)
        self.dec_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
        )
        self.unpool_2 = WaveUnpool(128)

        self.dec_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
        )
        self.unpool_3 = WaveUnpool(64)
        
        self.pad = nn.ReflectionPad2d(1)
        self.noise = None
    
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_copy = feat.view(-1)
        feat_copy = feat_copy[feat_copy.nonzero()]
        feat_var = feat_copy.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat_copy.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

        if self.noise is not None:
            style_mean = style_mean.expand(size) * self.noise.expand(size).cuda(self.GPU)

        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def downsample(self, input, times):
        ds = nn.MaxPool2d((1, 1), (2, 2), (0, 0))
        output = input.clone()

        with torch.no_grad():
            for i in range(times):
                output = ds(output)

        return output

    def AdaIN_per_class(self, content_feat, style_feat, content_mask, style_mask, c_ds=False, s_ds=False, eps=1e-5):
        res = torch.zeros_like(content_feat)
        class_n = content_mask.size(0)
        for k in range(class_n):
            if content_mask[k].max() != 0:
                if c_ds and s_ds:
                    if style_mask[k].max() == 0:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps))
                    else:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        s_mask = self.downsample(style_mask[k].unsqueeze(0), 1)[:, :style_feat.size(2), :style_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps)*s_mask)
                elif c_ds:
                    if style_mask[k].max() == 0:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps))
                    else:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps)*style_mask[k])
                else:
                    if style_mask[k].max() == 0:
                        res += content_mask[k] * self.adaptive_instance_normalization((content_feat+eps)*content_mask[k], (style_feat+eps))
                    else:
                        res += content_mask[k] * self.adaptive_instance_normalization((content_feat+eps)*content_mask[k], (style_feat+eps)*style_mask[k])
        return res

    def forward(self, features, skips, content_mask, style_mask, noise=None):

        if noise is not None:
            self.noise = noise.unsqueeze(0).unsqueeze(0)

        output = self.dec_1(features[2]) # decode relu_4-1 of target image
        LH, HL, HH = skips[2]
        output = self.unpool_1(output, LH, HL, HH)
        output = self.pad(output)

        # resize
        output = output[:, :, :skips[1][2].size(2), :skips[1][2].size(3)]

        output = self.dec_2(output)
        LH, HL, HH = skips[1]
        output = self.unpool_2(output, LH, HL, HH)
        output = self.pad(output)

        # resize
        output = output[:, :, :features[1].size(2), :features[1].size(3)]

        adain = self.AdaIN_per_class(output, features[1], content_mask, style_mask, True, True)
        output = self.dec_3(adain)
        LH, HL, HH = skips[0]
        output = self.unpool_3(output, LH, HL, HH)
        output = self.pad(output)

        return output


class Decoder2(nn.Module):
    """docstring for Decoder2"""
    def __init__(self, GPU, ratio=0.5):
        super(Decoder2, self).__init__()

        self.GPU = GPU
        self.ratio = ratio
        
        self.dec_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.noise = None
    
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_copy = feat.view(-1).cuda(self.GPU)
        feat_copy = feat_copy[feat_copy.nonzero()]
        feat_var = feat_copy.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat_copy.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

        if self.noise is not None:
            style_mean = style_mean.expand(size) * self.noise.expand(size).cuda(self.GPU)

        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def downsample(self, input, times):
        ds = nn.MaxPool2d((1, 1), (2, 2), (0, 0))
        output = input.clone()

        with torch.no_grad():
            for i in range(times):
                output = ds(output)

        return output

    def AdaIN_per_class(self, content_feat, style_feat, content_mask, style_mask, c_ds=False, s_ds=False, eps=1e-5):
        res = torch.zeros_like(content_feat)
        class_n = content_mask.size(0)
        for k in range(class_n):
            if content_mask[k].max() != 0:
                if c_ds and s_ds:
                    if style_mask[k].max() == 0:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps))
                    else:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        s_mask = self.downsample(style_mask[k].unsqueeze(0), 1)[:, :style_feat.size(2), :style_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps)*s_mask)
                elif c_ds:
                    if style_mask[k].max() == 0:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps))
                    else:
                        c_mask = self.downsample(content_mask[k].unsqueeze(0), 1)[:, :content_feat.size(2), :content_feat.size(3)]
                        res += c_mask * self.adaptive_instance_normalization((content_feat+eps)*c_mask, (style_feat+eps)*style_mask[k])
                else:
                    if style_mask[k].max() == 0:
                        res += content_mask[k] * self.adaptive_instance_normalization((content_feat+eps)*content_mask[k], (style_feat+eps))
                    else:
                        res += content_mask[k] * self.adaptive_instance_normalization((content_feat+eps)*content_mask[k], (style_feat+eps)*style_mask[k])
        return res

    def forward(self, out, feature, content_mask, style_mask, noise=None):
        if noise is not None:
            self.noise = noise.unsqueeze(0).unsqueeze(0)

        # resize
        output = out[:, :, :feature.size(2), :feature.size(3)]
        content_mask = content_mask[:, :feature.size(2), :feature.size(3)]
        style_mask = style_mask[:, :feature.size(2), :feature.size(3)]

        adain = self.AdaIN_per_class(output, feature, content_mask, style_mask)
        output = self.dec_4(self.ratio*output + (1-self.ratio)*adain)

        return output

class Realness_Smoothness_discriminator(nn.Module):
    """docstring for Realness_Smoothness_discriminator"""
    def __init__(self, seq_len):
        super(Realness_Smoothness_discriminator, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(6 * seq_len, 32, (1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, (3, 3)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.Conv2d(64, 64, (3, 3)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.Conv2d(64, 128, (3, 3)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, (3, 3)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.linear = nn.Sequential(
            nn.Linear(123008, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )

    def forward(self, original_vid, real_fake_vid):
        input = torch.cat((original_vid, real_fake_vid), 1)
        in_size = input.size(0)
        output = self.convolution(input)
        output = output.view(in_size, -1)
        output = self.linear(output)

        return output.view(-1)