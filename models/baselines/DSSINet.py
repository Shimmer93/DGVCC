import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from collections import OrderedDict

def same_padding_length(input_length, filter_size, stride, dilation=1):
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    output_length = (input_length + stride - 1) // stride
    pad_length = max(0, (output_length - 1) * stride + dilated_filter_size - input_length)
    return pad_length

def compute_same_padding2d(input_shape, kernel_size, strides, dilation):
    space = input_shape[2:]
    assert len(space) == 2, "{}".format(space)
    new_space = []
    new_input = []
    for i in range(len(space)):
        pad_length = same_padding_length(
            space[i],
            kernel_size[i],
            stride=strides[i],
            dilation=dilation[i])
        new_space.append(pad_length)
        new_input.append(pad_length % 2)
    return tuple(new_space), tuple(new_input)

class Conv2d_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, dilation=1, bn=False, bias=True, groups=1):
        super(Conv2d_dilated, self).__init__()
        self.conv = _Conv2d_dilated(in_channels, out_channels, kernel_size, 'zeros', stride, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        else:
            self.relu = None
    
    @torch.cuda.amp.autocast()
    def forward(self, x, dilation=None):
        x = self.conv(x, dilation)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class _Conv2d_dilated(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode, stride=1, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        super(_Conv2d_dilated, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding_mode=padding_mode, 
            stride=stride, groups=groups, dilation=dilation, bias=bias, padding=_pair(0), output_padding=_pair(0), transposed=False)
    
    @torch.cuda.amp.autocast()
    def forward(self, input, dilation=None):
        input_shape = list(input.size())
        dilation_rate = self.dilation if dilation is None else _pair(dilation)
        padding, pad_input = compute_same_padding2d(input_shape, kernel_size=self.kernel_size, strides=self.stride, dilation=dilation_rate)

        if pad_input[0] == 1 or pad_input[1] == 1:
            input = F.pad(input, [0, int(pad_input[0]), 0, int(pad_input[1])])
        return F.conv2d(input, self.weight, self.bias, self.stride,
                       (padding[0] // 2, padding[1] // 2), dilation_rate, self.groups)
        #https://github.com/pytorch/pytorch/issues/3867
        
class SequentialEndpoints(nn.Module):

    def __init__(self, layers, endpoints=None):
        super(SequentialEndpoints, self).__init__()
        assert isinstance(layers, OrderedDict)
        for key, module in layers.items():
            self.add_module(key, module)
        if endpoints is not None:
            self.Endpoints = namedtuple('Endpoints', endpoints.values(), verbose=True)
            self.endpoints = endpoints


    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def sub_forward(self, startpoint, endpoint):
        def forward(input):
            flag = False
            for key, module in self._modules.items():
                if startpoint == endpoint:
                    output = input
                    if key == startpoint:
                        output = module(output)
                        return output
                elif flag or key == startpoint:
                    if key == startpoint:
                        output = input
                    flag = True
                    output = module(output)
                    if key == endpoint:
                        return output
            return output
        return forward
    
    @torch.cuda.amp.autocast()
    def forward(self, input, require_endpoints=False):
        if require_endpoints:
            endpoints = self.Endpoints([None] * len(self.endpoints.keys()))
        for key, module in self._modules.items():
            input = module(input)
            if require_endpoints and key in self.endpoints.keys():
                setattr(endpoints, self.endpoints[key], input)
        if require_endpoints:
            return input, endpoints
        else:
            return input

import math

#model_urls = {
#    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
#}


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, output_stride=8, base_dilated_rate=1, NL='relu', bias=True):
    layers = []
    in_channels = 3
    layers = OrderedDict()
    idx = 0
    curr_stride = 1
    dilated_rate = base_dilated_rate
    for v in cfg:
        name, ks, padding = str(idx), (3, 3), (1, 1)
        if type(v) is tuple:
            if len(v) == 2:
                v, ks = v
            elif len(v) == 3:
                name, v, ks = v
            elif len(v) == 4:
                name, v, ks, padding = v
        if v == 'M':
            if curr_stride >= output_stride:
                dilated_rate = 2
                curr_stride *= 2
            else:
                layers[name] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                curr_stride *= 2
            idx += 1
        elif v == 'None':
            idx += 1
        else:
            # conv2d = _Conv2d_dilated(in_channels, v, dilation=dilated_rate, kernel_size=ks, bias=bias)
            conv2d = nn.Conv2d(in_channels, v, dilation=dilated_rate, kernel_size=ks, padding=padding, bias=bias)
            dilated_rate = base_dilated_rate
            layers[name] = conv2d
            idx += 1
            if batch_norm:
                layers[str(idx)] = nn.BatchNorm2d(v)
                idx += 1
            if NL == 'relu' :
                relu = nn.ReLU(inplace=True)
            if NL == 'nrelu' :
                relu = nn.ReLU(inplace=False)
            elif NL == 'prelu':
                relu = nn.PReLU()
            layers['relu'+str(idx)] = relu
            idx += 1
            in_channels = v
    print("\n".join(["{}: {}-{}".format(i, k, v) for i, (k,v) in enumerate(layers.items())]))
    return SequentialEndpoints(layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'G': [64, 64, 'M', 128, 128, 'M', 256, 256, 256],
    'H': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'None', 512, 512, 512, 'None', 512, 512, 512],
    'I': [24, 22, 'M', 41, 51, 'M', 108, 89, 111, 'M', 184, 276, 228],
}

def vgg16(struct='F', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg[struct], **kwargs))

    return model

import torch.nn.functional as F

## CRFFeatureRF
class MessagePassing(nn.Module):
    def __init__(self, branch_n, input_ncs, bn=False):
        super(MessagePassing, self).__init__()
        self.branch_n = branch_n
        self.iters = 2
        for i in range(branch_n):
            for j in range(branch_n):
                if i == j:
                    continue
                setattr(self, "w_0_{}_{}_0".format(j, i),                         nn.Sequential(
                                Conv2d_dilated(input_ncs[j],  input_ncs[i], 1, dilation=1, same_padding=True, NL=None, bn=bn),
                            )
                        )
        self.relu = nn.ReLU(inplace=False)
        self.prelu = nn.PReLU()
    
    @torch.cuda.amp.autocast()
    def forward(self, input):
        hidden_state = input
        side_state = []

        for _ in range(self.iters):
            hidden_state_new = []
            for i in range(self.branch_n):

                unary = hidden_state[i]
                binary = None
                for j in range(self.branch_n):
                    if i == j:
                        continue
                    if binary is None:
                        binary = getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j])
                    else:
                        binary = binary + getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j])

                binary = self.prelu(binary)
                hidden_state_new += [self.relu(unary + binary)]
            hidden_state = hidden_state_new

        return hidden_state

class CRFVGG(nn.Module):
    def __init__(self, output_stride=8, bn=False):
        super(CRFVGG, self).__init__()

        self.output_stride = output_stride

        self.pyramid = [2, 0.5]

        self.front_end = vgg16(struct='F', NL="prelu", output_stride=self.output_stride)


        self.passing1 = MessagePassing( branch_n=2, 
                                        input_ncs=[128, 64],
                                        )
        self.passing2 = MessagePassing( branch_n=3, 
                                        input_ncs=[256, 128, 64],
                                        )
        self.passing3 = MessagePassing( branch_n=3, 
                                        input_ncs=[512, 256, 128]
                                        )
        self.passing4 = MessagePassing( branch_n=2, 
                                        input_ncs=[512, 256]
                                        )


        self.decoder1 = nn.Sequential(
                Conv2d_dilated(512, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder2 = nn.Sequential(
                Conv2d_dilated(768, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder3 = nn.Sequential(
                Conv2d_dilated(896, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder4 = nn.Sequential(
                Conv2d_dilated(448, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder5 = nn.Sequential(
                Conv2d_dilated(192, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.passing_weight1 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight2 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight3 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight4 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()

    @torch.cuda.amp.autocast()
    def forward(self, im_data, return_feature=False):
        conv1_2 = ['0', 'relu3']
        conv1_2_na = ['0', '2']
        conv2_2 = ['4', 'relu8']
        conv2_2_na = ['4', '7']
        conv3_3 = ['9', 'relu15']
        conv3_3_na = ['9', '14']
        # layer 16 is the max pooling layer
        conv4_3 = ['16', 'relu22']
        conv4_3_na = ['16', '21']
        # droping the last pooling layer, 17 would become dilated with rate 2
        # conv4_3 = ['17', 'relu22']

        batch_size, C, H, W = im_data.shape

        with torch.no_grad():
            im_scale1 = nn.functional.upsample(im_data, size=(int(H * self.pyramid[0]), int(W * self.pyramid[0])), align_corners=False, mode="bilinear")
            im_scale2 = im_data
            im_scale3 = nn.functional.upsample(im_data, size=(int(H * self.pyramid[1]), int(W * self.pyramid[1])), align_corners=False, mode="bilinear")


        mp_scale1_feature_conv2_na = self.front_end.features.sub_forward(conv1_2[0], conv2_2_na[1])(im_scale1)
        mp_scale2_feature_conv1_na = self.front_end.features.sub_forward(*conv1_2_na)(im_scale2)

        mp_scale1_feature_conv2, mp_scale2_feature_conv1 \
                        = self.passing1([mp_scale1_feature_conv2_na, mp_scale2_feature_conv1_na])


        aggregation4 = torch.cat([mp_scale1_feature_conv2, mp_scale2_feature_conv1], dim=1)

        mp_scale1_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale1_feature_conv2)
        mp_scale2_feature_conv2_na = self.front_end.features.sub_forward(*conv2_2_na)(mp_scale2_feature_conv1)
        mp_scale3_feature_conv1_na = self.front_end.features.sub_forward(*conv1_2_na)(im_scale3)


        mp_scale1_feature_conv3, mp_scale2_feature_conv2, mp_scale3_feature_conv1 \
                        = self.passing2([mp_scale1_feature_conv3_na, mp_scale2_feature_conv2_na, mp_scale3_feature_conv1_na])
        aggregation3 = torch.cat([mp_scale1_feature_conv3, mp_scale2_feature_conv2, mp_scale3_feature_conv1], dim=1)


        mp_scale1_feature_conv4_na = self.front_end.features.sub_forward(*conv4_3_na)(mp_scale1_feature_conv3)
        mp_scale2_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale2_feature_conv2)
        mp_scale3_feature_conv2_na = self.front_end.features.sub_forward(*conv2_2_na)(mp_scale3_feature_conv1)

        mp_scale1_feature_conv4, mp_scale2_feature_conv3, mp_scale3_feature_conv2 \
                        = self.passing3([mp_scale1_feature_conv4_na, mp_scale2_feature_conv3_na, mp_scale3_feature_conv2_na])
        aggregation2 = torch.cat([mp_scale1_feature_conv4, mp_scale2_feature_conv3, mp_scale3_feature_conv2], dim=1)

        mp_scale2_feature_conv4_na = self.front_end.features.sub_forward(*conv4_3_na)(mp_scale2_feature_conv3)
        mp_scale3_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale3_feature_conv2)

        mp_scale2_feature_conv4, mp_scale3_feature_conv3 \
                        = self.passing4([mp_scale2_feature_conv4_na, mp_scale3_feature_conv3_na])
        aggregation1 = torch.cat([mp_scale2_feature_conv4, mp_scale3_feature_conv3], dim=1)

        mp_scale3_feature_conv4 = self.front_end.features.sub_forward(*conv4_3)(mp_scale3_feature_conv3)

        dens1 = self.decoder1(mp_scale3_feature_conv4)
        dens2 = self.decoder2(aggregation1)
        dens3 = self.decoder3(aggregation2)
        dens4 = self.decoder4(aggregation3)
        dens5 = self.decoder5(aggregation4)

        dens1 = self.prelu(dens1)
        dens2 = self.prelu(dens2 + self.passing_weight1(nn.functional.upsample(dens1, scale_factor=2, align_corners=False, mode="bilinear")))
        dens3 = self.prelu(dens3 + self.passing_weight2(nn.functional.upsample(dens2, scale_factor=2, align_corners=False, mode="bilinear")))
        dens4 = self.prelu(dens4 + self.passing_weight3(nn.functional.upsample(dens3, scale_factor=2, align_corners=False, mode="bilinear")))
        dens5 = self.relu(dens5 + self.passing_weight4(nn.functional.upsample(dens4, scale_factor=2, align_corners=False, mode="bilinear")))

        return dens5

if __name__ == '__main__':
    model = CRFVGG()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.shape)