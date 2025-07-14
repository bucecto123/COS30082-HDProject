import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(nn.Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MiniFASNet(nn.Module):
    def __init__(self, keep, embedding_size, conv6_kernel, drop_p=0.0, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])
        self.conv_23 = Depth_Wise(keep[1], keep[2], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])
        self.conv_3 = Residual(keep[2], num_block=keep[4], groups=keep[5], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(keep[2], keep[6], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[7])
        self.conv_4 = Residual(keep[6], num_block=keep[8], groups=keep[9], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(keep[6], keep[10], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[11])
        self.conv_5 = Residual(keep[10], num_block=keep[12], groups=keep[13], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6_sep = Conv_block(keep[10], keep[14], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv6_dw = Linear_block(keep[14], keep[14], groups=keep[14], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv6_sep(x)
        x = self.conv6_dw(x)
        x = self.conv6_flatten(x)
        if self.embedding_size > 0:
            x = self.linear(x)
            x = self.bn(x)
            x = self.drop(x)
        # x = self.prob(x)
        return x

def MiniFASNetV1(embedding_size=128, conv6_kernel=(7,7),drop_p=0.0,num_classes=3,img_channel=3):
    # Refer to paper Table 1.
    keep = [16, 16, 16, 32, 2, 32, 32, 64, 2, 64, 64, 128, 2, 128, 512]
    return MiniFASNet(keep, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)

def MiniFASNetV2(embedding_size=128, conv6_kernel=(7,7),drop_p=0.0,num_classes=3,img_channel=3):
    # Refer to paper Table 2.
    keep = [32, 32, 32, 64, 4, 64, 64, 128, 6, 128, 128, 128, 2, 128, 512]
    return MiniFASNet(keep, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Depth_Wise_SE(nn.Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise_SE, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se = SEModule(out_c)

     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)

        if self.residual:
            x = self.se(x)
            output = short_cut + x
        else:
            output = x
        return output


class Residual_SE(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual_SE, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise_SE(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)


class MiniFASNet_SE(nn.Module):
    def __init__(self, keep, embedding_size, conv6_kernel, drop_p=0.0, num_classes=3, img_channel=3):
        super(MiniFASNet_SE, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])
        self.conv_23 = Depth_Wise_SE(keep[1], keep[2], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])
        self.conv_3 = Residual_SE(keep[2], num_block=keep[4], groups=keep[5], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise_SE(keep[2], keep[6], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[7])
        self.conv_4 = Residual_SE(keep[6], num_block=keep[8], groups=keep[9], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise_SE(keep[6], keep[10], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[11])
        self.conv_5 = Residual_SE(keep[10], num_block=keep[12], groups=keep[13], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6_sep = Conv_block(keep[10], keep[14], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv6_dw = Linear_block(keep[14], keep[14], groups=keep[14], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv6_sep(x)
        x = self.conv6_dw(x)
        x = self.conv6_flatten(x)
        if self.embedding_size > 0:
            x = self.linear(x)
            x = self.bn(x)
            x = self.drop(x)
        # x = self.prob(x)
        return x


def MiniFASNetV1SE(embedding_size=128, conv6_kernel=(7,7),drop_p=0.0,num_classes=3,img_channel=3):
    # Refer to paper Table 1.
    keep = [16, 16, 16, 32, 2, 32, 32, 64, 2, 64, 64, 128, 2, 128, 512]
    return MiniFASNet_SE(keep, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)

def MiniFASNetV2SE(embedding_size=128, conv6_kernel=(7,7),drop_p=0.0,num_classes=3,img_channel=3):
    # Refer to paper Table 2.
    keep = [32, 32, 32, 64, 4, 64, 64, 128, 6, 128, 128, 128, 2, 128, 512]
    return MiniFASNet_SE(keep, embedding_size, conv6_kernel, drop_p, num_classes, img_channel)
