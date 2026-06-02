import torch
import torch.nn as nn


class Oriented_LK_EMA(nn.Module):
    def __init__(self, channels, factor=8, direction='both'):


        super(Oriented_LK_EMA, self).__init__()
        self.groups = factor
        self.direction = direction
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        

        
        if direction in ['both', 'h']:
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        if direction in ['both', 'v']:
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        

        self.conv_large = nn.Conv2d(
            channels // self.groups, 
            channels // self.groups, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=channels // self.groups
        )

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        

        if self.direction == 'h':

            x_h = self.pool_h(group_x) 
            att_h = self.conv1x1(x_h).sigmoid()
            x1 = self.gn(group_x * att_h)
            
        elif self.direction == 'v':

            x_w = self.pool_w(group_x)

            att_w = self.conv1x1(x_w).sigmoid()

            x1 = self.gn(group_x * att_w)
            
        else: # 'both' (标准模式)
            x_h = self.pool_h(group_x)
            x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
            hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
            x_h, x_w = torch.split(hw, [h, w], dim=2)
            x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        x2 = self.conv_large(group_x)
        
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class AdaptiveSimAM(torch.nn.Module):
    def __init__(self, channels, init_lambda=1e-4):
        super(AdaptiveSimAM, self).__init__()
        self.act = nn.Sigmoid()
        self.learnable_lambda = nn.Parameter(torch.ones(1, channels, 1, 1) * init_lambda)

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        safe_lambda = torch.abs(self.learnable_lambda) + 1e-5
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + safe_lambda)) + 0.5
        return x * self.act(y)


class OGWA(nn.Module):
    def __init__(self, channels, mode='direction', direction='both'):


        super(OGWA, self).__init__()
        self.mode = mode
        
        if mode == 'direction':

            self.att = Oriented_LK_EMA(channels, direction=direction)
        elif mode == 'sparse':
            self.att = AdaptiveSimAM(channels)
        else:
            raise ValueError("Mode must be 'direction' or 'sparse'")
            
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        return x + self.alpha * self.att(x)