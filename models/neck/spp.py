import torch
import torch.nn as nn

from ..basic.conv import Conv


# Spatial Pyramid Pooling
class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=[5, 9, 13], norm_type='BN', act_type='relu'):
        super(SPP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                for k in pooling_size
            ]
        )
        
        self.cv2 = Conv(inter_dim*(len(pooling_size) + 1), out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.cv2(x)

        return x


# SPP block with CSP module
class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=[5, 9, 13], act_type='relu', norm_type='BN'):
        super(SPPBlockCSP, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None)
        )
        inter_dim = int(out_dim * expand_ratio)
        self.cv1 = Conv(out_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(out_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SPP(inter_dim, 
                inter_dim, 
                expand_ratio=1.0, 
                pooling_size=pooling_size, 
                act_type=act_type, 
                norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        )
        self.cv3 = Conv(inter_dim * 2, out_dim, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x = self.projector(x)
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y

