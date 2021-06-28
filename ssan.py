import torch
from torch import nn, einsum
from einops import rearrange
import timm

class SSA(nn.Module):
    def __init__(self, dim, n_segment):
        super(SSA, self).__init__()
        self.scale = dim ** -0.5
        self.n_segment = n_segment

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size = 1)
        self.attend = nn.Softmax(dim = -1)
        self.to_temporal_qk = nn.Conv3d(dim, dim * 2, 
                                  kernel_size=(3, 1, 1), 
                                  padding=(1, 0, 0))

    def forward(self, x):
        bt, c, h, w = x.shape
        t = self.n_segment
        b = bt / t
        # Spatial Attention:
        qkv = self.to_qkv(x) 
        q, k, v = qkv.chunk(3, dim = 1) # bt, c, h, w
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v)) # bt, hw, c
        #  -pixel attention
        pixel_dots = einsum('b i c, b j c -> b i j', q, k) # * self.scale
        pixel_attn = torch.softmax(pixel_dots, dim=-1)
        pixel_out = einsum('b i j, b j d -> b i d', pixel_attn, v)
        #  -channel attention
        chan_dots = einsum('b i c, b i k -> b c k', q, k) # * self.scale # c x c
        chan_attn = torch.softmax(chan_dots, dim=-1)
        chan_out = einsum('b i j, b d j -> b d i', chan_attn, v) # hw, c
        
        # aggregation
        x_hat = pixel_out + chan_out
        x_hat = rearrange(x_hat, '(b t) (h w) c -> b c t h w', t=t, h=h, w=w)
        
        # Temporal attention
        t_qk = self.to_temporal_qkv(x_hat)
        tq, tk = t_qk.chunk(2, dim=1) # b, c, t, h, w
        tq, tk = map(lambda t: rearrange(t, 'b c t h w -> b t (c h w )'), (tq, tk)) # b, t, d
        tv = rearrange(v, '(b t) (h w) c -> b t (c h w)', t=t, h=h, w=w) # shared value embedding
        dots = einsum('b i d, b j d -> b i j', tq, tk) # txt
        attn = torch.softmax(dots, dim=-1)
        out = einsum('b k t, b t d -> b k d', attn, tv) # txd
        out = rearrange(out, 'b t (c h w) -> (b t) c h w', h=h,w=w,c=c)
        return out


class SSAWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(SSAWrapper, self).__init__()
        self.block = block
        self.ssa = SSA(block.bn1.num_features, n_segment)
        self.n_segment = n_segment
        self.downsample = block.downsample
    
    def forward(self, x):
        residual = x

        for idx, subm in enumerate(self.block.children()):
            if idx < 3: x = subm(x) # 1: conv->bn->relu

        x = self.ssa(x)

        for idx, subm in enumerate(self.block.children()):
            if idx < 3 or idx > 7: continue
            x = subm(x)             # 2,3: conv->bn->relu->conv->bn

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual               # shortcut
        x = self.block.act3(x)      # act
        return x


class SSAN(nn.Module):
    def __init__(self, n_segment, net):
        super(SSAN, self).__init__()
        self.n_segment = n_segment
        # modify res2 and res3        
        net.layer2 = nn.Sequential(
                SSAWrapper(net.layer2[0], n_segment),
                net.layer2[1],
                SSAWrapper(net.layer2[2], n_segment),
                net.layer2[3],
            )
        net.layer3 = nn.Sequential(
                SSAWrapper(net.layer3[0], n_segment),
                net.layer3[1],
                SSAWrapper(net.layer3[2], n_segment),
                net.layer3[3],
                SSAWrapper(net.layer3[4], n_segment),
                net.layer3[5],
            )
        self.backbone = net
        self.avgpool = net.global_pool
        self.fc = net.fc
    
    def forward(self, x): # BT, C, H, W
        x = self.backbone.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ssan50(pretrained=False, n_segment=8, **kwargs):
    """Constructs a SSAN model.
    part of the SSAN model refers to the ResNet-50.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    net = timm.create_model('resnet50', pretrained=pretrained)
    model = SSAN(n_segment=n_segment, net=net)
    return model


if __name__ == '__main__':
    x = torch.randn(8, 3, 224, 224) # bt, c, h, w
    backbone = timm.create_model('resnet50', pretrained=False)
    model = SSAN(n_segment=4, net=backbone)
    y = model(x) # (bt, num_classes)
    print(y.shape)

