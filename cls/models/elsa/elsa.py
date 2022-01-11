"""
This file is modified from DDFNet: https://github.com/thefoxofsky/ddfnet
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from timm.models.layers import DropPath, trunc_normal_, Mlp

from . import elsa_ext, elsa_faster_ext


class ELSAFunctionCUDA(Function):
    @staticmethod
    def forward(ctx, features, channel_mul, channel_add, spatial_filter,
                kernel_size=3, dilation=1, stride=1, version=''):
        # check args
        assert features.is_cuda, 'input feature must be a CUDA tensor.'
        assert channel_mul.is_cuda, 'channel_mul must be a CUDA tensor.'
        assert channel_add.is_cuda, 'channel_add must be a CUDA tensor.'
        assert spatial_filter.is_cuda, 'spatial_filter must be a CUDA tensor.'

        # TODO: fix CUDA code to support HALF operation
        if features.dtype == torch.float16:
            features = features.float()
        if channel_mul.dtype == torch.float16:
            channel_mul = channel_mul.float()
        if channel_add.dtype == torch.float16:
            channel_add = channel_add.float()
        if spatial_filter.dtype == torch.float16:
            spatial_filter = spatial_filter.float()

        # check channel_filter size
        b, c, h, w = features.size()
        bc, cc, hc, wc = channel_mul.size()
        assert channel_mul.size() == channel_add.size(),\
            "channel_mul size {} does not match channel_add size {}".format(
                channel_mul.size(), channel_add.size())
        assert bc == b and cc == c,\
            "channel_mul size {} does not match feature size {}".format(
                channel_mul.size(), features.size())
        assert hc == kernel_size and wc == kernel_size,\
            "channel_mul size {} does not match kernel size {}".format(
                channel_mul.size(), kernel_size)

        # check spatial_filter size
        bs, cs, hs, ws, = spatial_filter.size()
        assert bs == b and hs == h // stride and ws == w // stride,\
            "spatial_filter size {} does not match feature size {} with stride {}".format(
                spatial_filter.size(), features.size(), stride)
        assert cs == kernel_size ** 2,\
            "spatial_filter size {} does not match kernel size {}".format(
                spatial_filter.size(), kernel_size)

        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1 and dilation >= 1 and stride >= 1

        features = features.contiguous()
        channel_mul = channel_mul.contiguous()
        channel_add = channel_add.contiguous()
        spatial_filter = spatial_filter.contiguous()

        # record important info
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.stride = stride

        # build output tensor
        output = features.new_zeros((b, c, h//stride, w//stride))

        # choose a suitable CUDA implementation based on the input feature, filter size, and combination type.
        if version == 'f':
            op_type = elsa_faster_ext
        elif version == 'o':
            op_type = elsa_ext
        elif kernel_size <= 5 and h >= 14 and w >= 14 and stride == 1:
            op_type = elsa_faster_ext
        else:
            op_type = elsa_ext

        op_type.forward(features, channel_mul, channel_add, spatial_filter,
                        kernel_size, dilation, stride, output)
        if features.requires_grad or channel_mul.requires_grad \
                or channel_add.requires_grad or spatial_filter.requires_grad:
            ctx.save_for_backward(features, channel_mul, channel_add, spatial_filter)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        # TODO: support HALF operation
        if grad_output.dtype == torch.float16:
            grad_output = grad_output.float()

        grad_output = grad_output.contiguous()

        kernel_size = ctx.kernel_size
        dilation = ctx.dilation
        stride = ctx.stride

        features, channel_mul, channel_add, spatial_filter = ctx.saved_tensors
        rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input = torch.zeros_like(features, requires_grad=False)
        rgrad_spatial_filter = torch.zeros_like(spatial_filter, requires_grad=False)
        grad_input = torch.zeros_like(features, requires_grad=False)
        grad_channel_mul = torch.zeros_like(channel_mul, requires_grad=False)
        grad_channel_add = torch.zeros_like(channel_add, requires_grad=False)
        grad_spatial_filter = torch.zeros_like(spatial_filter, requires_grad=False)

        # TODO: optimize backward CUDA code.
        elsa_ext.backward(grad_output, features, channel_mul, channel_add,
                          spatial_filter, kernel_size, dilation, stride,
                          rgrad_output, rgrad_input, rgrad_spatial_filter,
                          grad_input, grad_channel_mul, grad_channel_add, grad_spatial_filter)

        return grad_input, grad_channel_mul, grad_channel_add, grad_spatial_filter, None, None, None, None


elsa_funcgion_cuda = ELSAFunctionCUDA.apply


def elsa_op(features, channel_mul, channel_add, spatial_filter,
            kernel_size=3, dilation=1, stride=1, version=''):
    if features.is_cuda and channel_mul.is_cuda and channel_add.is_cuda and spatial_filter.is_cuda:
        return elsa_funcgion_cuda(features, channel_mul, channel_add, spatial_filter,
                                  kernel_size, dilation, stride, version)
    else:
        B, C, H, W = features.shape
        _pad = kernel_size // 2 * dilation
        features = Function.unfold(
            features, kernel_size=kernel_size, dilation=dilation, padding=_pad, stride=stride) \
            .reshape(B, C, kernel_size ** 2, H * W)
        channel_mul = channel_mul.reshape(B, C, kernel_size ** 2, 1)
        channel_add = channel_add.reshape(B, C, kernel_size ** 2, 1)
        spatial_filter = spatial_filter.reshape(B, 1, kernel_size ** 2, H * W)
        filters = channel_mul * spatial_filter + channel_add  # B, C, K, N
        return (features * filters).sum(2).reshape(B, C, H, W)


class ELSA(nn.Module):
    """
    Implementation of enhanced local self-attention
    """
    def __init__(self, dim, num_heads, dim_qk=None, dim_v=None, kernel_size=5,
                 stride=1, dilation=1, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., group_width=8, groups=1, lam=1,
                 gamma=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_qk = dim_qk or self.dim // 3 * 2
        self.dim_v = dim_v or dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        head_dim = self.dim_v // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if self.dim_qk % group_width != 0:
            self.dim_qk = math.ceil(float(self.dim_qk) / group_width) * group_width

        self.group_width = group_width
        self.groups = groups
        self.lam = lam
        self.gamma = gamma
        print(f'lambda = {lam}, gamma = {gamma}, scale = {self.scale}')

        self.pre_proj = nn.Conv2d(dim, self.dim_qk * 2 + self.dim_v, 1, bias=qkv_bias)
        self.attn = nn.Sequential(
            nn.Conv2d(self.dim_qk, self.dim_qk, kernel_size, padding=(kernel_size // 2)*dilation,
                      dilation=dilation, groups=self.dim_qk // group_width),
            nn.GELU(),
            nn.Conv2d(self.dim_qk, kernel_size ** 2 * num_heads, 1, groups=groups))

        if self.lam != 0 and self.gamma != 0:
            gh_mul = torch.randn(1, 1, self.dim_v, kernel_size, kernel_size)
            gh_add = torch.zeros(1, 1, self.dim_v, kernel_size, kernel_size)
            trunc_normal_(gh_add, std=.02)
            self.ghost_head = nn.Parameter(torch.cat((gh_mul, gh_add), dim=0), requires_grad=True)
        elif self.lam == 0 and self.gamma != 0:
            gh_add = torch.zeros(1, self.dim_v, kernel_size, kernel_size)
            trunc_normal_(gh_add, std=.02)
            self.ghost_head = nn.Parameter(gh_add, requires_grad=True)
        elif self.lam != 0 and self.gamma == 0:
            gh_mul = torch.randn(1, self.dim_v, kernel_size, kernel_size)
            self.ghost_head = nn.Parameter(gh_mul, requires_grad=True)
        else:
            self.ghost_head = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.post_proj = nn.Linear(self.dim_v, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, H, W, _ = x.shape
        C = self.dim_v
        ks = self.kernel_size
        G = self.num_heads
        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        qkv = self.pre_proj(x)

        q, k, v = torch.split(qkv, (self.dim_qk, self.dim_qk, self.dim_v), dim=1)
        quadratic = q * k * self.scale

        if self.stride > 1:
            quadratic = F.avg_pool2d(quadratic, self.stride)

        attn = self.attn(quadratic)

        v = v.reshape(B * G, C // G, H, W)
        attn = attn.reshape(B * G, -1, H, W).softmax(1)
        attn = self.attn_drop(attn)
        if self.lam != 0 and self.gamma != 0:
            gh = self.ghost_head.expand(2, B, C, ks, ks).reshape(2, B * G, C // G, ks, ks)
            gh_mul, gh_add = gh[0] ** self.lam, gh[1] * self.gamma
        elif self.lam == 0 and self.gamma != 0:
            gh_mul = torch.ones(B * G, C // G, ks, ks,
                                device=v.device, requires_grad=False)
            gh_add = self.ghost_head.expand(B, C, ks, ks).reshape(B * G, C // G, ks, ks) * self.gamma
        elif self.lam != 0 and self.gamma == 0:
            gh_mul = self.ghost_head.expand(B, C, ks, ks).reshape(B * G, C // G, ks, ks) ** self.lam
            gh_add = torch.zeros(B * G, C // G, ks, ks,
                                 device=v.device, requires_grad=False)
        else:
            gh_mul = torch.ones(B * G, C // G, ks, ks,
                                device=v.device, requires_grad=False)
            gh_add = torch.zeros(B * G, C // G, ks, ks,
                                 device=v.device, requires_grad=False)
        x = elsa_op(v, gh_mul, gh_add, attn, self.kernel_size, self.dilation, self.stride)
        x = x.reshape(B, C, H // self.stride, W // self.stride)
        x = self.post_proj(x.permute(0, 2, 3, 1))  # B, H, W, C
        x = self.proj_drop(x)
        return x


class ELSABlock(nn.Module):
    """
    Implementation of ELSA block: including ELSA + MLP
    """
    def __init__(self, dim, kernel_size,
                 stride=1, num_heads=1, mlp_ratio=3.,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=1, dim_qk=None, dim_v=None,
                 lam=1, gamma=1, dilation=1, group_width=8, groups=1,
                 **kwargs):
        super().__init__()
        assert stride == 1
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = ELSA(dim, num_heads, dim_qk=dim_qk, dim_v=dim_v, kernel_size=kernel_size,
                         stride=stride, dilation=dilation,
                         qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                         group_width=group_width, groups=groups, lam=lam, gamma=gamma, **kwargs)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x