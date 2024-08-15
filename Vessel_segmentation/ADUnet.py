import math
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import *
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

from einops.layers.torch import Rearrange


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            moduleation(bool, optional): If True, Modulated Defromable Convolution(Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # self.p_conv偏置层，学习公式（2）中的偏移量。
        # 2*kernel_size*kernel_size：代表了卷积核中所有元素的偏移坐标，因为同时存在x和y的偏移，故要乘以2。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            # self.m_conv权重学习层，是后来提出的第二个版本的卷积也就是公式（3）描述的卷积。
            # kernel_size*kernel_size：代表了卷积核中每个元素的权重。
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    # 生成卷积核的邻域坐标
    def _get_p_n(self, N, dtype):
        """
        torch.meshgrid():Creates grids of coordinates specified by the 1D inputs in attr:tensors.
        功能是生成网格，可以用于生成坐标。
        函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，
        列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。

        其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；
        第二个输出张量填充第二个输入张量中的元素各列元素相同。
        """
        """
        torch.meshgrid用于生成网格坐标，https://blog.csdn.net/weixin_39504171/article/details/106356977
        生成卷积的相对位置
        """
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # p_n ===>offsets_x(kernel_size*kernel_size,) concat offsets_y(kernel_size*kernel_size,)
        #     ===> (2*kernel_size*kernel_size,)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # （1， 2*kernel_size*kernel_size, 1, 1）
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # 获取卷积核在feature map上所有对应的中心坐标，也就是p0
    """
    生成坐标的绝对位置
    """

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # (b, 2*kernel_size, h, w)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # 将获取的相对坐标信息与中心坐标相加就获得了卷积核的所有坐标。
    # 再加上之前学习得到的offset后，就是加上了偏移量后的坐标信息。
    # 即对应论文中公式(2)中的(p0+pn+Δpn)
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # p_n ===> (1, 2*kernel_size*kernel_size, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # p_0 ===> (1, 2*kernel_size*kernel_size, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # (1, 2*kernel_size*kernel_size, h, w)
        p = p_0 + p_n + offset  # 绝对位置＋相对位置+偏置
        return p

    def _get_x_q(self, x, q, N):
        # b, h, w, 2*kerel_size*kernel_size
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # x ===> (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # 因为x是与h轴方向平行，y是与w轴方向平行。故将2D卷积核投影到1D上，位移公式如下：
        # 各个卷积核中心坐标及邻域坐标的索引 offsets_x * w + offsets_y
        # (b, h, w, kernel_size*kernel_size)
        index = q[..., :N] * padded_w + q[..., N:]
        # index = q[..., :N]  + q[..., N:]
        # (b, c, h, w, kernel_size*kernel_size) ===> (b, c, h*w*kernel_size*kernel_size)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # (b, c, h*w)
        # x_offset[0][0][0] = x[0][0][index[0][0][0]]
        # index[i][j][k]的值应该是一一对应着输入x的(h*w)的坐标，且在之前将index[i][j][k]的值clamp在[0, h]及[0, w]范围里。
        # (b, c, h, w, kernel_size*kernel_size)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # (b, c, h, w, kernel_size*kernel_size)
        b, c, h, w, N = x_offset.size()
        # (b, c, h, w*kernel_size)
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        # (b, c, h*kernel_size, w*kernel_size)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

    def forward(self, x):
        # (b, c, h, w) ===> (b, 2*kernel_size*kernel_size, h, w)
        offset = self.p_conv(x)
        if self.modulation:
            # (b, c, h, w) ===> (b, kernel_size*kernel_size, h, w)
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        # kernel_size*kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)
        # (b, 2*kernel_size*kernel_size, h, w)
        p = self._get_p(offset, dtype)
        # (b, h, w, 2*kernel_size*kernel_size)
        p = p.contiguous().permute(0, 2, 3, 1)
        """
        在这里添加一个Transformer不知道效果怎么样
        """
        # 将p从tensor的前向计算中取出来，并向下取整得到左上角坐标q_lt。
        q_lt = p.detach().floor()
        # 将p向上再取整，得到右下角坐标q_rb。
        q_rb = q_lt + 1
        # 学习的偏移量是float类型，需要用双线性插值的方法去推算相应的值。
        # 同时防止偏移量太大，超出feature map，故需要torch.clamp来约束。
        # Clamps all elements in input into the range [ min, max ].
        # torch.clamp(a, min=-0.5, max=0.5)

        # p左上角x方向的偏移量不超过h,y方向的偏移量不超过w。
        """
        迫使x和y的大小锁定在图像原本的范围内
        """
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # p右下角x方向的偏移量不超过h,y方向的偏移量不超过w。
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        """
        通过组合不同的左上角和右下角的位置来合并左下角和右上角
        p左上角的x方向的偏移量和右下角y方向的偏移量组合起来，得到p左下角的值。
        """
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # p右下角的x方向的偏移量和左上角y方向的偏移量组合起来，得到p右上角的值。
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p。
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # 双线性插值公式里的四个系数。即bilinear kernel。
        # 作者代码为了保持整齐，每行的变量计算形式一样，所以计算需要做一点对应变量的对应变化。
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # 计算双线性插值的四个坐标对应的像素值。
        # (b, c, h, w, kernel_size*kernel_size)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 双线性插值的最后计算
        # (b, c, h, w, kernel_size*kernel_size)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            # (b, kernel_size*kernel_size, h, w) ===> (b, h, w, kernel_size*kernel_size)
            m = m.contiguous().permute(0, 2, 3, 1)
            # (b, h, w, kernel_size*kernel_size) ===>  (b, 1, h, w, kernel_size*kernel_size)
            m = m.unsqueeze(dim=1)
            # (b, c, h, w, kernel_size*kernel_size)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        # x_offset: (b, c, h, w, kernel_size*kernel_size)
        # x_offset: (b, c, h*kernel_size, w*kernel_size)
        x_offset = self._reshape_x_offset(x_offset, ks)
        # out: (b, c, h, w)
        out = self.conv(x_offset)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DeformConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DeformConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("x1",x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.conv2patch = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=4, stride=4),
            nn.GELU(),
            # nn.ReLU(),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        x = self.conv2patch(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class oneXone_conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(oneXone_conv, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(out_features)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.drop(x)
        x = self.Conv2(x)
        x = self.drop(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup=None, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        oup = oup or inp
        init_channels = math.ceil(oup // ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class GhostModule_Up(nn.Module):
    def __init__(self, inp, oup=None, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule_Up, self).__init__()
        oup = oup or inp
        init_channels = inp
        new_channels = init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, C_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.softmax = Softmax(dim=-1)
        self.c_lambda = C_lambda
        self.activaton = nn.Sigmoid()

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Conv2d(dim // 12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim // 12),
            nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, C // 12, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C // 12, -1).permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, C // 12, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C // 12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out


class PAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, P_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Conv2d(dim // 12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim // 12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.softmax = Softmax(dim=-1)

        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C // 12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out


class CHAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, P_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(CHAM_Module, self).__init__()
        self.chanel_in = in_dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(
            nn.Conv2d(dim // 12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim // 12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.softmax = Softmax(dim=-1)

        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, C // 12 * height, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C // 12 * height, -1).permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, C // 12 * height, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C // 12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out


class CWAM_Module(Module):
    def __init__(self, in_dim, dim, num_heads, qk_scale=None, P_lambda=1e-4, attn_drop=0., proj_drop=0.):
        super(CWAM_Module, self).__init__()
        self.chanel_in = in_dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim),
                                  nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=-1)

        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        proj_query = x.view(m_batchsize, C * width, -1)
        proj_key = x.view(m_batchsize, C * width, -1).permute(0, 2, 1)
        proj_value = x.view(m_batchsize, C * width, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out


class WindowAttention_ACAM(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim

        self.C_C = CAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.H_W = PAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.C_H = CHAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.C_W = CWAM_Module(self.dim, dim=dim, num_heads=num_heads)

        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))
        self.gamma3 = Parameter(torch.ones(1) * 0.5)
        self.gamma4 = Parameter(torch.ones(1) * 0.5)

    def _build_projection(self, dim_in, kernel_size=3, stride=1, padding=1):
        proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(dim_in))
        return proj

    def forward(self, x, mask=None):
        x_out1 = self.C_C(x)

        x_out2 = self.H_W(x)

        x_out3 = self.C_H(x)

        x_out4 = self.C_W(x)

        x_out = (self.gamma1 * x_out1) + (self.gamma2 * x_out2) + (self.gamma3 * x_out3) + (self.gamma4 * x_out4)

        return x_out


""" =============================================================================================================== """


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer([dim, input_resolution[0], input_resolution[1]])

        self.attn = WindowAttention_ACAM(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            self.attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            self.attn_mask = self.attn_mask.masked_fill(self.attn_mask != 0, float(-100.0)).masked_fill(
                self.attn_mask == 0, float(0.0))
        else:
            self.attn_mask = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = GhostModule(inp=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut1 = x
        x = x.view(B, H, W, C)
        x = self.norm1(x)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, C, H, W)
        x = shortcut1 + self.drop_path(x)
        shortcut2 = x
        x = self.norm2(x)
        x = shortcut2 + self.drop_path(self.mlp(x))
        return x


###-------------------------------主模块-----------------------------------------
class ADUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ADUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 36))
        self.down1 = (Down(36, 72))
        self.down2 = (Down(72, 144))
        self.down3 = (Down(144, 288))
        factor = 2 if bilinear else 1
        self.up1 = (Up(288, 144 // factor, bilinear))
        self.up2 = (Up(144, 72 // factor, bilinear))
        self.up3 = (Up(72, 36 // factor, bilinear))
        self.outc = (OutConv(36, n_classes))
        self.connect1 = (SwinTransformerBlock(36, (48, 48), 3, 6))
        self.connect2 = (SwinTransformerBlock(72, (24, 24), 2, 6))
        self.connect3 = (SwinTransformerBlock(144, (12, 12), 1, 6))
        self.connect4 = nn.Linear(288 * 6 * 6, 288 * 6 * 6)

    def forward(self, x):
        N, C, H, W = x.size()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4_ = x4.view(x4.size(0), -1)
        x4_ = self.connect4(x4_)
        x4_ = x4_.view(N, 288, 6, 6)
        c_x3 = self.connect3(x3)
        x3_ = self.up1(x4_, c_x3)
        c_x2 = self.connect2(x2)
        x2_ = self.up2(x3_, c_x2)
        c_x1 = self.connect1(x1)
        x1_ = self.up3(x2_, c_x1)
        logits = self.outc(x1_)
        # logits = F.relu(logits)
        # logits = F.softmax(logits, dim=1)
        return logits


# if __name__ == '__main__':
#     print('*--' * 5)
#     rgb = torch.randn(1, 1, 48, 48)
#     net = ADUNet(1, 2)
#     out = net(rgb)
#     print(out.shape)
