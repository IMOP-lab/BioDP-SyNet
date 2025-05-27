import torch
import torch.fft
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from .encoder import FeatureExtractor


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# # --------------------------------------------

class EPEDLayer(nn.Module):
    def __init__(self, in_channels, diffusion_coefficient=0, num_iterations=3):
        super(EPEDLayer, self).__init__()
        self.in_channels = in_channels
        self.diffusion_coefficient = diffusion_coefficient
        self.num_iterations = num_iterations

        # Laplacian operator for diffusion
        self.laplacian = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).repeat(in_channels, 1, 1, 1)
        self.laplacian.weight.data = laplacian_kernel
        self.laplacian.weight.requires_grad = False

    def forward(self, x):
        for _ in range(self.num_iterations):
            laplacian_x = self.laplacian(x)
            x = x + self.diffusion_coefficient * laplacian_x
        return x
    

# ----------------------------------------------

# class EPEDLayer(nn.Module):
#     def __init__(self, in_channels, diffusion_coefficient=0.1, num_iterations=3):
#         super(EPEDLayer, self).__init__()
#         self.in_channels = in_channels
#         self.num_iterations = num_iterations

#         # Laplacian operator for diffusion
#         self.laplacian = nn.Conv2d(
#             in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
#         )
#         laplacian_kernel = torch.tensor(
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
#         ).repeat(in_channels, 1, 1, 1)
#         self.laplacian.weight.data = laplacian_kernel
#         self.laplacian.weight.requires_grad = False

#         # Adaptive diffusion coefficient
#         self.adaptive_diffusion = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         residual = x.clone() # Residual Connection
#         for _ in range(self.num_iterations):
#             laplacian_x = self.laplacian(x)
#             adaptive_coeff = self.adaptive_diffusion(x)
#             x = x + adaptive_coeff * laplacian_x
#         x = x + residual # Adding Residual Connection
#         return x




class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        
        self.relu6 = nn.ReLU6(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def h_sigmoid(self, x):
        return self.relu6(x + 3) / 6

    def h_swish(self, x):
        return x * self.h_sigmoid(x)

    def forward(self, x):
        identity = x   
        n, c, h, w = x.size()
        
        # 坐标注意力的通道分割与融合
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        
        y = self.h_swish(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

    
class HoloschrodAtt(nn.Module):
    def __init__(self, in_channels, reduction=32, num_iterations=3, diffusion_coeff=0.1):
        super(HoloschrodAtt, self).__init__()
        self.in_channels = in_channels
        self.num_iterations = num_iterations
        self.diffusion_coeff = diffusion_coeff

        # 坐标注意力用于分别处理实部和虚部
        self.coord_att_real = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.coord_att_imag = CoordAtt(in_channels, in_channels, reduction=reduction)

        # 频域操作
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Schrödinger 方程中的偏导数操作
        self.laplacian = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).repeat(in_channels, 1, 1, 1)
        self.laplacian.weight.data = laplacian_kernel
        self.laplacian.weight.requires_grad = False

    def forward(self, x):
        # Step 1: 频域转换
        fft_x = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        fft_x_real = fft_x.real
        fft_x_imag = fft_x.imag

        # Step 2: 使用坐标注意力增强实部和虚部
        fft_x_real = self.coord_att_real(fft_x_real)
        fft_x_imag = self.coord_att_imag(fft_x_imag)

        # Step 3: Schrödinger 方程约束 (波动方程)
        for _ in range(self.num_iterations):
            laplacian_real = self.laplacian(fft_x_real)
            laplacian_imag = self.laplacian(fft_x_imag)

            # 更新实部和虚部，确保满足 Schrödinger 方程
            fft_x_real = fft_x_real - self.diffusion_coeff * laplacian_imag
            fft_x_imag = fft_x_imag + self.diffusion_coeff * laplacian_real

        # Step 4: 合成结果并返回到时域
        fft_x = torch.complex(fft_x_real, fft_x_imag)
        ifft_x = torch.fft.irfftn(fft_x, s=x.shape[-2:], dim=(-2, -1), norm='ortho')
        return ifft_x
    
class LaplacianGradientAttention(nn.Module):
    """
    拉普拉斯梯度注意力模块。
    结合了多头自注意力、多尺度卷积特征融合以及基于物理（拉普拉斯、梯度）的局部信息提取。
    """
    def __init__(self, channels, num_heads, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32, dynamic_ratio=4,
                 diffusion_coefficient=0.1, convection_coefficient=0.1):
        """
        初始化函数。
        Args:
            channels (int): 输入特征图的通道数。
            num_heads (int): 注意力头的数量。
            kernels (list): 用于多尺度卷积的核大小列表。
            reduction (int): 用于计算动态权重的中间线性层的缩减比例。
            group (int): 卷积中的分组数。
            L (int): 用于计算动态权重的线性层输出维度下限。
            dynamic_ratio (int): 用于动态权重计算的参数。
            diffusion_coefficient (float): 模拟扩散的物理系数。
            convection_coefficient (float): 模拟对流的物理系数。
        """
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.dynamic_ratio = dynamic_ratio
        self.scale = 1 / (self.head_dim ** 0.5) # 注意力分数的缩放因子
        self.qkv_conv = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False)                # QKV 计算层
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)                 # 输出投影层
        self.d = max(L, channels // reduction) # 用于动态权重计算的中间维度  # 多尺度卷积分支相关的参数
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=group)),
                ('bn', nn.BatchNorm2d(channels)),
                ('relu', nn.ReLU())
            ])) for k in kernels
        ])

        self.fc = nn.Linear(channels, self.d)                                                       # 用于计算动态权重的全连接层
        self.fcs = nn.ModuleList([nn.Linear(self.d, channels) for _ in kernels])                    # 为每个尺度生成权重
        self.softmax = nn.Softmax(dim=0)                                                            # 在不同尺度上进行Softmax加权
        # 新增：用于物理约束的分支（物理残差计算）        
        self.conv_laplacian = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)  # 用于计算拉普拉斯算子（模拟扩散）的深度可分离卷积

        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]]]], dtype=torch.float32)                      # 定义并设置拉普拉斯核
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1) # 复制到所有通道
        self.conv_laplacian.weight.data = laplacian_kernel
        self.conv_laplacian.weight.requires_grad = False # 固定拉普拉斯核不参与训练
        # 可选：其他物理算子，比如对流项计算（基于梯度）
        self.conv_gradient_x = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels, bias=False)           # 用于计算 x 方向梯度的卷积
        self.conv_gradient_y = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=False)           # 用于计算 y 方向梯度的卷积
        self.diffusion_coefficient = diffusion_coefficient              # 存储物理参数
        self.convection_coefficient = convection_coefficient

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入特征图，形状为 (bsz, ch, ht, wd)。
        Returns:
            torch.Tensor: 融合了注意力、多尺度卷积和物理信息的增强特征图。
        """
        bsz, ch, ht, wd = x.shape
        # 1. 多头自注意力计算
        qkv = self.qkv_conv(x) # 计算 Q, K, V
        q, k, v = torch.split(qkv, self.channels, dim=1) # 按通道分割 Q, K, V
        
        q = q.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)     # 重塑并转置 Q, K, V 以适应注意力计算 (bsz, num_heads, num_pixels, head_dim)
        k = k.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        v = v.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        scores = torch.einsum('bnid,bnjd->bnij', q, k) * self.scale                     # 计算注意力分数 (Q @ K^T)
        attention_probs = torch.softmax(scores, dim=-1)                                 # 计算注意力权重
        context = torch.einsum('bnij,bnjd->bnid', attention_probs, v)                   # 计算上下文向量 (Attention_probs @ V)
        context = context.permute(0, 1, 3, 2).contiguous().view(bsz, -1, ht, wd)        # 重塑上下文向量回原始形状 (bsz, channels, ht, wd)
        output = self.output_proj(context)                                              # 输出投影
        # 2. 多尺度卷积分支
        conv_outs = [conv(output) for conv in self.convs]                               # 对注意力输出应用不同尺度的卷积
        feats = torch.stack(conv_outs, dim=0)                                           # 将不同尺度的输出堆叠起来 (num_kernels, bsz, ch, ht, wd)
        # 3. 计算全局特征用于动态权重（SE-like 机制）
        U = torch.sum(feats, dim=0)                                                     # 在尺度维度求和（或可以使用其他聚合方式）
        S = U.mean([-2, -1])                                                            # 全局平均池化 (bsz, ch)
        Z = self.fc(S)                                                                  # 第一个全连接层 (bsz, d)
        # 计算每个尺度的动态权重
        weights = [fc(Z).view(bsz, ch, 1, 1) for fc in self.fcs]                        # 每个尺度一个权重向量 (bsz, ch, 1, 1)
        attention_weights = self.softmax(torch.stack(weights, dim=0))                   # 在尺度维度应用Softmax (num_kernels, bsz, ch, 1, 1)
        # 加权融合多尺度特征
        fused_feats = torch.einsum('nbcwh,nbcwh->bcwh', attention_weights, feats)       # (bsz, ch, ht, wd)
        # 4. 物理信息分支：计算局部物理残差
        laplacian_response = self.conv_laplacian(x) # 计算拉普拉斯响应
        laplacian_response = torch.sigmoid(laplacian_response) # 限制数值范围，防止过大或过小
        # 计算 x 和 y 方向的梯度
        grad_x = self.conv_gradient_x(x)
        grad_y = self.conv_gradient_y(x)
        # 计算对流项 (u * grad(u) 的简化形式，这里用 x * grad(x) + x * grad(y) 作为示例)
        convection_response = self.convection_coefficient * (x * grad_x + x * grad_y)
        # 物理残差项可以看作是 diffusive + convective 部分的加权和
        physics_response = self.diffusion_coefficient * laplacian_response + convection_response
        # 5. 融合原始注意力输出和物理响应
        # 这里设计为残差连接方式融合
        enhanced_feats = fused_feats + physics_response
        return enhanced_feats
    

class DP_CoNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DP_CoNet, self).__init__()
    
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(32, 64)
        self.inc1 = DoubleConv(3, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.diffusion_attn64 = EPEDLayer(64)
        self.diffusion_attn128 = EPEDLayer(128)
        self.diffusion_attn256 = EPEDLayer(256)
        self.diffusion_attn512 = EPEDLayer(512)
        self.diffusion_attn1024 = EPEDLayer(1024)
    
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)

        self.outc = OutConv(128, n_classes)

        self.combined_attention = LaplacianGradientAttention(
            channels=1024,
            num_heads=8,
            kernels=[1, 3, 5, 7],
            reduction=16,
            group=1,
            L=32,
            dynamic_ratio=4
        )

        self.freq_attn64 = HoloschrodAtt(64)
        self.freq_attn128 = HoloschrodAtt(128)
        self.freq_attn256 = HoloschrodAtt(256)
        self.freq_attn512 = HoloschrodAtt(512)
        self.freq_attn1024 = HoloschrodAtt(1024)


        self.feature_extractor = FeatureExtractor(
            input_channels=3,
            n_stages=5,
            features_per_stage=[64, 128, 256, 512, 1024],
            conv_op=nn.Conv2d,
            kernel_sizes=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            n_blocks_per_stage=[2, 2, 2, 2, 2],
            norm_op=nn.BatchNorm2d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.ReLU,
            nonlin_kwargs={'inplace': True}
        )


    def forward(self, x):
        y1 = self.inc1(x)
        y1 = self.diffusion_attn64(y1)
        y2 = self.down1(y1)
        y2 = self.diffusion_attn128(y2)
        y3 = self.down2(y2)
        y3 = self.diffusion_attn256(y3)
        y4 = self.down3(y3)
        y4 = self.diffusion_attn512(y4)
        y5 = self.down4(y4)
        y5 = self.diffusion_attn1024(y5)

        features = self.feature_extractor(x)
        x1 = features[0]
        x1 = self.freq_attn64(x1)
        e1 = torch.cat([y1, x1], dim=1)
        x2 = features[1]
        x2 = self.freq_attn128(x2)
        e2 = torch.cat([y2, x2], dim=1)
        x3 = features[2]
        x3 = self.freq_attn256(x3)
        e3 = torch.cat([y3, x3], dim=1)
        x4 = features[3]
        x4 = self.freq_attn512(x4)
        e4 = torch.cat([y4, x4], dim=1)
        x5 = features[4]
        x5 = self.combined_attention(x5)
        e5 = torch.cat([y5, x5], dim=1)

        z4 = self.up1(e5, e4)
        z3 = self.up2(z4, e3)
        z2 = self.up3(z3, e2)
        z1 = self.up4(z2, e1)
        z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

        logits = self.outc(z1)
        return logits
    
