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

# --------------------------------------------

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
    
    # --------------------------------------------

# class EPEDLayer(nn.Module):
#     def __init__(self, in_channels, initial_diffusion_coefficient=0.1, max_iterations=5, min_iterations=2,
#                  adaptive_rate=0.1):
#         super(EPEDLayer, self).__init__()
#         self.in_channels = in_channels
#         self.initial_diffusion_coefficient = initial_diffusion_coefficient
#         self.max_iterations = max_iterations
#         self.min_iterations = min_iterations
#         self.adaptive_rate = adaptive_rate

#         # Laplacian operator for diffusion
#         self.laplacian = nn.Conv2d(
#             in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
#         )
#         laplacian_kernel = torch.tensor(
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
#         ).repeat(in_channels, 1, 1, 1)
#         self.laplacian.weight.data = laplacian_kernel
#         self.laplacian.weight.requires_grad = False

#         # Learnable diffusion coefficient
#         self.diffusion_coefficient = nn.Parameter(torch.tensor(initial_diffusion_coefficient), requires_grad=True)
#         self.iterations = self.min_iterations

#     def forward(self, x):
#         diffusion_coeff = self.adaptive_rate * self.diffusion_coefficient.clamp(min=0, max=1)
#         num_iterations = max(self.min_iterations, int(self.iterations))

#         for _ in range(num_iterations):
#             laplacian_x = self.laplacian(x)
#             x = x + diffusion_coeff * laplacian_x

#             # Dynamically adjust iterations based on the gradient norm
#             # Simple heuristic: increase iterations if the gradient norm is high
#             grad_norm = torch.norm(laplacian_x, p=2, dim=[2, 3], keepdim=True).mean()
#             if grad_norm > 0.05:
#                 self.iterations = min(self.max_iterations, self.iterations + 1)
#             else:
#                 self.iterations = max(self.min_iterations, self.iterations - 1)

#         return x

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

# --------------------------------------------
# class EPEDLayer(nn.Module):
#     def __init__(self, in_channels, initial_diffusion_coefficient=0.1, max_iterations=5, min_iterations=2, adaptive_rate=0.1):
#         super(EPEDLayer, self).__init__()
#         self.in_channels = in_channels
#         self.initial_diffusion_coefficient = initial_diffusion_coefficient
#         self.max_iterations = max_iterations
#         self.min_iterations = min_iterations
#         self.adaptive_rate = adaptive_rate
        
#         # LAPLACIAN OPERATOR FOR DIFFUSION
#         self.laplacian = nn.Conv2d(
#             in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
#         )
#         laplacian_kernel = torch.tensor(
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
#         ).repeat(in_channels, 1, 1, 1)
#         self.laplacian.weight.data = laplacian_kernel
#         self.laplacian.weight.requires_grad = False
        
#         # LEARNABLE DIFFUSION COEFFICIENT
#         self.diffusion_coefficient = nn.Parameter(torch.tensor(initial_diffusion_coefficient), requires_grad=True)
#         self.iterations = self.min_iterations
        
#         # ATTENTION MECHANISM
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         residual = x
#         diffusion_coeff = self.adaptive_rate * self.diffusion_coefficient.clamp(min=0, max=1)
#         num_iterations = max(self.min_iterations, int(self.iterations))
        
#         for _ in range(num_iterations):
#             laplacian_x = self.laplacian(x)
#             x = x + diffusion_coeff * laplacian_x
            
#             # DYNAMICALLY ADJUST ITERATIONS BASED ON THE GRADIENT NORM
#             grad_norm = torch.norm(laplacian_x, p=2, dim=[2, 3], keepdim=True).mean()
#             if grad_norm > 0.05:
#                 self.iterations = min(self.max_iterations, self.iterations + 1)
#             else:
#                 self.iterations = max(self.min_iterations, self.iterations - 1)
        
#         # APPLY ATTENTION MECHANISM
#         att = self.attention(x)
#         x = x * att + residual
        
#         return x

# --------------------------------------------

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction_ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction_ratio, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class EPEDLayer(nn.Module):
#     def __init__(self, in_channels, initial_diffusion_coefficient=0.1, max_iterations=5, min_iterations=2, adaptive_rate=0.1):
#         super(EPEDLayer, self).__init__()
#         self.in_channels = in_channels
#         self.initial_diffusion_coefficient = initial_diffusion_coefficient
#         self.max_iterations = max_iterations
#         self.min_iterations = min_iterations
#         self.adaptive_rate = adaptive_rate

#         # Laplacian operator for diffusion
#         self.laplacian = nn.Conv2d(
#             in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
#         )
#         laplacian_kernel = torch.tensor(
#             [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
#         ).repeat(in_channels, 1, 1, 1)
#         self.laplacian.weight.data = laplacian_kernel
#         self.laplacian.weight.requires_grad = False

#         # Learnable diffusion coefficient
#         self.diffusion_coefficient = nn.Parameter(torch.tensor(initial_diffusion_coefficient), requires_grad=True)
#         self.iterations = self.min_iterations

#         # SE Block for attention
#         self.se = SEBlock(in_channels)

#     def forward(self, x):
#         identity = x.clone()  # Save the identity for residual connection
#         diffusion_coeff = self.adaptive_rate * self.diffusion_coefficient.clamp(min=0, max=1)
#         num_iterations = max(self.min_iterations, int(self.iterations))

#         for _ in range(num_iterations):
#             laplacian_x = self.laplacian(x)
#             x = x + diffusion_coeff * laplacian_x

#             # Dynamically adjust iterations based on the gradient norm
#             # Simple heuristic: increase iterations if the gradient norm is high
#             grad_norm = torch.norm(laplacian_x, p=2, dim=[2, 3], keepdim=True).mean()
#             if grad_norm > 0.05:
#                 self.iterations = min(self.max_iterations, self.iterations + 1)
#             else:
#                 self.iterations = max(self.min_iterations, self.iterations - 1)

#         x = self.se(x)  # Apply SE block for feature reweighting
#         x = x + identity  # Residual connection
#         return x
    
# ---------------------------------------------


class LaplacianGradientAttention(nn.Module):
    def __init__(self, channels, num_heads, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32, dynamic_ratio=4,
                 diffusion_coefficient=0.1, convection_coefficient=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.dynamic_ratio = dynamic_ratio
        self.scale = 1 / (self.head_dim ** 0.5)
        self.qkv_conv = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dynamic_weights = nn.Parameter(torch.Tensor(num_heads, self.head_dim // dynamic_ratio, self.head_dim))
        nn.init.xavier_uniform_(self.dynamic_weights)
        self.d = max(L, channels // reduction)
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=group)),
                ('bn', nn.BatchNorm2d(channels)),
                ('relu', nn.ReLU())
            ])) for k in kernels
        ])
        self.fc = nn.Linear(channels, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channels) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)
        
        # 新增：用于物理约束的分支（物理残差计算）
        self.conv_laplacian = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]]]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.conv_laplacian.weight.data = laplacian_kernel
        self.conv_laplacian.weight.requires_grad = False
        # 可选：其他物理算子，比如对流项计算
        self.conv_gradient_x = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels, bias=False)
        self.conv_gradient_y = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=False)
        # 存储物理参数
        self.diffusion_coefficient = diffusion_coefficient
        self.convection_coefficient = convection_coefficient

    def forward(self, x):
        bsz, ch, ht, wd = x.shape
        
        # 1. 多头自注意力计算
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, self.channels, dim=1)
        q = q.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        k = k.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        v = v.view(bsz, self.num_heads, self.head_dim, ht * wd).permute(0, 1, 3, 2)
        scores = torch.einsum('bnid,bnjd->bnij', q, k) * self.scale
        attention_probs = torch.softmax(scores, dim=-1)
        context = torch.einsum('bnij,bnjd->bnid', attention_probs, v)
        context = context.permute(0, 1, 3, 2).contiguous().view(bsz, -1, ht, wd)
        output = self.output_proj(context)
        
        # 2. 多尺度卷积分支
        conv_outs = [conv(output) for conv in self.convs]
        feats = torch.stack(conv_outs, dim=0)
        
        # 3. 计算全局特征用于动态权重
        U = torch.sum(feats, dim=0)
        S = U.mean([-2, -1])
        Z = self.fc(S)
        weights = [fc(Z).view(bsz, ch, 1, 1) for fc in self.fcs]
        attention_weights = self.softmax(torch.stack(weights, dim=0))
        fused_feats = torch.einsum('nbcwh,nbcwh->bcwh', attention_weights, feats)
        
        # 4. 物理信息分支：计算局部物理残差
        # 例如，利用局部拉普拉斯算子捕捉扩散特性，并计算对流项
        laplacian_response = self.conv_laplacian(x)
        laplacian_response = torch.sigmoid(laplacian_response)  # 限制数值范围
        grad_x = self.conv_gradient_x(x)
        grad_y = self.conv_gradient_y(x)
        convection_response = self.convection_coefficient * (x * grad_x + x * grad_y)
        # 物理残差项可以看作是 diffusive + convective 部分
        physics_response = self.diffusion_coefficient * laplacian_response + convection_response
        
        # 5. 融合原始注意力输出和物理响应
        # 这里可以设计为加权融合，也可以作为残差连接
        enhanced_feats = fused_feats + physics_response
        
        return enhanced_feats



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
# # 未使用任何模块      0.7219350337982178_metric_model
        # y1 = self.inc1(x)
        # y2 = self.down1(y1)
        # y3 = self.down2(y2)
        # y4 = self.down3(y3)
        # y5 = self.down4(y4)

        # features = self.feature_extractor(x)
        # x1 = features[0]
        # e1 = torch.cat([y1, x1], dim=1)
        # x2 = features[1]
        # e2 = torch.cat([y2, x2], dim=1)
        # x3 = features[2]
        # e3 = torch.cat([y3, x3], dim=1)
        # x4 = features[3]
        # e4 = torch.cat([y4, x4], dim=1)
        # x5 = features[4]
        # e5 = torch.cat([y5, x5], dim=1)

        # z4 = self.up1(e5, e4)
        # z3 = self.up2(z4, e3)
        # z2 = self.up3(z3, e2)
        # z1 = self.up4(z2, e1)
        # z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

        # logits = self.outc(z1)
        # return logits

# # 使用FreqHolo    0.7626082897186279_metric_model
#         y1 = self.inc1(x)
#         y2 = self.down1(y1)
#         y3 = self.down2(y2)
#         y4 = self.down3(y3)
#         y5 = self.down4(y4)

#         features = self.feature_extractor(x)
#         x1 = features[0]
#         x1 = self.freq_attn64(x1)
#         e1 = torch.cat([y1, x1], dim=1)
#         x2 = features[1]
#         x2 = self.freq_attn128(x2)
#         e2 = torch.cat([y2, x2], dim=1)
#         x3 = features[2]
#         x3 = self.freq_attn256(x3)
#         e3 = torch.cat([y3, x3], dim=1)
#         x4 = features[3]
#         x4 = self.freq_attn512(x4)
#         e4 = torch.cat([y4, x4], dim=1)
#         x5 = features[4]
#         e5 = torch.cat([y5, x5], dim=1)

#         z4 = self.up1(e5, e4)
#         z3 = self.up2(z4, e3)
#         z2 = self.up3(z3, e2)
#         z1 = self.up4(z2, e1)
#         z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

#         logits = self.outc(z1)
#         return logits

# # 使用EPED 0.734025239944458_metric_model
#         y1 = self.inc1(x)
#         y1 = self.diffusion_attn64(y1)
#         y2 = self.down1(y1)
#         y2 = self.diffusion_attn128(y2)
#         y3 = self.down2(y2)
#         y3 = self.diffusion_attn256(y3)
#         y4 = self.down3(y3)
#         y4 = self.diffusion_attn512(y4)
#         y5 = self.down4(y4)
#         y5 = self.diffusion_attn1024(y5)

#         features = self.feature_extractor(x)
#         x1 = features[0]
#         e1 = torch.cat([y1, x1], dim=1)
#         x2 = features[1]
#         e2 = torch.cat([y2, x2], dim=1)
#         x3 = features[2]
#         e3 = torch.cat([y3, x3], dim=1)
#         x4 = features[3]
#         e4 = torch.cat([y4, x4], dim=1)
#         x5 = features[4]
#         e5 = torch.cat([y5, x5], dim=1)

#         z4 = self.up1(e5, e4)
#         z3 = self.up2(z4, e3)
#         z2 = self.up3(z3, e2)
#         z1 = self.up4(z2, e1)
#         z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

#         logits = self.outc(z1)
#         return logits

# # 使用LaGra   0.7658878564834595
#         y1 = self.inc1(x)
#         y2 = self.down1(y1)
#         y3 = self.down2(y2)
#         y4 = self.down3(y3)
#         y5 = self.down4(y4)

#         features = self.feature_extractor(x)
#         x1 = features[0]
#         e1 = torch.cat([y1, x1], dim=1)
#         x2 = features[1]
#         e2 = torch.cat([y2, x2], dim=1)
#         x3 = features[2]
#         e3 = torch.cat([y3, x3], dim=1)
#         x4 = features[3]
#         e4 = torch.cat([y4, x4], dim=1)
#         x5 = features[4]
#         x5 = self.combined_attention(x5)
#         e5 = torch.cat([y5, x5], dim=1)

#         z4 = self.up1(e5, e4)
#         z3 = self.up2(z4, e3)
#         z2 = self.up3(z3, e2)
#         z1 = self.up4(z2, e1)
#         z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

#         logits = self.outc(z1)
#         return logits
    
# # 使用EPED+FreqHolo 0.7690739636421204_metric_model
#         y1 = self.inc1(x)
#         y1 = self.diffusion_attn64(y1)
#         y2 = self.down1(y1)
#         y2 = self.diffusion_attn128(y2)
#         y3 = self.down2(y2)
#         y3 = self.diffusion_attn256(y3)
#         y4 = self.down3(y3)
#         y4 = self.diffusion_attn512(y4)
#         y5 = self.down4(y4)
#         y5 = self.diffusion_attn1024(y5)

#         features = self.feature_extractor(x)
#         x1 = features[0]
#         x1 = self.freq_attn64(x1)
#         e1 = torch.cat([y1, x1], dim=1)
#         x2 = features[1]
#         x2 = self.freq_attn128(x2)
#         e2 = torch.cat([y2, x2], dim=1)
#         x3 = features[2]
#         x3 = self.freq_attn256(x3)
#         e3 = torch.cat([y3, x3], dim=1)
#         x4 = features[3]
#         x4 = self.freq_attn512(x4)
#         e4 = torch.cat([y4, x4], dim=1)
#         x5 = features[4]
#         e5 = torch.cat([y5, x5], dim=1)

#         z4 = self.up1(e5, e4)
#         z3 = self.up2(z4, e3)
#         z2 = self.up3(z3, e2) 
#         z1 = self.up4(z2, e1)
#         z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

#         logits = self.outc(z1)
#         return logits

# # 使用 FreqHolo+LaGr 0.7667967081069946_metric_model
#         y1 = self.inc1(x)
#         y2 = self.down1(y1)
#         y3 = self.down2(y2)
#         y4 = self.down3(y3)
#         y5 = self.down4(y4)

#         features = self.feature_extractor(x)
#         x1 = features[0]
#         x1 = self.freq_attn64(x1)
#         e1 = torch.cat([y1, x1], dim=1)
#         x2 = features[1]
#         x2 = self.freq_attn128(x2)
#         e2 = torch.cat([y2, x2], dim=1)
#         x3 = features[2]
#         x3 = self.freq_attn256(x3)
#         e3 = torch.cat([y3, x3], dim=1)
#         x4 = features[3]
#         x4 = self.freq_attn512(x4)
#         e4 = torch.cat([y4, x4], dim=1)
#         x5 = features[4]
#         x5 = self.combined_attention(x5)
#         e5 = torch.cat([y5, x5], dim=1)

#         z4 = self.up1(e5, e4)
#         z3 = self.up2(z4, e3)
#         z2 = self.up3(z3, e2)
#         z1 = self.up4(z2, e1)
#         z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

#         logits = self.outc(z1)
#         return logits

# # 使用LaplacianGradientAttention+EPED Layer 0.7718335390090942_metric_model.pth
#         y1 = self.inc1(x)
#         y1 = self.diffusion_attn64(y1)
#         y2 = self.down1(y1)
#         y2 = self.diffusion_attn128(y2)
#         y3 = self.down2(y2)
#         y3 = self.diffusion_attn256(y3)
#         y4 = self.down3(y3)
#         y4 = self.diffusion_attn512(y4)
#         y5 = self.down4(y4)
#         y5 = self.diffusion_attn1024(y5)

#         features = self.feature_extractor(x)
#         x1 = features[0]
#         e1 = torch.cat([y1, x1], dim=1)
#         x2 = features[1]
#         e2 = torch.cat([y2, x2], dim=1)
#         x3 = features[2]
#         e3 = torch.cat([y3, x3], dim=1)
#         x4 = features[3]
#         e4 = torch.cat([y4, x4], dim=1)
#         x5 = features[4]
#         x5 = self.combined_attention(x5)
#         e5 = torch.cat([y5, x5], dim=1)

#         z4 = self.up1(e5, e4)
#         z3 = self.up2(z4, e3)
#         z2 = self.up3(z3, e2)
#         z1 = self.up4(z2, e1)
#         z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

#         logits = self.outc(z1)
#         return logits

# 使用HoloschrodAtt+LaplacianGradientAttention+EPED Layer 0.7847081422805786_metric_model
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
        # x1 = self.diffusion_attn64(x1)
        e1 = torch.cat([y1, x1], dim=1)
        x2 = features[1]
        x2 = self.freq_attn128(x2)
        # x2 = self.diffusion_attn128(x2)
        e2 = torch.cat([y2, x2], dim=1)
        x3 = features[2]
        x3 = self.freq_attn256(x3)
        # x3 = self.diffusion_attn256(x3)
        e3 = torch.cat([y3, x3], dim=1)
        x4 = features[3]
        x4 = self.freq_attn512(x4)
        # x4 = self.diffusion_attn512(x4)
        e4 = torch.cat([y4, x4], dim=1)
        x5 = features[4]
        x5 = self.combined_attention(x5)
        # x5 = self.diffusion_attn1024(x5)
        e5 = torch.cat([y5, x5], dim=1)

        z4 = self.up1(e5, e4)
        z3 = self.up2(z4, e3)
        z2 = self.up3(z3, e2)
        z1 = self.up4(z2, e1)
        z1 = F.interpolate(z1, size=(224, 224), mode='bilinear', align_corners=True)

        logits = self.outc(z1)
        return logits
    
