import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VisionEncoder(nn.Module):
    def __init__(
        self, 
        mid_channels=[128, 512], 
        out_channels=3072,
        downsample_time=[1, 2],
        downsample_joint=[1, 1],
        ):
        super(VisionEncoder, self).__init__()

        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.downsample1 = Downsample(mid_channels[0], mid_channels[0], downsample_time[0], downsample_joint[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.downsample2 = Downsample(mid_channels[1], mid_channels[1], downsample_time[1], downsample_joint[1])
        self.conv_out = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        for resnet in self.resnet1:
            x = resnet(x)
        x = self.downsample1(x)
        
        x = self.resnet2(x)
        for resnet in self.resnet3:
            x = resnet(x)
        x = self.downsample2(x)

        x = self.conv_out(x)

        return x
    

class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        mid_channels=[128, 512], 
        out_channels=3072,
        downsample_time=[1, 2],
        downsample_joint=[1, 1],
        num_attention_heads=8,
        attention_head_dim=64,
        dim=3072,
        ):
        super(Encoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.downsample1 = Downsample(mid_channels[0], mid_channels[0], downsample_time[0], downsample_joint[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.downsample2 = Downsample(mid_channels[1], mid_channels[1], downsample_time[1], downsample_joint[1])
        self.conv_out = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        for resnet in self.resnet1:
            x = resnet(x)
        x = self.downsample1(x)
        
        x = self.resnet2(x)
        for resnet in self.resnet3:
            x = resnet(x)
        x = self.downsample2(x)

        x = self.conv_out(x)

        return x


class VectorQuantizer(nn.Module):
    def __init__(self, nb_code, code_dim, is_train=True):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()
        self.reset_count = 0
        self.usage = torch.zeros((self.nb_code, 1))
        self.is_train = is_train
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())
    
    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out
    
    def init_codebook(self, x):
        if torch.all(self.codebook == 0):
            out = self._tile(x)
            self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        if self.is_train:
          self.init = True

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[torch.randperm(out.shape[0])[:self.nb_code]]

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        self.usage = self.usage.to(usage.device)
        if self.reset_count >= 20:
            self.reset_count = 0
            usage = (usage + self.usage >= 1.0).float()
        else:
            self.reset_count += 1
            self.usage = (usage + self.usage >= 1.0).float()
            usage = torch.ones_like(self.usage, device=x.device)
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-6)))
            
        return perplexity

    def preprocess(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x, return_vq=False):
        bs, c, f, j = x.shape

        # Preprocess
        x = self.preprocess(x)
        assert x.shape[-1] == self.code_dim

        # Init codebook if not inited
        if not self.init and self.is_train:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.is_train:
            perplexity = self.update_codebook(x, code_idx)
        
        # Passthrough
        commit_loss = F.mse_loss(x, x_d.detach())
        x_d = x + (x_d - x).detach()

        if return_vq:
            return x_d.view(bs, f*j, c).contiguous(), commit_loss

        # Postprocess
        x_d = x_d.view(bs, f, j, c).permute(0, 3, 1, 2).contiguous()

        if self.is_train:
            return x_d, commit_loss, perplexity
        else:
            return x_d, commit_loss, code_idx.view(bs, f, j)


class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels=3072, 
        mid_channels=[512, 128], 
        out_channels=3,
        upsample_rate=None,
        frame_upsample_rate=[2.0, 1.0],
        joint_upsample_rate=[1.0, 1.0],
        dim=128,
        attention_head_dim=64,
        num_attention_heads=8,
        ):
        super(Decoder, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, stride=1, padding=1)
        self.resnet1 = nn.ModuleList([ResBlock(mid_channels[0], mid_channels[0]) for _ in range(3)])
        self.upsample1 = Upsample(mid_channels[0], mid_channels[0], frame_upsample_rate=frame_upsample_rate[0], joint_upsample_rate=joint_upsample_rate[0])
        self.resnet2 = ResBlock(mid_channels[0], mid_channels[1])
        self.resnet3 = nn.ModuleList([ResBlock(mid_channels[1], mid_channels[1]) for _ in range(3)])
        self.upsample2 = Upsample(mid_channels[1], mid_channels[1], frame_upsample_rate=frame_upsample_rate[1], joint_upsample_rate=joint_upsample_rate[1])
        self.conv_out = nn.Conv2d(mid_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for resnet in self.resnet1:
            x = resnet(x)
        x = self.upsample1(x)

        x = self.resnet2(x)
        for resnet in self.resnet3:
            x = resnet(x)
        x = self.upsample2(x)

        x = self.conv_out(x)

        return x


class Upsample(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        upsample_rate=None, 
        frame_upsample_rate=None,
        joint_upsample_rate=None,
        ):
        super(Upsample, self).__init__()

        self.upsampler = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample_rate = upsample_rate
        self.frame_upsample_rate = frame_upsample_rate
        self.joint_upsample_rate = joint_upsample_rate
        self.upsample_rate = upsample_rate

    def forward(self, inputs):
        if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
            x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

            if self.upsample_rate is not None:
                x_first = F.interpolate(x_first, scale_factor=self.upsample_rate)
                x_rest = F.interpolate(x_rest, scale_factor=self.upsample_rate)
            else:
                x_rest = F.interpolate(x_rest, scale_factor=(self.frame_upsample_rate, self.joint_upsample_rate), mode="bilinear", align_corners=True)
            x_first = x_first[:, :, None, :]
            inputs = torch.cat([x_first, x_rest], dim=2)
        elif inputs.shape[2] > 1:
            if self.upsample_rate is not None:
                inputs = F.interpolate(inputs, scale_factor=self.upsample_rate)
            else:
                inputs = F.interpolate(inputs, scale_factor=(self.frame_upsample_rate, self.joint_upsample_rate), mode="bilinear", align_corners=True)
        else:
            inputs = inputs.squeeze(2)
            if self.upsample_rate is not None:
                inputs = F.interpolate(inputs, scale_factor=self.upsample_rate)
            else:
                inputs = F.interpolate(inputs[:, :, None, :], scale_factor=(self.frame_upsample_rate, self.joint_upsample_rate), mode="bilinear", align_corners=True)

        b, c, t, j = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3).reshape(b * t, c, j)
        inputs = self.upsampler(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3)

        return inputs


class Downsample(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        frame_downsample_rate, 
        joint_downsample_rate
        ):
        super(Downsample, self).__init__()

        self.frame_downsample_rate = frame_downsample_rate
        self.joint_downsample_rate = joint_downsample_rate
        self.joint_downsample = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=self.joint_downsample_rate, padding=1)

    def forward(self, x):
        if self.frame_downsample_rate > 1:
            batch_size, channels, frames, joints = x.shape
            x = x.permute(0, 3, 1, 2).reshape(batch_size * joints, channels, frames)
            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = F.avg_pool1d(x_rest, kernel_size=self.frame_downsample_rate, stride=self.frame_downsample_rate)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = x.reshape(batch_size, joints, channels, x.shape[-1]).permute(0, 2, 3, 1)
            else:
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                x = x.reshape(batch_size, joints, channels, x.shape[-1]).permute(0, 2, 3, 1)
        
        batch_size, channels, frames, joints = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * frames, channels, joints)
        x = self.joint_downsample(x)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)
        return x


class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 group_num=32,
                 max_channels=512):
        super(ResBlock, self).__init__()
        skip = max(1, max_channels // out_channels - 1)
        self.block = nn.Sequential(
            nn.GroupNorm(group_num, in_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=skip, dilation=skip),
            nn.GroupNorm(group_num, out_channels, eps=1e-06, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.conv_short = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        hidden_states = self.block(x)
        if hidden_states.shape != x.shape:
            x = self.conv_short(x)
        x = x + hidden_states
        return x
