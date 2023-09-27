from torch import nn
import numpy as np
from abc import abstractmethod

from .__init__ import time_embedding
from .__init__ import Downsample
from .__init__ import Upsample


# use GN for norm layer
def group_norm(channels):
    return nn.GroupNorm(32, channels)


# 包含 time_embedding 的 block
class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        函数不能为空，但可以添加注释
        """


class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x, emb):
        for layer in self:
            # 判断该 layer 中是否包含 time_embedding
            if isinstance(layer, TimeBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ResBlock 继承 TimeBlock
# 所有的 ResBlock 中均包含 time_embedding，其他 layer 不包含 time_embedding
class ResBlock(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class NoisePred(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_channels,
                 num_res_blocks,
                 dropout,
                 time_embed_dim_mult,
                 down_sample_mult,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.down_sample_mult = down_sample_mult

        # time embedding
        time_embed_dim = model_channels * time_embed_dim_mult
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 下采样和上采样的通道数
        down_channels = [model_channels * i for i in down_sample_mult]
        up_channels = down_channels[::-1]

        # 每个块中 ResBlock 的数量
        downBlock_chanNum = [num_res_blocks + 1] * (len(down_sample_mult) - 1)
        downBlock_chanNum.append(num_res_blocks)
        upBlock_chanNum = downBlock_chanNum[::-1]
        self.downBlock_chanNum_cumsum = np.cumsum(downBlock_chanNum)
        self.upBlock_chanNum_cumsum = np.cumsum(upBlock_chanNum)[:-1]

        # 初始卷积层
        self.inBlock = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

        # 下采样
        # 共四个才采样块，每个下采样块包括两个包含 time_embed 的 ResBlock 和一个不包含 time_embed 的 DownSample 块
        self.downBlock = nn.ModuleList()
        down_init_channel = model_channels
        for level, channel in enumerate(down_channels):
            for _ in range(num_res_blocks):
                layer1 = ResBlock(in_channels=down_init_channel,
                                  out_channels=channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)
                down_init_channel = channel
                self.downBlock.append(TimeSequential(layer1))
            # 最后一步不做下采样
            if level != len(down_sample_mult) - 1:
                down_layer = Downsample(channels=channel)
                self.downBlock.append(TimeSequential(down_layer))

        # middle block
        self.middleBlock = nn.ModuleList()
        for _ in range(num_res_blocks):
            layer2 = ResBlock(in_channels=down_channels[-1],
                              out_channels=down_channels[-1],
                              time_channels=time_embed_dim,
                              dropout=dropout)
            self.middleBlock.append(TimeSequential(layer2))

        # 上采样
        # 共四个上采样块，每个上采样块包括两个包含 time_embed 的 ResBlock 和一个不包含 time_embed 的 DownSample 块
        self.upBlock = nn.ModuleList()
        up_init_channel = down_channels[-1]
        for level, channel in enumerate(up_channels):
            if level == len(up_channels) - 1:
                out_channel = model_channels
            else:
                out_channel = channel // 2
            for _ in range(num_res_blocks):
                layer3 = ResBlock(in_channels=up_init_channel,
                                  out_channels=out_channel,
                                  time_channels=time_embed_dim,
                                  dropout=dropout)
                up_init_channel = out_channel
                self.upBlock.append(TimeSequential(layer3))
            if level > 0:
                up_layer = Upsample(channels=out_channel)
                self.upBlock.append(TimeSequential(up_layer))

        # out block
        self.outBlock = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        embedding = time_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(embedding)

        # 用于存放每个下采样步骤的输出
        res = []

        # in stage
        x = self.inBlock(x)

        # down stage
        h = x
        num_down = 1
        for down_block in self.downBlock:
            h = down_block(h, time_emb)
            if num_down in self.downBlock_chanNum_cumsum:
                res.append(h)
            num_down += 1

        # middle stage
        for middle_block in self.middleBlock:
            h = middle_block(h, time_emb)
        h = h + res.pop()
        assert len(res) == len(self.upBlock_chanNum_cumsum)

        # up stage
        num_up = 1
        for up_block in self.upBlock:
            # 对于非2的幂次方的img_size，残差连接时会存在下采样和上采样尺寸不一致的现象
            # 以 res.pop()为标准，对 h进行裁剪
            if num_up in self.upBlock_chanNum_cumsum:  # [2,5,8]
                h = up_block(h, time_emb)
                h_crop = h[:, :, :res[-1].shape[2], :res[-1].shape[3]]
                h = h_crop + res.pop()
            else:
                h = up_block(h, time_emb)
            num_up += 1
        assert len(res) == 0

        # out stage
        out = self.outBlock(h)

        return out