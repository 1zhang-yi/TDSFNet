import torch.nn as nn
import torch

class Tuckergeneration(nn.Module):
    def __init__(self, config):
        super(Tuckergeneration, self).__init__()
        self.tensor = config.extracted_feature
        self.rank = config.rank
        self.core = config.core
        conv1_1, conv1_2, conv1_3 = self.ConvGeneration(self.rank, config.C_scale)

        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.tensor[0], out_channels=self.tensor[0]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.tensor[0]//2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.tensor[0]//2, out_channels=self.core[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.core[0]),
            nn.ReLU(True)
        )

        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=self.tensor[0], out_channels=self.tensor[0] // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.tensor[0] // 2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.tensor[0] // 2, out_channels=config.C_scale, kernel_size=1, stride=1),
            nn.BatchNorm2d(config.C_scale),
            nn.ReLU(True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')


    def forward(self, x):
        C = self.pool(x)
        C = self.conv_c(C)
        H = self.pool(x.permute(0, 2, 1, 3).contiguous())
        W = self.pool(x.permute(0, 3, 2, 1).contiguous())
        C_dim, H_dim, W_dim = self.TRM(C, H, W)
        core = self.conv(x)

        return core, C_dim, H_dim, W_dim

    def ConvGeneration(self, rank, C):
        conv1 = []
        for _ in range(0, rank[0]):
            conv1.append(nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.Sigmoid(),
        ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, rank[1]):
            conv2.append(nn.Sequential(
            nn.Conv2d(self.tensor[1], self.tensor[1], kernel_size=1, bias=False),
            nn.Sigmoid(),
        ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, rank[2]):
            conv3.append(nn.Sequential(
            nn.Conv2d(self.tensor[2], self.tensor[2], kernel_size=1, bias=False),
            nn.Sigmoid(),
        ))
        conv3 = nn.ModuleList(conv3)

        return conv1, conv2, conv3

    def TRM(self, C_weight, H_weight, W_weight):
        for i in range(self.rank[0]):
            C = self.conv1_1[i](C_weight)
            if i == 0:
                C_dim = C
            else:
                C_dim = torch.cat([C_dim, C], dim=2)

        for i in range(self.rank[1]):
            H = self.conv1_2[i](H_weight).contiguous()
            if i == 0:
                H_dim = H
            else:
                H_dim = torch.cat([H_dim, H], dim=2)

        for i in range(self.rank[2]):
            W = self.conv1_3[i](W_weight).contiguous()
            if i == 0:
                W_dim = W
            else:
                W_dim = torch.cat([W_dim, W], dim=2)

        C_dim = torch.squeeze(C_dim, 3)
        H_dim = torch.squeeze(H_dim, 3)
        W_dim = torch.squeeze(W_dim, 3)

        return C_dim, H_dim, W_dim

    # def TRM(self, C_weight, H_weight, W_weight):
    #     lam = self.lam.split(1, 0)
    #     for i in range(self.rank[0]):
    #         C = self.conv1_1[i](C_weight)
    #         H = self.conv1_2[i](H_weight).contiguous()
    #         H_ = H.permute(0, 2, 1, 3)
    #         W = self.conv1_3[i](W_weight).contiguous()
    #         W_ = W.permute(0, 3, 2, 1)
    #         CHW = C * H_ * W_
    #         if i == 0:
    #             C_dim = C
    #             H_dim = H
    #             W_dim = W
    #             y = CHW * lam[i]
    #         else:
    #             C_dim = torch.cat([C_dim, C], dim=2)
    #             H_dim = torch.cat([H_dim, H], dim=2)
    #             W_dim = torch.cat([W_dim, W], dim=2)
    #             y += CHW * lam[i]
    #
    #     C_dim = torch.squeeze(C_dim, 3)
    #     H_dim = torch.squeeze(H_dim, 3)
    #     W_dim = torch.squeeze(W_dim, 3)
    #
    #     return y, C_dim, H_dim, W_dim