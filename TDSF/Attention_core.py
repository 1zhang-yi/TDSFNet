import torch.nn as nn
import copy
import math
from TDSF.MLP import Mlp
import torch

class Attention_core(nn.Module):
    def __init__(self, config, channel):
        super(Attention_core, self).__init__()
        self.channel_num = channel
        self.core_shape = config.core
        expand_ratio = config.expand_ratio
        self.num_attention_heads = config.num_heads
        self.out_channel = 1
        self.conv_c = nn.Sequential(
            nn.Conv2d(config.core[1], self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(config.core[2], self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(config.core[0], self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.attn_norm1 = nn.LayerNorm(config.core[0], eps=1e-6)
        self.attn_norm2 = nn.LayerNorm(config.core[1], eps=1e-6)
        self.attn_norm3 = nn.LayerNorm(config.core[2], eps=1e-6)

        self.norm1 = nn.LayerNorm(config.core[0], eps=1e-6)
        self.norm2 = nn.LayerNorm(config.core[1], eps=1e-6)
        self.norm3 = nn.LayerNorm(config.core[2], eps=1e-6)

        self.linear1 = nn.Linear(self.channel_num[0], self.core_shape[0])
        self.linear2 = nn.Linear(self.core_shape[0], self.channel_num[0])

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.key1 = nn.ModuleList()
        self.key2 = nn.ModuleList()
        self.key3 = nn.ModuleList()
        self.value1 = nn.ModuleList()
        self.value2 = nn.ModuleList()
        self.value3 = nn.ModuleList()
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)

        for _ in range(self.num_attention_heads):
            query1 = nn.Linear(self.core_shape[0], self.core_shape[0], bias=False)
            query2 = nn.Linear(self.core_shape[1], self.core_shape[1], bias=False)
            query3 = nn.Linear(self.core_shape[2], self.core_shape[2], bias=False)

            key1 = nn.Linear(self.core_shape[0], self.core_shape[0], bias=False)
            key2 = nn.Linear(self.core_shape[1], self.core_shape[1], bias=False)
            key3 = nn.Linear(self.core_shape[2], self.core_shape[2], bias=False)

            value1 = nn.Linear(self.core_shape[0], self.core_shape[0], bias=False)
            value2 = nn.Linear(self.core_shape[1], self.core_shape[1], bias=False)
            value3 = nn.Linear(self.core_shape[2], self.core_shape[2], bias=False)

            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))

            self.key1.append(copy.deepcopy(key1))
            self.key2.append(copy.deepcopy(key2))
            self.key3.append(copy.deepcopy(key3))

            self.value1.append(copy.deepcopy(value1))
            self.value2.append(copy.deepcopy(value2))
            self.value3.append(copy.deepcopy(value3))

        self.softmax = nn.Softmax(dim=3)
        self.out1 = nn.Linear(self.core_shape[0], self.core_shape[0], bias=False)
        self.out2 = nn.Linear(self.core_shape[1], self.core_shape[1], bias=False)
        self.out3 = nn.Linear(self.core_shape[2], self.core_shape[2], bias=False)

        self.attn_dropout = nn.Dropout(0.2)
        self.proj_dropout = nn.Dropout(0.2)

        self.ffn_norm1 = nn.LayerNorm(self.core_shape[0], eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(self.core_shape[1], eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(self.core_shape[1], eps=1e-6)

        self.ffn1 = Mlp(self.core_shape[0], self.core_shape[0] * expand_ratio)
        self.ffn2 = Mlp(self.core_shape[1], self.core_shape[1] * expand_ratio)
        self.ffn3 = Mlp(self.core_shape[2], self.core_shape[2] * expand_ratio)

    def forward(self, c, h, w, core):
        core_c = core.permute(0, 2, 3, 1).contiguous()
        core_h = core.permute(0, 3, 1, 2).contiguous()
        core_w = core

        core_c = self.conv_c(core_c).mean(dim=1)
        core_h = self.conv_h(core_h).mean(dim=1)
        core_w = self.conv_w(core_w).mean(dim=1)

        core_c = self.attn_norm1(core_c)
        core_h = self.attn_norm2(core_h)
        core_w = self.attn_norm3(core_w)

        c_k = c_v = core_c #[B, 4, 256]
        h_k = h_v = core_h #[B, 256, 4]
        w_k = w_v = core_w #[B, 4, 4]


        c = self.linear1(c) #[B, 256, 256]
        c_q = self.norm1(c) #[B, 256, 256]
        h_q = self.norm2(h) #[B, 256, 4]
        w_q = self.norm3(w) #[B, 256, 4]

        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []

        multi_head_K1_list = []
        multi_head_K2_list = []
        multi_head_K3_list = []

        multi_head_V1_list = []
        multi_head_V2_list = []
        multi_head_V3_list = []

        for query1, query2, query3 in zip(self.query1, self.query2, self.query3):
            Q1 = query1(c_q)
            Q2 = query2(h_q)
            Q3 = query3(w_q)
            multi_head_Q1_list.append(Q1)
            multi_head_Q2_list.append(Q2)
            multi_head_Q3_list.append(Q3)
        for key1, key2, key3 in zip(self.key1, self.key2, self.key3):
            k1 = key1(c_k)
            k2 = key2(h_k)
            k3 = key3(w_k)
            multi_head_K1_list.append(k1)
            multi_head_K2_list.append(k2)
            multi_head_K3_list.append(k3)
        for value1, value2, value3 in zip(self.value1, self.value2, self.value3):
            V1 = value1(c_v)
            V2 = value2(h_v)
            V3 = value3(w_v)
            multi_head_V1_list.append(V1)
            multi_head_V2_list.append(V2)
            multi_head_V3_list.append(V3)

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1)
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1)

        multi_head_K1 = torch.stack(multi_head_K1_list, dim=1).transpose(-1, -2)
        multi_head_K2 = torch.stack(multi_head_K2_list, dim=1).transpose(-1, -2)
        multi_head_K3 = torch.stack(multi_head_K3_list, dim=1).transpose(-1, -2)

        multi_head_V1 = torch.stack(multi_head_V1_list, dim=1)
        multi_head_V2 = torch.stack(multi_head_V2_list, dim=1)
        multi_head_V3 = torch.stack(multi_head_V3_list, dim=1)

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K1)
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K2)
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K3)

        attention_scores1 = attention_scores1 / math.sqrt(self.channel_num[0])
        attention_scores2 = attention_scores2 / math.sqrt(self.channel_num[1])
        attention_scores3 = attention_scores3 / math.sqrt(self.channel_num[2])

        attention_probs1 = self.softmax(self.psi(attention_scores1))
        attention_probs2 = self.softmax(self.psi(attention_scores2))
        attention_probs3 = self.softmax(self.psi(attention_scores3))

        attention_probs1 = self.attn_dropout(attention_probs1)
        attention_probs2 = self.attn_dropout(attention_probs2)
        attention_probs3 = self.attn_dropout(attention_probs3)

        context_layer1 = torch.matmul(attention_probs1, multi_head_V1)
        context_layer2 = torch.matmul(attention_probs2, multi_head_V2)
        context_layer3 = torch.matmul(attention_probs3, multi_head_V3)

        context_layer1 = context_layer1.permute(0, 2, 3, 1).contiguous()
        context_layer2 = context_layer2.permute(0, 2, 3, 1).contiguous()
        context_layer3 = context_layer3.permute(0, 2, 3, 1).contiguous()

        context_layer1 = context_layer1.mean(dim=3)
        context_layer2 = context_layer2.mean(dim=3)
        context_layer3 = context_layer3.mean(dim=3)

        O1 = self.out1(context_layer1)
        O2 = self.out2(context_layer2)
        O3 = self.out3(context_layer3)

        O1 = self.proj_dropout(O1)
        O2 = self.proj_dropout(O2)
        O3 = self.proj_dropout(O3)

        O1 = c + O1
        O2 = h + O2
        O3 = w + O3

        x1 = self.ffn_norm1(O1)
        x2 = self.ffn_norm2(O2)
        x3 = self.ffn_norm3(O3)

        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)

        O1 = x1 + O1
        O2 = x2 + O2
        O3 = x3 + O3
        O1 = self.linear2(O1)

        return O1, O2, O3