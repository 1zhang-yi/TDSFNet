import torchvision
import torch.nn as nn
import torch
import copy
from torch.nn import LayerNorm
from TDSF.TD import Tuckergeneration
from TDSF.Block_ViT import Block_ViT
from TDSF.Attention_core import Attention_core
sigmoid = nn.Sigmoid()

class Encoder(nn.Module):
    def __init__(self, config, attention_type='chw'):
        super(Encoder, self).__init__()
        self.type = attention_type
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(config.C_scale, eps=1e-6)
        self.encoder_norm2 = LayerNorm(config.rank[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(config.rank[2], eps=1e-6)
        channel = []
        channel.append(config.C_scale)
        channel.append(config.rank[1])
        channel.append(config.rank[2])
        if attention_type == 'chw':
            for _ in range(config.num_layers):
                layer = Block_ViT(config, channel)
                self.layer.append(copy.deepcopy(layer))
        else:
            layer = Attention_core(config, channel)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, core):
        if self.type == 'chw':
            for layer_block in self.layer:
                emb1, emb2, emb3 = layer_block(emb1, emb2, emb3)
        else:
            for layer_block in self.layer:
                emb1, emb2, emb3 = layer_block(emb1, emb2, emb3, core)

        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None

        return emb1, emb2, emb3


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class TDSFNet(nn.Module):
    def __init__(self, class_list, config):
        super(TDSFNet, self).__init__()
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.dropout = nn.Dropout(0.3)

        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm = torchvision.models.resnet50(pretrained=True)

        # define the clinic model
        self.conv1_cli = self.model_clinic.conv1
        self.bn1_cli = self.model_clinic.bn1
        self.relu_cli = self.model_clinic.relu
        self.maxpool_cli = self.model_clinic.maxpool
        self.layer1_cli = self.model_clinic.layer1
        self.layer2_cli = self.model_clinic.layer2
        self.layer3_cli = self.model_clinic.layer3

        self.conv1_derm = self.model_derm.conv1
        self.bn1_derm = self.model_derm.bn1
        self.relu_derm = self.model_derm.relu
        self.maxpool_derm = self.model_derm.maxpool
        self.layer1_derm = self.model_derm.layer1
        self.layer2_derm = self.model_derm.layer2
        self.layer3_derm = self.model_derm.layer3
        self.fusion = self.model_derm.layer4

        self.rank = config.rank
        self.tensor = config.extracted_feature
        self.scale_dim = config.scale_dim
        self.core = config.core
        self.tucker_cli = Tuckergeneration(config)
        self.tucker_derm = Tuckergeneration(config)

        self.linear_c = nn.Linear(self.rank[0], self.scale_dim)
        self.linear_h = nn.Linear(self.rank[1], self.scale_dim)
        self.linear_w = nn.Linear(self.rank[2], self.scale_dim)

        self.c_linear = nn.Linear(self.scale_dim, self.core[0])
        self.h_linear = nn.Linear(self.scale_dim, self.core[1])
        self.w_linear = nn.Linear(self.scale_dim, self.core[2])


        self.encoder1 = Encoder(config=config, attention_type='chw')
        self.encoder2 = Encoder(config=config, attention_type='all')
        # self.Cscale_to_tensor = nn.Linear(config.C_scale, config.extracted_feature[0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_fusion_ = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )

        self.fc_fusion = nn.Linear(128, self.num_label)
        self.fc_pn_fusion = nn.Linear(128, self.num_pn)
        self.fc_str_fusion = nn.Linear(128, self.num_str)
        self.fc_pig_fusion = nn.Linear(128, self.num_pig)
        self.fc_rs_fusion = nn.Linear(128, self.num_rs)
        self.fc_dag_fusion = nn.Linear(128, self.num_dag)
        self.fc_bwv_fusion = nn.Linear(128, self.num_bwv)
        self.fc_vs_fusion = nn.Linear(128, self.num_vs)


    def forward(self, x):
        (x_clic, x_derm) = x

        x_clic = self.conv1_cli(x_clic)
        x_clic = self.bn1_cli(x_clic)
        x_clic = self.relu_cli(x_clic)
        x_clic = self.maxpool_cli(x_clic)

        x_derm = self.conv1_derm(x_derm)
        x_derm = self.bn1_derm(x_derm)
        x_derm = self.relu_derm(x_derm)
        x_derm = self.maxpool_derm(x_derm)

        x_clic = self.layer1_cli(x_clic)
        x_derm = self.layer1_derm(x_derm)

        x_clic = self.layer2_cli(x_clic)
        x_derm = self.layer2_derm(x_derm)

        x_clic = self.layer3_cli(x_clic)
        x_derm = self.layer3_derm(x_derm)

        cli_core, cli_c, cli_h, cli_w = self.tucker_cli(x_clic)
        derm_core, derm_c, derm_h, derm_w = self.tucker_derm(x_derm)

        f_core = (cli_core + derm_core) / 2 #[256, 14, 14]
        f_c = (cli_c + derm_c) / 2 #[C_scale, 256]
        f_h = (cli_h + derm_h) / 2 #[14, 14]
        f_w = (cli_w + derm_w) / 2 #[14, 14]

        # f_core = self.conv_core(f_core)#[256, 14, 14]
        c = self.linear_c(f_c)  # (B, C, dim)
        h = self.linear_h(f_h)  # (B, 14, dim)
        w = self.linear_w(f_w)  # (B, 14, dim)

        c_ = c.transpose(-1, -2)  # (B, dim, C)
        h_ = h.transpose(-1, -2)  # (B, dim, 14)
        w_ = w.transpose(-1, -2)  # (B, dim, 14)

        encoded_c, encoded_h, encoded_w = self.encoder1(c_, h_, w_, f_core)
        encoded_c, encoded_h, encoded_w = self.encoder2(encoded_c, encoded_h, encoded_w, f_core)

        # encoded_c = self.Cscale_to_tensor(encoded_c) #(B, dim, 1024)
        encoded_c = encoded_c.transpose(-1, -2)  # (B, 1024, dim)
        encoded_h = encoded_h.transpose(-1, -2)  # (B, 14, dim)
        encoded_w = encoded_w.transpose(-1, -2)  # (B, 14, dim)

        encoded_c = self.c_linear(encoded_c) #(B, 1024, 256)
        encoded_h = self.h_linear(encoded_h) #(B, 14, 14)
        encoded_w = self.w_linear(encoded_w) #(B, 14, 14)

        fusion_weight = torch.einsum('busv, biu, bjs, bkv -> bijk', f_core, encoded_c, encoded_h, encoded_w)
        fusion = self.avgpool(self.fusion(fusion_weight))


        fusion = torch.flatten(fusion, start_dim=1)
        x_fusion = self.fc_fusion_(fusion)

        x_fusion = self.dropout(x_fusion)
        logit_fusion = self.fc_fusion(x_fusion)
        logit_pn_fusion = self.fc_pn_fusion(x_fusion)
        logit_str_fusion = self.fc_str_fusion(x_fusion)
        logit_pig_fusion = self.fc_pig_fusion(x_fusion)
        logit_rs_fusion = self.fc_rs_fusion(x_fusion)
        logit_dag_fusion = self.fc_dag_fusion(x_fusion)
        logit_bwv_fusion = self.fc_bwv_fusion(x_fusion)
        logit_vs_fusion = self.fc_vs_fusion(x_fusion)


        return [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
                 logit_bwv_fusion, logit_vs_fusion)
                ]

    def criterion(self, logit, truth):

        loss = nn.CrossEntropyLoss()(logit, truth)

        return loss

    def criterion1(self, logit, truth):

        loss = nn.L1Loss()(logit, truth)

        return loss

    def metric(self, logit, truth):
        # prob = F.sigmoid(logit)
        _, prediction = torch.max(logit.data, 1)

        acc = torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError
