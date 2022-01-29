# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base_model import base_model
from transformers import BertModel, BertConfig
LARGE_NUM = 1e9
class ContrastiveNetwork(base_model):
    def __init__(self, config, encoder, dropout_layer, hidden_size):
        super(ContrastiveNetwork, self).__init__()
        self.encoder = encoder
        self.dropout_layer = dropout_layer
        self.hidden_size = hidden_size
        self.output_size = config.encoder_output_size  # for classification number
        self.projector = nn.Linear(hidden_size, self.output_size)
        self.relu = nn.GELU()
        self.projector1 = nn.Linear(self.output_size, self.output_size // 2)
        self.temperature = config.temperature
        self.layer_normalization = nn.LayerNorm([config.output_size])

    def forward(self, enc_inp, proj_inp, comparison, memory_network=None, mem_for_batch=None, FUN_CODE=1):
        if FUN_CODE == 2:
            mid_hidden = self.encoder(enc_inp)
            if mid_hidden.shape[1] != self.hidden_size:
                mid_hidden = self.dim_trans(mid_hidden)
            proj_inp = self.emb_trans(proj_inp)
            hidden = self.projector1(
                self.layer_normalization(self.relu(self.projector(torch.cat([mid_hidden, proj_inp], dim=0)))))
            hidden = F.normalize(hidden, dim=-1, p=2)
            # hidden = torch.linalg.norm(hidden, dim=-1)
            hidden1, hidden2 = torch.split(hidden, [len(enc_inp), len(proj_inp)], dim=0)
            logits_aa = torch.matmul(hidden1, torch.transpose(hidden2, -1, -2)) / self.temperature  # B1*B2
            logits_aa = logits_aa + (comparison == 0).float() * (-LARGE_NUM)  # mask#B1 * B2
            return logits_aa
        elif FUN_CODE == 0:  # mem
            self.encoder.eval()
            self.dropout_layer.eval()
            with torch.no_grad():
                right = self.dropout_layer(self.encoder(proj_inp))[1]
            right.detach_()
            self.encoder.train()
            self.dropout_layer.train()
            left = self.dropout_layer(self.encoder(enc_inp))[1]
            hidden = memory_network(torch.cat([left, right], dim=0), mem_for_batch)
            hidden = self.projector1(self.layer_normalization(self.relu(self.projector(hidden))))
            hidden = F.normalize(hidden, dim=-1, p=2)
            hidden1, hidden2 = torch.split(hidden, [len(left), len(right)], dim=0)
            logits_aa = torch.matmul(hidden1, torch.transpose(hidden2, -1, -2)) / self.temperature  # B*K
            logits_aa = logits_aa + (comparison == 0).float() * (-LARGE_NUM)  # mask#B1 * B2
            return logits_aa
        elif FUN_CODE == 1:
            self.encoder.eval()
            self.dropout_layer.eval()
            with torch.no_grad():
                right = self.dropout_layer(self.encoder(proj_inp))[1]
            right.detach_()
            self.encoder.train()
            self.dropout_layer.train()
            left = self.dropout_layer(self.encoder(enc_inp))[1]
            hidden = torch.cat([left, right], dim=0)
            hidden = self.projector1(self.layer_normalization(self.relu(self.projector(hidden))))
            hidden = F.normalize(hidden, dim=-1, p=2)
            hidden1, hidden2 = torch.split(hidden, [len(left), len(right)], dim=0)
            logits_aa = torch.matmul(hidden1, torch.transpose(hidden2, -1, -2)) / self.temperature  # B*K
            logits_aa = logits_aa + (comparison == 0).float() * (-LARGE_NUM)  # mask#B1 * B2
            return logits_aa
        elif FUN_CODE == 3:
            left = self.dropout_layer(self.encoder(enc_inp))[1]
            hidden = memory_network(torch.cat([left, proj_inp], dim=0), mem_for_batch)
            hidden = self.projector1(self.layer_normalization(self.relu(self.projector(hidden))))
            hidden = F.normalize(hidden, dim=-1, p=2)
            hidden1, hidden2 = torch.split(hidden, [len(left), len(proj_inp)], dim=0)
            logits_aa = torch.matmul(hidden1, torch.transpose(hidden2, -1, -2)) / self.temperature  # B*K
            logits_aa = logits_aa + (comparison == 0).float() * (-LARGE_NUM)  # mask#B1 * B2
            return logits_aa
        elif FUN_CODE == 4:  # no mem
            left = self.dropout_layer(self.encoder(enc_inp))[1]
            hidden = torch.cat([left, proj_inp], dim=0)
            hidden = self.projector1(self.layer_normalization(self.relu(self.projector(hidden))))
            hidden = F.normalize(hidden, dim=-1, p=2)
            hidden1, hidden2 = torch.split(hidden, [len(left), len(proj_inp)], dim=0)
            logits_aa = torch.matmul(hidden1, torch.transpose(hidden2, -1, -2)) / self.temperature  # B*K
            logits_aa = logits_aa + (comparison == 0).float() * (-LARGE_NUM)  # mask#B1 * B2
            return logits_aa
        elif FUN_CODE ==5: #picture
            left = self.dropout_layer(self.encoder(enc_inp))[1]
            hidden = torch.cat([left, proj_inp], dim=0)
            hidden = self.projector1(self.layer_normalization(self.relu(self.projector(hidden))))
            hidden = F.normalize(hidden, dim=-1, p=2)
            hidden1, hidden2 = torch.split(hidden, [len(left), len(proj_inp)], dim=0)
            return np.array(hidden1.cpu()), np.array(hidden2.cpu())