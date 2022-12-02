import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from rotary_embedding_torch import RotaryEmbedding
from sru import SRUpp
from gconv_standalone import GConv
from mega_pytorch import MegaLayer

import math
import pandas as pd


num_features = 5
num_periods = 9
num_cuts = 10

eur_usd_medians = torch.Tensor(pd.read_csv("data/OANDA_DS/EUR_USD.ds/cuts.csv").values.T).cuda()



class CausalConvolution(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_conv = nn.Conv1d(
            input_size, hidden_size, kernel_size = kernel_size,
            padding = 0, bias = True
        )
        
        self.gelu = nn.GELU()
        
        self.out_proj = nn.Linear(
            hidden_size, hidden_size, bias = False
        )
        
        
    def forward(self, hidden_states):
        # batch len, seq len, embedding -> batch len, embedding, seq len (conv1d input format)
        mod = hidden_states.permute(0, 2, 1)
        
        # padding to ensure causality
        mod = F.pad(mod, pad=(self.kernel_size-1, 0), mode='constant', value=0)
        mod = self.in_conv(mod)
        
        # unpermute
        mod = mod.permute(0, 2, 1)
                
        mod = self.gelu(mod)
        
        mod = self.out_proj(mod)
        
        return mod



class SRUConfig(PretrainedConfig):
    model_type = "SRUTrader"
    def __init__(self, n_embd = 256, n_head = 8, hidden_dropout_prob = .1,
                 initializer_range = None):
        super().__init__(
            n_embd = n_embd,
            n_head = n_head,
            hidden_dropout_prob = hidden_dropout_prob
        )



class SRUTrader(PreTrainedModel):
    config_class = SRUConfig
    
    def __init__(self, config):
        super().__init__(config)
                
        self.conv_embed = CausalConvolution(
            input_size = num_features, hidden_size = config.n_embd, kernel_size = 5
        )
        
        self.sru = SRUpp(input_size = config.n_embd,
                         hidden_size = config.n_embd,
                         proj_size = 4 * config.n_embd,
                         num_layers = 10, # paper seemed to have tuned to find this to work
                         dropout = config.hidden_dropout_prob,
                         attn_dropout = 0,
                         rescale = True,
                         layer_norm = True,
                         num_heads = config.n_head,
                         attention_every_n_layers = 2,
                         amp_recurrence_fp16 = True)
        
        self.logits = nn.Linear(config.n_embd, num_periods * num_cuts, bias = False)

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing = .1)
        
        
    def _init_weights(self, module):
        # just for loading model
        pass


    def forward(self, ohlcv, labels = None, future = None, std_future = None):
        batch_size, seq_len, _ = ohlcv.shape
        
        embed = self.conv_embed(ohlcv)
        embed = torch.permute(embed, (1, 0, 2)) # sequence first for SRU
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device = ohlcv.device), diagonal = 1) * -10000.0
        hidden = torch.permute(
            self.sru(embed, attn_mask = mask)[0], (1, 0, 2)
        )
        
        logits = self.logits(hidden)
        
        probas = F.softmax(logits.reshape(batch_size, seq_len, num_periods, num_cuts), dim = -1)
        down_probs, up_probs = probas.chunk(2, dim = -1)
        down_prob = down_probs.sum(dim = -1)
        up_prob = up_probs.sum(dim = -1)
        
        outcome_expectation = probas * eur_usd_medians
        risks, rewards = outcome_expectation.chunk(2, dim = -1)
        risk = -(risks.sum(dim = -1) / down_prob)
        reward = (rewards.sum(dim = -1) / up_prob)
        
        buy_kelly = up_prob - (1 - up_prob) / (reward / risk)
        sell_kelly = down_prob - (1 - down_prob) / (risk / reward)
        soft_trade = torch.where(buy_kelly > sell_kelly, buy_kelly, -sell_kelly)
        
        if labels is None:
            return soft_trade
                
        ce_loss = self.ce_loss(logits.reshape(-1, num_cuts), labels.long().reshape(-1))
        loss = ce_loss
        
#         labels = F.one_hot(labels.long(), num_cuts)
#         loss = .1 * ce_loss + self.l1(probas.reshape(-1, num_cuts), labels.reshape(-1, num_cuts))

        # lq loss
#         q = .2
#         loss = ((1 - (probas * labels).sum(dim = -1) ** q) / q).mean()
        
        soft_profit = soft_trade * future
        return {
            'loss': loss,
            'profits': soft_profit,
            'trades': soft_trade,
        }
    
    
    
class SGConvConfig(PretrainedConfig):
    model_type = "SRUTrader"
    def __init__(self, n_embd = 256, n_head = 8, hidden_dropout_prob = .1,
                 initializer_range = None):
        super().__init__(
            n_embd = n_embd,
            n_head = n_head,
            hidden_dropout_prob = hidden_dropout_prob
        )
        
        
        
class SGConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mega_prenorm = nn.LayerNorm(config.n_embd)
        self.gconv_layer = GConv(
                d_model=config.n_embd,
                d_state=64,
                channels=config.n_head,
                dropout=config.hidden_dropout_prob,
                l_max=1440,
                bidirectional=False,
                transposed=False
        )
        
        self.ff_prenorm = nn.LayerNorm(config.n_embd)
        self.Wgates = nn.Linear(config.n_embd, config.n_embd*4)
        self.Wvalues = nn.Linear(config.n_embd, config.n_embd*4)
        self.proj = nn.Linear(config.n_embd*4, config.n_embd)


    def forward(self, mod):
        mod = self.gconv_layer(self.mega_prenorm(mod))[0] + mod # residual
        
        residual = mod
        
        # SwiGLU
        mod = self.ff_prenorm(mod)
        gates = F.silu(self.Wgates(mod))
        values = self.Wvalues(mod)
        
        return self.proj(gates * values) + residual



class SGConvTrader(PreTrainedModel):
    config_class = SGConvConfig
    
    def __init__(self, config):
        super().__init__(config)
                
        self.conv_embed = CausalConvolution(
            input_size = num_features, hidden_size = config.n_embd, kernel_size = num_features
        )
        
        n_layer = round((math.log(config.n_embd) - 5.039) / 5.55e-2)
        n_layer = max(1, n_layer)
        print(f'Using {n_layer} layers')
        
        self.layers = nn.ModuleList([SGConvBlock(config) for _ in range(n_layer)])
        
        self.logits = nn.Linear(config.n_embd, num_periods * num_cuts, bias = False)

        self.ce_loss = nn.CrossEntropyLoss()
        
        
    def _init_weights(self, module):
        # just for loading model
        pass


    def forward(self, ohlcv, labels = None, future = None, std_future = None):
        batch_size, seq_len, _ = ohlcv.shape
        
        embed = self.conv_embed(ohlcv)
        for layer in self.layers:
            hidden = layer(embed)
        logits = self.logits(hidden)
        
        probas = F.softmax(logits.reshape(batch_size, seq_len, num_periods, num_cuts), dim = -1)
        down_probs, up_probs = probas.chunk(2, dim = -1)
        down_prob = down_probs.sum(dim = -1)
        up_prob = up_probs.sum(dim = -1)
        
        outcome_expectation = probas * eur_usd_medians
        risks, rewards = outcome_expectation.chunk(2, dim = -1)
        risk = -(risks.sum(dim = -1) / down_prob)
        reward = (rewards.sum(dim = -1) / up_prob)
        
        buy_kelly = up_prob - (1 - up_prob) / (reward / risk)
        sell_kelly = down_prob - (1 - down_prob) / (risk / reward)
        soft_trade = torch.where(buy_kelly > sell_kelly, buy_kelly, -sell_kelly)
        
        if labels is None:
            return soft_trade
                
        ce_loss = self.ce_loss(logits.reshape(-1, num_cuts), labels.long().reshape(-1))
        loss = ce_loss
        
#         labels = F.one_hot(labels.long(), num_cuts)
#         loss = .1 * ce_loss + self.l1(probas.reshape(-1, num_cuts), labels.reshape(-1, num_cuts))

        # lq loss
#         q = .2
#         loss = ((1 - (probas * labels).sum(dim = -1) ** q) / q).mean()
        
        soft_profit = soft_trade * future
        return {
            'loss': loss,
            'profits': soft_profit,
            'trades': soft_trade,
        }
    


class MegaConfig(PretrainedConfig):
    model_type = "MegaTrader"
    def __init__(self, n_embd = 256, n_head = 8, hidden_dropout_prob = .1,
                 initializer_range = None):
        super().__init__(
            n_embd = n_embd,
            n_head = n_head,
            hidden_dropout_prob = hidden_dropout_prob
        )



class MegaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mega_prenorm = nn.LayerNorm(config.n_embd)
        self.mega_layer = MegaLayer(
            dim = config.n_embd,
            ema_heads = config.n_head,
            attn_dim_qk = 64,
            attn_dim_value = config.n_embd,
            laplacian_attn_fn = False,
        )
        
        self.ff_prenorm = nn.LayerNorm(config.n_embd)
        self.Wgates = nn.Linear(config.n_embd, config.n_embd*4)
        self.Wvalues = nn.Linear(config.n_embd, config.n_embd*4)
        self.proj = nn.Linear(config.n_embd*4, config.n_embd)


    def forward(self, mod):
        mod = self.mega_layer(self.mega_prenorm(mod))
        
        residual = mod
        
        # SwiGLU
        mod = self.ff_prenorm(mod)
        gates = F.silu(self.Wgates(mod))
        values = self.Wvalues(mod)
        
        return self.proj(gates * values) + residual



class MegaTrader(PreTrainedModel):
    config_class = MegaConfig
    
    def __init__(self, config):
        super().__init__(config)
                
        self.conv_embed = CausalConvolution(
            input_size = num_features, hidden_size = config.n_embd, kernel_size = num_features
        )
        
        n_layer = round((math.log(config.n_embd) - 5.039) / 5.55e-2 / 2) # they don't use high depth in paper
        n_layer = max(1, n_layer)
        print(f'Using {n_layer} layers')
        
        self.layers = nn.ModuleList([MegaBlock(config) for _ in range(n_layer)])
        self.logits = nn.Linear(config.n_embd, num_periods * num_cuts, bias = False)

        self.ce_loss = nn.CrossEntropyLoss()
        
        
    def _init_weights(self, module):
        # just for loading model
        pass


    def forward(self, ohlcv, labels = None, future = None, std_future = None):
        batch_size, seq_len, _ = ohlcv.shape
        
        embed = self.conv_embed(ohlcv)
        for layer in self.layers:
            hidden = layer(embed)
        logits = self.logits(hidden)
        
        probas = F.softmax(logits.reshape(batch_size, seq_len, num_periods, num_cuts), dim = -1)
        down_probs, up_probs = probas.chunk(2, dim = -1)
        down_prob = down_probs.sum(dim = -1)
        up_prob = up_probs.sum(dim = -1)
        
        outcome_expectation = probas * eur_usd_medians
        risks, rewards = outcome_expectation.chunk(2, dim = -1)
        risk = -(risks.sum(dim = -1) / down_prob)
        reward = (rewards.sum(dim = -1) / up_prob)
        
        buy_kelly = up_prob - (1 - up_prob) / (reward / risk)
        sell_kelly = down_prob - (1 - down_prob) / (risk / reward)
        soft_trade = torch.where(buy_kelly > sell_kelly, buy_kelly, -sell_kelly)
        
        if labels is None:
            return soft_trade
                
        ce_loss = self.ce_loss(logits.reshape(-1, num_cuts), labels.long().reshape(-1))
        loss = ce_loss
        
#         labels = F.one_hot(labels.long(), num_cuts)
#         loss = .1 * ce_loss + self.l1(probas.reshape(-1, num_cuts), labels.reshape(-1, num_cuts))

        # lq loss
#         q = .2
#         loss = ((1 - (probas * labels).sum(dim = -1) ** q) / q).mean()
        
        soft_profit = soft_trade * future
        return {
            'loss': loss,
            'profits': soft_profit,
            'trades': soft_trade,
        }