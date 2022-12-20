import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from rotary_embedding_torch import RotaryEmbedding
from sru import SRUpp
from gconv_standalone import GConv
from mega_pytorch import MegaLayer

import math
import numpy as np



# globals
num_features = 4
num_periods = 9



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



class SoftTrade(nn.Module):
    def __init__(self, hidden_size, num_levels):
        super().__init__()
        self.num_levels = num_levels
        
        self.proj_signal = nn.Linear(hidden_size, num_periods * num_levels)
        self.proj_gate = nn.Linear(hidden_size, num_periods * num_levels)


    def forward(self, hidden_state):
        batch_size, length, _ = hidden_state.shape
        
        signals = torch.tanh(self.proj_signal(hidden_state).reshape(
            batch_size, length, num_periods, self.num_levels
        ))
        gates = torch.sigmoid(self.proj_gate(hidden_state).reshape(
            batch_size, length, num_periods, self.num_levels
        ))
        
        return (signals * gates).sum(dim = -1) / self.num_levels

    def elu_softmax(self, logits, dim = -1):
        positive = (F.elu(logits) + 1)
        probas = positive / positive.sum(dim = dim, keepdim = True)
        return probas



class SGConvConfig(PretrainedConfig):
    model_type = "SGConvTrader"
    def __init__(self, n_embd = 256, n_head = 4, hidden_dropout_prob = .1,
                 kernel_size = 5, num_levels = 41, max_loss = .9,
                 initializer_range = None):
        super().__init__(
            n_embd = n_embd,
            n_head = n_head,
            hidden_dropout_prob = hidden_dropout_prob,
            kernel_size = kernel_size,
            num_levels = num_levels,
            max_loss = max_loss,
            initializer_range = initializer_range
        )
        
        
        
class SGConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_gconv = nn.Sequential(
            nn.LayerNorm(config.n_embd), nn.Linear(config.n_embd, config.n_embd)
        )
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
        mod = self.gconv_layer(self.pre_gconv(mod))[0] + mod # residual
        
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
        self.max_loss = config.max_loss
                
        self.conv_embed = CausalConvolution(
            input_size = num_features, hidden_size = config.n_embd,
            kernel_size = config.kernel_size
        )
        
        n_layer = round((math.log(config.n_embd) - 5.039) / 5.55e-2)
        n_layer = max(1, n_layer)
        print(f'Using {n_layer} layers')
        
        self.layers = nn.ModuleList([SGConvBlock(config) for _ in range(n_layer)])
        
        self.trade = SoftTrade(config.n_embd, config.num_levels)


    def _init_weights(self, module):
        # just for loading model
        pass


    def forward(self, ohlcv, labels = None, overnight_masks = None):
        batch_size, seq_len, _ = ohlcv.shape
        future = labels # rename for readability
        
        embed = self.conv_embed(ohlcv)
        for layer in self.layers:
            hidden = layer(embed)
        
        soft_trade = self.trade(hidden)
        
        if future is None:
            return soft_trade
        
        # clean up soft trades to get rid of overnight trades
        if overnight_masks is not None:
            mask = torch.where(overnight_masks.long() != 1, 1, 0)
            soft_trade = soft_trade * mask
        
        soft_profit = soft_trade * future

        # floor losses (so that the log can operate correctly), notice detach
        adjustment = torch.where(
            soft_profit > -self.max_loss, 1, -self.max_loss / soft_profit.detach()
        )
        floored_profit = soft_profit * adjustment
        
        # apply commission fee to floored profit (3% is 6 pips at 500x leverage)
        floored_profit = floored_profit - soft_trade.abs() * .03
        
        # negative log return loss function (i.e. growth maximization) 
        loss = -torch.log(1 + soft_profit).mean()
        
        return {
            'loss': loss,
            'profits': soft_profit,
            'trades': soft_trade,
        }