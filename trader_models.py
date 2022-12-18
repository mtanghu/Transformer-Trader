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

        assert (num_levels + 1) % 2 == 0, "the number of tradeable levels should be odd"
        self.num_levels = num_levels
        
        self.proj_logits = nn.Linear(hidden_size, num_periods * num_levels)
        self.linspace = nn.Parameter(
            torch.tensor(np.linspace(-1, 1, num_levels)),
            requires_grad = False
        )


    def forward(self, mod):
        batch_size, length, _ = mod.shape
        logits = self.proj_logits(mod).reshape(
            batch_size, length, num_periods, self.num_levels
        )
        probas = F.softmax(logits, dim = -1)
        
        return (probas * self.linspace).sum(dim = -1)



class SGConvConfig(PretrainedConfig):
    model_type = "SGConvTrader"
    def __init__(self, n_embd = 256, n_head = 4, hidden_dropout_prob = .1,
                 kernel_size = 5, num_levels = 41, initializer_range = None):
        super().__init__(
            n_embd = n_embd,
            n_head = n_head,
            hidden_dropout_prob = hidden_dropout_prob,
            kernel_size = kernel_size,
            num_levels = num_levels
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


    def forward(self, ohlcv, future = None, overnight_masks = None):
        batch_size, seq_len, _ = ohlcv.shape
        
        embed = self.conv_embed(ohlcv)
        for layer in self.layers:
            hidden = layer(embed)
        
        soft_trade = self.trade(hidden)
        
        if future is None:
            return soft_trade
        
        # clean up soft trades to get rid of overnight trades
        soft_trade = torch.where(overnight_masks.long() != 1, soft_trade, 0)
        
        std_future = future / future.std(dim = 1).unsqueeze(1)
    
        std_profit = soft_trade * std_future
        
        gains = std_profit[std_profit > 0]
        losses = -std_profit[std_profit <= 0]
        
        loss = (2 * losses + .2).sum() / (gains + .1).sum()

        # scale_factor = torch.clamp(std_profit.detach(), min = -4, max = None)
        # loss = (-.2 * std_profit / (1 + .2 * scale_factor)).mean()
        
        soft_profit = soft_trade * future
        return {
            'loss': loss,
            'profits': soft_profit,
            'trades': soft_trade,
        }