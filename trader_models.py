import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from flash_attn.modules.mha import MHA
from robust_loss import NCEandRCE

import math
import numpy as np



# globals
num_features = 5
num_periods = 9
num_classes = 64



class CausalConvolution(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_conv = nn.Conv1d(
            input_size, hidden_size, kernel_size = kernel_size,
            padding = 0, bias = False
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



class TraderConfig(PretrainedConfig):
    model_type = "Trader"
    def __init__(self, n_embd = 256, n_head = 4, hidden_dropout_prob = 0,
                 kernel_size = 10, num_levels = 21, max_loss = .9,
                 commission = .01, use_swiglu = False, initializer_range = None):
        super().__init__(
            n_embd = n_embd,
            n_head = n_head,
            hidden_dropout_prob = hidden_dropout_prob,
            kernel_size = kernel_size,
            num_levels = num_levels,
            max_loss = max_loss,
            commission = commission,
            use_swiglu = use_swiglu,
            initializer_range = initializer_range
        )
        
        
        
class MHABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.n_embd, elementwise_affine = False)
        self.attn_layer = MHA(
            embed_dim = config.n_embd,
            num_heads = config.n_head,
            qkv_proj_bias=True,
            out_proj_bias=True,
            dropout = config.hidden_dropout_prob,
            causal = True,
            rotary_emb_dim = config.n_embd // config.n_head,
            rotary_emb_interleaved = True,
            use_flash_attn = True,
        )
        
        self.ff_prenorm = nn.LayerNorm(config.n_embd, elementwise_affine = False)

        self.use_swiglu = config.use_swiglu
        if config.use_swiglu:
            self.Wgates = nn.Linear(config.n_embd, config.n_embd, bias = False)
            self.Wvalues = nn.Linear(config.n_embd, config.n_embd, bias = False)
            self.proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        else:
            self.boom = nn.Linear(config.n_embd, config.n_embd * 2, bias = False)
            self.gelu = nn.GELU()
            self.unboom = nn.Linear(config.n_embd * 2, config.n_embd, bias = False)


    def forward(self, mod):
        mod = self.attn_layer(self.attn_norm(mod))[0] + mod # residual
        
        residual = mod

        mod = self.ff_prenorm(mod)        
        if self.use_swiglu:
            gates = F.silu(self.Wgates(mod))
            values = self.Wvalues(mod)
            mlp_output = self.proj(gates * values)
        else:
            mod = self.gelu(self.boom(mod))
            mlp_output = self.unboom(mod)
        
        return mlp_output + residual



class Trader(PreTrainedModel):
    config_class = TraderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.max_loss = config.max_loss
        self.commission = config.commission
                
        self.conv_embed = CausalConvolution(
            input_size = num_features, hidden_size = config.n_embd,
            kernel_size = config.kernel_size
        )
        
        self.embed_drop = nn.Dropout(config.hidden_dropout_prob)
        
        # levine 2020 number of layers
        n_layer = round((math.log(config.n_embd) - 5.039) / 5.55e-2)
        n_layer = max(2, n_layer) # at least 2 layers for scaling
        print(f'Using {n_layer} layers')
        
        self.layers = nn.ModuleList([
            MHABlock(config) for i in range(n_layer)
        ])
        self.final_norm = nn.LayerNorm(config.n_embd, elementwise_affine = False)
        
        self.trade = SoftTrade(config.n_embd, config.num_levels)
        self.logits = nn.Linear(config.n_embd, num_periods * num_classes)

        # self.classification_loss = nn.CrossEntropyLoss()
        self.classification_loss = NCEandRCE(alpha = 1, beta = 1, num_classes = num_classes)


    def _init_weights(self, module):
        # just for loading model
        pass


    def forward(self, ohlcv, labels = None, overnight_masks = None, classes = None):
        batch_size, seq_len, _ = ohlcv.shape
        future = labels # rename for readability
        
        embed = self.embed_drop(self.conv_embed(ohlcv))
        for layer in self.layers:
            hidden = layer(embed)
            
        # standard for modern transformers
        # hidden = self.final_norm(hidden)
        
        soft_trade = self.trade(hidden)
        
        if future is None:
            return soft_trade
        
        # clean up soft trades to get rid of overnight trades
        if overnight_masks is not None:
            mask = torch.where(overnight_masks.long() != 1, 1, 0)
            soft_trade = soft_trade * mask
        
        soft_profit = soft_trade * future

        # floor losses (so that the log can operate correctly), notice detach
        floor_mask = torch.where(
            soft_profit > -self.max_loss, 1, -self.max_loss / soft_profit.detach()
        )
        
        # also ceiling the gains for symmetry
        ceiling_mask =  torch.where(
            soft_profit < self.max_loss, 1, self.max_loss / soft_profit.detach()
        )
        
        # apply mask(s)
        capped_profit = soft_profit * floor_mask * ceiling_mask        
        
        # apply commission fee to floored profit
        capped_profit = capped_profit - soft_trade.abs() * self.commission
        
        # negative log return loss function (i.e. growth maximization) 
        trade_loss = -torch.log1p(capped_profit).mean()

        # classification loss (to help with price distribution learning)
        logits = self.logits(hidden)
        # classes = torch.where(overnight_masks.long() != 1, classes, -100)
        class_loss = self.classification_loss(
            logits.reshape(-1, num_classes),
            classes.long().reshape(-1)
        )
        class_loss = torch.where(overnight_masks.reshape(-1) != 1, class_loss, 0).mean()
    
        loss = trade_loss + class_loss

        return {
            'loss': loss,
            'classification loss': class_loss,
            'trade loss': trade_loss,
            'profits': soft_profit,
            'trades': soft_trade,
        }