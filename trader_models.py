import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, PreTrainedModel

from sru import SRUpp

import math



class SRUTrader(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        config.initializer_range = 1 / math.sqrt(config.n_embd)
        
        self.embed = nn.Linear(5, config.n_embd, bias = False)
        self.norm = nn.LayerNorm(config.n_embd)
        self.sru = SRUpp(input_size = config.n_embd,
                         hidden_size = config.n_embd,
                         proj_size = 4 * config.n_embd, # paper recommended
                         num_layers = 10, # paper seemed to have tuned to find this to work
                         dropout = .1,
                         attn_dropout = .1,
                         rescale = True,
                         layer_norm = True,
                         num_heads = config.n_head,
                         attention_every_n_layers = 2,
                         amp_recurrence_fp16 = True)
        
        self.logits = nn.Linear(config.n_embd, 120, bias = False)
        self.trade_sign = nn.Parameter(torch.Tensor([-1, 1]), requires_grad = True)

        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        
        
    def _init_weights(self, module):
        # just for loading model
        pass


    def forward(self, ohlcv, future = None):
        # manual positional embeddings
        batch_size, seq_len, _ = ohlcv.shape
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device = ohlcv.device), diagonal=1) * -10000.0
        
        embed = self.norm(self.embed(ohlcv))
        embed = torch.permute(embed, (1, 0, 2)) # sequence first for SRU
        hidden = torch.permute(
            self.sru(embed, attn_mask = mask)[0], (1, 0, 2)
        )
        
        logits = self.logits(hidden)
        
        soft_trade = F.softmax(logits.reshape(batch_size, seq_len, 60, 2), dim = -1)
        soft_trade = (soft_trade * self.trade_sign).sum(dim = -1)
        soft_profit = soft_trade * future
        
        if future is None:
            return soft_trade
        
        # cost scaled CrossEntropyLoss
        classes = torch.where(future != 0, (future > 0).long(), -100)
        ce_loss = self.ce_loss(logits.reshape(-1, 2), classes.reshape(-1))
        loss = ce_loss.mean()
        
        return {
            'loss': loss,
            'profits': soft_profit,
            'trades': soft_trade,
        }