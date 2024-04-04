from .util import mask_

import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Canonical implementation of multi-head self attention."""

    def __init__(self, emb, heads=8, mask=False, kqnorm=False):
        super().__init__()
        assert (
            emb % heads == 0
        ), f"Embedding dimension ({emb}) should be divisible by number of heads ({heads})"

        self.emb = emb
        self.heads = heads
        self.mask = mask

        # We will break the embedding into `heads` chunks and feed each to a different attention head
        s = emb // heads

        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert (
            e == self.emb
        ), f"Input embedding dim ({e}) should match layer embedding dim ({self.emb})"

        s = e // h

        keys = self.tokeys(x).view(b, t, h, s)
        queries = self.toqueries(x).view(b, t, h, s)
        values = self.tovalues(x).view(b, t, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        # Compute scaled dot-product self-attention
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)

        if self.mask:
            mask_(dot, maskval=float("-inf"), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class Attention(nn.Module):
    """Implementation of attention with the queries, keys and values separated."""

    def __init__(self, emb, heads=8, mask=False, kqnorm=False):
        super().__init__()
        assert (
            emb % heads == 0
        ), f"Embedding dimension ({emb}) should be divisible by number of heads ({heads})"

        self.emb = emb
        self.heads = heads
        self.mask = mask

        # We will break the embedding into `heads` chunks and feed each to a different attention head
        s = emb // heads

        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

    def forward(self, queries, keys, values):
        b, tk, e = keys.size()
        assert queries.size(0) == b and queries.size(2) == e
        tq = queries.size(1)

        h = self.heads
        assert (
            e == self.emb
        ), f"Input embedding dim ({e}) should match layer embedding dim ({self.emb})"

        s = e // h

        keys = self.tokeys(keys).view(b, tk, h, s)
        queries = self.toqueries(queries).view(b, tq, h, s)
        values = self.tovalues(values).view(b, tk, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        queries = queries.transpose(1, 2).contiguous().view(b * h, tq, s)
        keys = keys.transpose(1, 2).contiguous().view(b * h, tk, s)
        values = values.transpose(1, 2).contiguous().view(b * h, tk, s)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, tq, tk)

        if self.mask:
            mask_(dot, maskval=float("-inf"), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, tq, s)
        out = out.transpose(1, 2).contiguous().view(b, tq, s * h)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb,
        heads,
        mask,
        seq_length,
        ff_hidden_mult=4,
        dropout=0.0,
        attention_type="default",
        pos_embedding=None,
        sa_kwargs={},
    ):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask, **sa_kwargs)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.dropout_rate = dropout
        self.do = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)

        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)

        return x

    def update_dropout(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.do = nn.Dropout(self.dropout_rate)
