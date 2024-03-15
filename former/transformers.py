import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

from .util import d


class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(
        self, emb, heads, depth, seq_length, num_tokens, attention_type="default"
    ):

        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(
            embedding_dim=emb, num_embeddings=num_tokens
        )
        self.pos_embedding = nn.Embedding(
            embedding_dim=emb,
            num_embeddings=(
                seq_length * 2 - 1 if attention_type == "relative" else seq_length
            ),
        )

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    seq_length=seq_length,
                    mask=True,
                    attention_type=attention_type,
                    pos_embedding=self.pos_embedding,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(
            b, t, e
        )
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)
