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


class DistGen(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(
        self,
        emb,
        heads,
        depth,
        seq_length,
        num_tokens,
        attention_type="default",
        distpoint=None,
    ):
        super().__init__()

        self.num_tokens = num_tokens

        self.token_embedding = nn.Embedding(
            embedding_dim=emb, num_embeddings=num_tokens
        )
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        self.toprobs = nn.Linear(emb, num_tokens)
        self.toprobsdist = nn.Linear(emb, num_tokens)

        self.distpoint = distpoint

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    seq_length=seq_length,
                    mask=True,
                    attention_type=attention_type,
                )
            )

        self.tblocks = nn.ModuleList(modules=tblocks)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens,
                and outputs at the distillation point, one layer above, and one layer below it.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=x.device))[
            None, :, :
        ].expand(b, t, e)
        x = tokens + positions

        # Initialize outputs for the distillation point, one layer above, and one layer below
        dist_output = None
        dist_output_below = None
        dist_output_above = None

        for i, block in enumerate(self.tblocks):
            x = block(x) + x

            # Capture the output one layer below the distillation point
            if i == self.distpoint - 1:
                dist_output_below = x

            # Capture the output at the distillation point
            if i == self.distpoint:
                dist_output = x

            # Capture the output one layer above the distillation point
            if i == self.distpoint + 1:
                dist_output_above = x

        x = self.toprobs(x)

        # Optionally, apply the top layer to the distillation outputs if needed
        y_below = (
            None if dist_output_below is None else self.toprobsdist(dist_output_below)
        )
        y = None if dist_output is None else self.toprobsdist(dist_output)
        y_above = (
            None if dist_output_above is None else self.toprobsdist(dist_output_above)
        )

        return x, y_below, y, y_above
