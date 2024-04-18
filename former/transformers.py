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
        self.dist1 = nn.Linear(emb, num_tokens)
        self.dist2 = nn.Linear(emb, num_tokens)
        self.dist3 = nn.Linear(emb, num_tokens)

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
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=x.device))[
            None, :, :
        ].expand(b, t, e)
        x = tokens + positions

        # Calculate the indices for the distillation points
        fourth_depth = len(self.tblocks) // 4
        dist_points = [fourth_depth - 1, 2 * fourth_depth - 1, 3 * fourth_depth - 1]

        # Initialize outputs for the distillation points
        dist_output_1st = None
        dist_output_2nd = None
        dist_output_3rd = None

        for i, block in enumerate(self.tblocks):
            x = block(x)

            # Capture the outputs at the distillation points
            if i == dist_points[0]:
                dist_output_1st = x
            elif i == dist_points[1]:
                dist_output_2nd = x
            elif i == dist_points[2]:
                dist_output_3rd = x

        # Apply the top layer to the distillation outputs if needed
        y_1st = None if dist_output_1st is None else self.toprobs(dist_output_1st)
        y_2nd = None if dist_output_2nd is None else self.toprobs(dist_output_2nd)
        y_3rd = None if dist_output_3rd is None else self.toprobs(dist_output_3rd)

        x = self.toprobs(x)

        return y_1st, y_2nd, y_3rd, x
