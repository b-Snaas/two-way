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

        self.depth = depth
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

    def forward(self, x, return_intermediate=False):
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(
            b, t, e
        )
        x = tokens + positions

        # Intermediate logits storage
        intermediate_logits = None

        # Determine the 1/3th layer index
        intermediate_layer_index = self.depth // 3

        for i, block in enumerate(self.tblocks):
            x = block(x)
            if i == intermediate_layer_index:
                # Capture the intermediate logits
                intermediate_logits = self.toprobs(x.view(b * t, e)).view(
                    b, t, self.num_tokens
                )

        final_logits = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        if return_intermediate:
            return F.log_softmax(final_logits, dim=2), F.log_softmax(
                intermediate_logits, dim=2
            )
        else:
            return F.log_softmax(final_logits, dim=2)

    def DistillLoss(y_true, y_pred_final, y_pred_intermediate, alpha=0.5, T=1):
        print("Inputs:")
        print("y_true shape:", y_true.shape)
        print("y_pred_final shape:", y_pred_final.shape)
        print("y_pred_intermediate shape:", y_pred_intermediate.shape)
        print("alpha:", alpha)
        print("T:", T)

        # Traditional Cross-Entropy Loss
        ce_loss = F.cross_entropy(y_pred_intermediate, y_true)
        print("\nAfter Cross-Entropy Loss:")
        print("ce_loss:", ce_loss.item())

        # Distillation Loss (KL Divergence)
        soft_pred_final = F.log_softmax(y_pred_final / T, dim=1)
        soft_pred_intermediate = F.softmax(y_pred_intermediate / T, dim=1)
        print("\nAfter Softening Probabilities:")
        print(
            "soft_pred_final stats - mean:",
            soft_pred_final.mean().item(),
            "min:",
            soft_pred_final.min().item(),
            "max:",
            soft_pred_final.max().item(),
        )
        print(
            "soft_pred_intermediate stats - mean:",
            soft_pred_intermediate.mean().item(),
            "min:",
            soft_pred_intermediate.min().item(),
            "max:",
            soft_pred_intermediate.max().item(),
        )

        distillation_loss = F.kl_div(
            soft_pred_intermediate, soft_pred_final.detach(), reduction="batchmean"
        ) * (T * T)
        print("\nAfter Distillation Loss:")
        print("distillation_loss:", distillation_loss.item())

        # Combined Loss
        total_loss = alpha * ce_loss + (1 - alpha) * distillation_loss
        print("\nFinal Combined Loss:")
        print("total_loss:", total_loss.item())

        return total_loss
