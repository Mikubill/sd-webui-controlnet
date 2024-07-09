from collections import OrderedDict
import torch
import torch.nn as nn

try:
    from sgm.modules.diffusionmodules.openaimodel import (
        timestep_embedding,
    )

    using_sgm = True
except ImportError:
    from ldm.modules.diffusionmodules.openaimodel import (
        timestep_embedding,
    )

    using_sgm = False


def attention_pytorch(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


class ControlAddEmbedding(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_control_type,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_control_type = num_control_type
        self.in_dim = in_dim
        self.linear_1 = nn.Linear(
            in_dim * num_control_type, out_dim, dtype=dtype, device=device
        )
        self.linear_2 = nn.Linear(out_dim, out_dim, dtype=dtype, device=device)

    def forward(self, control_type, dtype, device):
        c_type = torch.zeros((self.num_control_type,), device=device)
        c_type[control_type] = 1.0
        c_type = (
            timestep_embedding(c_type.flatten(), self.in_dim, repeat_only=False)
            .to(dtype)
            .reshape((-1, self.num_control_type * self.in_dim))
        )
        return self.linear_2(torch.nn.functional.silu(self.linear_1(c_type)))


class OptimizedAttention(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = nhead
        self.c = c

        self.in_proj = nn.Linear(c, c * 3, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.in_proj(x)
        q, k, v = x.split(self.c, dim=2)
        out = attention_pytorch(q, k, v, self.heads)
        return self.out_proj(out)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResBlockUnionControlnet(nn.Module):
    def __init__(self, dim, nhead, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = OptimizedAttention(
            dim, nhead, dtype=dtype, device=device, operations=operations
        )
        self.ln_1 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    (
                        "c_fc",
                        nn.Linear(dim, dim * 4, dtype=dtype, device=device),
                    ),
                    ("gelu", QuickGELU()),
                    (
                        "c_proj",
                        nn.Linear(dim * 4, dim, dtype=dtype, device=device),
                    ),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(dim, dtype=dtype, device=device)

    def attention(self, x: torch.Tensor):
        return self.attn(x)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
