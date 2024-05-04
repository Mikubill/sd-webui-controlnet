import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable


@dataclass
class PuLIDAttnSetting:
    num_zero: int = 0
    ortho: bool = False
    ortho_v2: bool = False

    def eval(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        id_embedding: torch.Tensor,
        attn_heads: int,
        head_dim: int,
        id_to_k: Callable[[torch.Tensor], torch.Tensor],
        id_to_v: Callable[[torch.Tensor], torch.Tensor],
    ):
        assert hidden_states.ndim == 3
        batch_size, sequence_length, inner_dim = hidden_states.shape

        if self.num_zero == 0:
            id_key = id_to_k(id_embedding).to(query.dtype)
            id_value = id_to_v(id_embedding).to(query.dtype)
        else:
            zero_tensor = torch.zeros(
                (id_embedding.size(0), self.num_zero, id_embedding.size(-1)),
                dtype=id_embedding.dtype,
                device=id_embedding.device,
            )
            id_key = id_to_k(torch.cat((id_embedding, zero_tensor), dim=1)).to(
                query.dtype
            )
            id_value = id_to_v(torch.cat((id_embedding, zero_tensor), dim=1)).to(
                query.dtype
            )

        id_key = id_key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
        id_value = id_value.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        id_hidden_states = F.scaled_dot_product_attention(
            query, id_key, id_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        id_hidden_states = id_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn_heads * head_dim
        )
        id_hidden_states = id_hidden_states.to(query.dtype)

        if not self.ortho and not self.ortho_v2:
            return id_hidden_states
        elif self.ortho_v2:
            orig_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            id_hidden_states = id_hidden_states.to(torch.float32)
            attn_map = query @ id_key.transpose(-2, -1)
            attn_mean = attn_map.softmax(dim=-1).mean(dim=1)
            attn_mean = attn_mean[:, :, :5].sum(dim=-1, keepdim=True)
            projection = (
                torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                * hidden_states
            )
            orthogonal = id_hidden_states + (attn_mean - 1) * projection
            return orthogonal.to(orig_dtype)
        else:
            orig_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            id_hidden_states = id_hidden_states.to(torch.float32)
            projection = (
                torch.sum((hidden_states * id_hidden_states), dim=-2, keepdim=True)
                / torch.sum((hidden_states * hidden_states), dim=-2, keepdim=True)
                * hidden_states
            )
            orthogonal = id_hidden_states - projection
            return orthogonal.to(orig_dtype)


PULID_SETTING_FIDELITY = PuLIDAttnSetting(
    num_zero=8,
    ortho=False,
    ortho_v2=True,
)

PULID_SETTING_STYLE = PuLIDAttnSetting(
    num_zero=16,
    ortho=True,
    ortho_v2=False,
)
