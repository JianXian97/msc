import torch
import torch.nn as nn

from monai.utils import optional_import

einops, _ = optional_import("einops")

torch.manual_seed(0)

class CABlock(nn.Module):
    
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, xq, xkv):
        q = einops.rearrange(self.q(xq), "b h (q l d) -> q b l h d", q=1, l=self.num_heads)
        k, v = einops.rearrange(self.kv(xkv), "b h (kv l d) -> kv b l h d", kv=2, l=self.num_heads)

        att_mat = (torch.einsum("blxd,blyd->blxy", q.squeeze(0), k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
