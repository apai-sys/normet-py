"""Physics-Informed Dual-Graph Spatio-Temporal Graph Neural Network (PI-STGNN)."""

from __future__ import annotations

import logging
from functools import cache
from typing import Any, cast

log = logging.getLogger(__name__)


@cache
def get_pi_stgnn_class() -> Any:
    """Lazy import PyTorch PI-STGNN module.

    Building the class inside a function means a fresh ``type`` object on every
    call, so ``isinstance(obj, get_..._class())`` was false even for an object that
    function had just produced. The lookup is memoised so the class is built once
    and keeps a stable identity for the life of the process.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class DualGraphConv(nn.Module):
            def __init__(
                self, in_dim: int, out_dim: int, n_nodes: int = 321, embed_dim: int = 32
            ) -> None:
                super().__init__()
                self.node_embed1 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.05)
                self.node_embed2 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.05)
                self.weight_geo = nn.Linear(in_dim, out_dim, bias=False)
                self.weight_wind = nn.Linear(in_dim, out_dim, bias=False)
                self.weight_adapt = nn.Linear(in_dim, out_dim, bias=False)
                self.weight_self = nn.Linear(in_dim, out_dim, bias=True)
                self.layer_norm = nn.LayerNorm(out_dim)

            def forward(
                self, x: torch.Tensor, A_geo: torch.Tensor, A_wind: torch.Tensor
            ) -> torch.Tensor:
                B, N, _ = x.shape
                h_geo = torch.matmul(A_geo.unsqueeze(0), x)
                out_geo = self.weight_geo(h_geo)

                A_wind_b = A_wind.unsqueeze(0).expand(B, -1, -1) if A_wind.dim() == 2 else A_wind
                h_wind = torch.bmm(A_wind_b, x)
                out_wind = self.weight_wind(h_wind)

                A_adapt = F.softmax(F.relu(torch.mm(self.node_embed1, self.node_embed2.T)), dim=-1)
                h_adapt = torch.matmul(A_adapt.unsqueeze(0), x)
                out_adapt = self.weight_adapt(h_adapt)

                out_self = self.weight_self(x)
                return self.layer_norm(F.silu(out_geo + out_wind + out_adapt + out_self))

        class TemporalConvBlock(nn.Module):
            def __init__(
                self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1
            ) -> None:
                super().__init__()
                self.pad = (kernel_size - 1) * dilation
                self.conv_filter = nn.Conv1d(
                    in_channels, out_channels, kernel_size, dilation=dilation, padding=self.pad
                )
                self.conv_gate = nn.Conv1d(
                    in_channels, out_channels, kernel_size, dilation=dilation, padding=self.pad
                )
                self.residual = (
                    nn.Conv1d(in_channels, out_channels, kernel_size=1)
                    if in_channels != out_channels
                    else nn.Identity()
                )
                self.layer_norm = nn.LayerNorm(out_channels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, T, C = x.shape
                h = x.view(B * N, T, C).transpose(1, 2)
                f = torch.tanh(self.conv_filter(h)[:, :, :T])
                g = torch.sigmoid(self.conv_gate(h)[:, :, :T])
                res = self.residual(h)[:, :, :T]
                out = (f * g + res).transpose(1, 2).view(B, N, T, -1)
                return self.layer_norm(out)

        class PI_STGNN(nn.Module):
            """PI-STGNN Spatio-Temporal Graph Neural Network."""

            def __init__(
                self,
                in_dim: int = 779,
                hidden_dim: int = 128,
                n_nodes: int = 321,
                n_layers: int = 2,
                dropout: float = 0.1,
            ) -> None:
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.st_blocks = nn.ModuleList()
                for layer in range(n_layers):
                    self.st_blocks.append(
                        nn.ModuleDict(
                            {
                                "spatial": DualGraphConv(hidden_dim, hidden_dim, n_nodes=n_nodes),
                                "temporal": TemporalConvBlock(
                                    hidden_dim, hidden_dim, kernel_size=3, dilation=2**layer
                                ),
                                "norm": nn.LayerNorm(hidden_dim),
                                "dropout": nn.Dropout(dropout),
                            }
                        )
                    )
                self.head_conc = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Softplus(),
                )
                self.head_emission = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Softplus(),
                )
                self.head_lifetime = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )

            def forward(
                self, x: torch.Tensor, A_geo: torch.Tensor, A_wind: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                B, N, T, _ = x.shape
                h = self.input_proj(x)
                for module in self.st_blocks:
                    # nn.ModuleList yields plain Modules as far as the type
                    # stubs are concerned, and Module is not subscriptable.
                    block = cast(nn.ModuleDict, module)
                    h_spatial = []
                    for t_step in range(T):
                        h_t = block["spatial"](h[:, :, t_step, :], A_geo, A_wind)
                        h_spatial.append(h_t.unsqueeze(2))
                    h_s = torch.cat(h_spatial, dim=2)
                    h_t = block["temporal"](h_s)
                    h = block["norm"](h + block["dropout"](h_t))

                C_hat = self.head_conc(h).squeeze(-1)
                S_hat = self.head_emission(h).squeeze(-1)
                h_time_mean = torch.mean(h, dim=2)
                tau_eff = 1.0 + 23.0 * self.head_lifetime(h_time_mean).squeeze(-1)

                return C_hat, S_hat, tau_eff

        return PI_STGNN
    except ImportError as err:
        raise ImportError(
            "PyTorch required for PI_STGNN. Install with `pip install torch`."
        ) from err


def build_pi_stgnn(*args: Any, **kwargs: Any) -> Any:
    """Construct a PI-STGNN, importing PyTorch on first use.

    A function rather than a class: the previous wrapper was a module-level
    ``PI_STGNN`` whose ``__new__`` returned an instance of the *inner*, lazily
    built class, so ``isinstance(build_pi_stgnn(...), PI_STGNN)`` was ``False``
    and the exported name matched nothing the object actually was. Anything
    that needs the class itself -- ``isinstance``, subclassing, type
    annotations -- should call :func:`get_pi_stgnn_class`, which returns the
    real ``nn.Module`` subclass.
    """
    return get_pi_stgnn_class()(*args, **kwargs)
