"""Advection-Diffusion-Reaction (ADR) Mass Conservation PDE Loss for Irregular Graphs."""

from __future__ import annotations

import logging
from functools import cache
from typing import Any

log = logging.getLogger(__name__)


@cache
def get_adr_pde_loss_class() -> Any:
    """Lazy import PyTorch ADR-PDE Loss module.

    Building the class inside a function means a fresh ``type`` object on every
    call, so ``isinstance(obj, get_..._class())`` was false even for an object that
    function had just produced. The lookup is memoised so the class is built once
    and keeps a stable identity for the life of the process.
    """
    try:
        import torch
        import torch.nn as nn

        class ADR_PDE_Loss(nn.Module):
            """Physics Mass Conservation Loss.

            ``|| dC/dt + u.grad(C) - div(K grad(C)) - S + C/tau ||^2``, evaluated
            on an irregular station graph. See ``direction`` for how the wind
            graph enters the advection term.
            """

            def __init__(
                self,
                kappa: float = 0.4,
                scale_loss: float = 1.0,
                direction: str = "upwind",
            ) -> None:
                """
                Args:
                    kappa: Turbulent diffusivity scaling coefficient.
                    scale_loss: Multiplier applied to the mean squared residual.
                    direction: How the wind graph enters the advection term.
                        ``"upwind"`` (default) evaluates u.grad(C) at node i from
                        its *upwind* neighbours, the standard upwind discretisation
                        and the one a mass-conservation residual at a receptor
                        needs. ``"outflow"`` keeps the original convention, where
                        the term at node i is the flux it exports to its downwind
                        neighbours; under it a receptor with no downwind neighbour
                        carries no advection term at all.
                """
                super().__init__()
                if direction not in ("upwind", "outflow"):
                    raise ValueError(f"direction must be 'upwind' or 'outflow', got {direction!r}")
                self.kappa = kappa
                self.scale_loss = scale_loss
                self.direction = direction

            def forward(
                self,
                C_hat: torch.Tensor,
                S_hat: torch.Tensor,
                tau_eff: torch.Tensor,
                dC_dt: torch.Tensor,
                dist_matrix: torch.Tensor,
                A_geo: torch.Tensor,
                A_wind: torch.Tensor,
                u10: torch.Tensor,
                v10: torch.Tensor,
                ws: torch.Tensor,
                blh: torch.Tensor,
            ) -> torch.Tensor:
                B, N, T = C_hat.shape
                dist_m = torch.clamp(dist_matrix * 1000.0, min=500.0)

                # 1. Advection
                # A_wind[i, j] weights j *downwind* of i, so summing over j gives
                # node i the neighbours it exports to. Under "upwind" the matrix
                # is transposed first, which makes the sum run over the
                # neighbours that advect into i instead.
                A_dir = A_wind.transpose(-2, -1) if self.direction == "upwind" else A_wind
                A_w = A_dir.unsqueeze(0).expand(B, -1, -1) if A_dir.dim() == 2 else A_dir
                advection_op = A_w / (dist_m.unsqueeze(0) + 1e-5)
                adv_term = torch.zeros_like(C_hat)
                for t_step in range(T):
                    C_t = C_hat[:, :, t_step].unsqueeze(-1)
                    diff_C = C_t - C_t.transpose(1, 2)
                    adv_flux = torch.sum(advection_op * diff_C, dim=-1)
                    adv_term[:, :, t_step] = adv_flux * ws[:, :, t_step]

                # 2. Turbulent Diffusion
                K_diff = torch.clamp(
                    self.kappa * (0.05 * ws) * torch.clamp(blh, min=100.0), min=5.0, max=500.0
                )
                laplacian_op = A_geo / ((dist_m**2) + 1e-5)
                diff_term = torch.zeros_like(C_hat)
                for t_step in range(T):
                    C_t = C_hat[:, :, t_step].unsqueeze(-1)
                    diff_C = C_t.transpose(1, 2) - C_t
                    lap_flux = torch.sum(laplacian_op.unsqueeze(0) * diff_C, dim=-1)
                    diff_term[:, :, t_step] = K_diff[:, :, t_step] * lap_flux

                # 3. Reaction Sink
                tau_sec = tau_eff.unsqueeze(-1) * 3600.0
                react_term = C_hat / torch.clamp(tau_sec, min=3600.0)

                # 4. Total Residual
                dC_dt_sec = dC_dt / 3600.0
                pde_residual = dC_dt_sec + adv_term - diff_term - (S_hat / 3600.0) + react_term

                return torch.mean(pde_residual**2) * self.scale_loss

        return ADR_PDE_Loss
    except ImportError as err:
        raise ImportError(
            "PyTorch required for ADR_PDE_Loss. Install with `pip install torch`."
        ) from err


def build_adr_pde_loss(*args: Any, **kwargs: Any) -> Any:
    """Construct an ADR-PDE loss module, importing PyTorch on first use.

    See :func:`normet.physics.stgnn.build_pi_stgnn` for why this is a function
    and not a class. :func:`get_adr_pde_loss_class` returns the real
    ``nn.Module`` subclass when you need the type.
    """
    return get_adr_pde_loss_class()(*args, **kwargs)
