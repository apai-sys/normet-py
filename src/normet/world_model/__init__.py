"""World-Model atmospheric counterfactual modeling subpackage.

Combines Action-Conditioned JEPA latent dynamics & causal synthetic control
(:mod:`~normet.world_model.jepa`), bijective separable latent representation
learning (:mod:`~normet.world_model.latent_wm`), hierarchical multi-site network
decomposition (:mod:`~normet.world_model.multisite`), diagnostic plots
(:mod:`~normet.world_model.plotting`), and ``F1``–``F6`` ablation harnesses
(:mod:`~normet.world_model.benchmarks`).
"""

from __future__ import annotations

from .benchmarks import benchmark_interaction_sweep, benchmark_world_models
from .jepa import (
    JEPAResult,
    JEPASCMResult,
    LatentJEPA,
    detect_jepa_events,
    fit_latent_jepa,
    jepa_anomaly_scores,
    jepa_scm,
    jepa_select_donors,
)
from .latent_wm import (
    SeparableWorldModel,
    TrajectoryRolloutResult,
    WorldModelConfig,
    WorldModelNormaliseResult,
    fit_world_model,
)
from .multisite import (
    MultisiteDecompositionResult,
    MultisiteWorldModel,
    fit_multisite_world_model,
)
from .plotting import (
    plot_jepa_anomalies,
    plot_rollout,
    plot_separability_landscape,
)

__all__ = [
    "JEPAResult",
    "JEPASCMResult",
    "LatentJEPA",
    "fit_latent_jepa",
    "jepa_anomaly_scores",
    "detect_jepa_events",
    "jepa_select_donors",
    "jepa_scm",
    "WorldModelConfig",
    "WorldModelNormaliseResult",
    "TrajectoryRolloutResult",
    "SeparableWorldModel",
    "fit_world_model",
    "MultisiteDecompositionResult",
    "MultisiteWorldModel",
    "fit_multisite_world_model",
    "plot_separability_landscape",
    "plot_rollout",
    "plot_jepa_anomalies",
    "benchmark_world_models",
    "benchmark_interaction_sweep",
]
