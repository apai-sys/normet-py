"""Tests for the dual-graph builder and the lazily-built torch modules.

The torch pieces are gated on ``torch`` being installed. The graph builder is
pure numpy and always runs.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from normet.physics import (
    PhysicsGraphBuilder,
    build_adr_pde_loss,
    build_pi_stgnn,
    get_adr_pde_loss_class,
    get_pi_stgnn_class,
)
from normet.physics.graph import bearing_pairwise, haversine_pairwise


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


needs_torch = pytest.mark.skipif(not _has("torch"), reason="torch not installed")


# ------------------------------------------------------------------- geometry


def test_haversine_matches_a_known_distance():
    """London to Manchester is about 262 km great-circle."""
    d = haversine_pairwise(
        np.array([51.5074]), np.array([-0.1278]), np.array([53.4808]), np.array([-2.2426])
    )
    assert d[0] == pytest.approx(262.0, abs=5.0)


def test_bearing_is_east_for_a_due_east_neighbour():
    b = bearing_pairwise(np.array([51.5]), np.array([-0.30]), np.array([51.5]), np.array([-0.10]))
    assert np.degrees(b[0]) == pytest.approx(90.0, abs=0.5)


# -------------------------------------------------------------------- builder


def test_builder_rejects_mismatched_coordinate_lengths():
    """Used to surface as an IndexError from deep inside build_static_graph."""
    with pytest.raises(ValueError, match="latitudes has 2 entries but 3 stations"):
        PhysicsGraphBuilder(
            stations=["a", "b", "c"], latitudes=[51.5, 53.5], longitudes=[-0.1, -2.2]
        )
    with pytest.raises(ValueError, match="cities has 1 entries but 2 stations"):
        PhysicsGraphBuilder(
            stations=["a", "b"], latitudes=[51.5, 53.5], longitudes=[-0.1, -2.2], cities=["X"]
        )


def test_static_graph_is_symmetric_and_hollow():
    rng = np.random.default_rng(0)
    n = 12
    b = PhysicsGraphBuilder(
        stations=[f"s{i}" for i in range(n)],
        latitudes=51.0 + rng.uniform(0, 1.5, n),
        longitudes=-1.0 + rng.uniform(0, 1.5, n),
        k_nn=3,
    )
    A, edge_index, edge_weight = b.build_static_graph()

    assert A.shape == (n, n)
    np.testing.assert_allclose(A, A.T)
    np.testing.assert_allclose(np.diag(A), 0.0)
    assert edge_index.shape == (2, edge_weight.size)
    assert (edge_weight > 0.01).all()


def test_static_graph_respects_the_distance_cutoff():
    """Two stations 260 km apart, with a 120 km cutoff, must not be connected."""
    b = PhysicsGraphBuilder(
        stations=["LON", "MAN"],
        latitudes=[51.5074, 53.4808],
        longitudes=[-0.1278, -2.2426],
        max_dist_cutoff_km=120.0,
    )
    A, _, _ = b.build_static_graph()
    assert A[0, 1] == 0.0


def test_wind_graph_points_downwind():
    """A westerly wind must weight the eastern neighbour, not the western one."""
    b = PhysicsGraphBuilder(stations=["W", "E"], latitudes=[51.5, 51.5], longitudes=[-0.30, -0.10])
    # u = +5 m/s eastward, v = 0: the plume travels W -> E.
    A_wind = b.build_wind_graph(
        u10=np.array([5.0, 5.0]), v10=np.array([0.0, 0.0]), ws=np.array([5.0, 5.0])
    )

    assert A_wind[0, 1] > 0.5
    assert A_wind[1, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(np.diag(A_wind), 0.0)


def test_wind_graph_is_calm_when_the_wind_reverses():
    b = PhysicsGraphBuilder(stations=["W", "E"], latitudes=[51.5, 51.5], longitudes=[-0.30, -0.10])
    easterly = b.build_wind_graph(
        u10=np.array([-5.0, -5.0]), v10=np.array([0.0, 0.0]), ws=np.array([5.0, 5.0])
    )
    assert easterly[1, 0] > 0.5
    assert easterly[0, 1] == pytest.approx(0.0)


# ------------------------------------------------------------ lazy torch classes


@needs_torch
def test_lazy_class_lookup_is_stable():
    """The class must be built once, or ``isinstance`` against it is meaningless.

    ``get_*_class`` defines the class inside the function body, so before it was
    memoised every call produced a fresh ``type`` and
    ``isinstance(build_pi_stgnn(...), get_pi_stgnn_class())`` was ``False``.
    """
    assert get_pi_stgnn_class() is get_pi_stgnn_class()
    assert get_adr_pde_loss_class() is get_adr_pde_loss_class()


@needs_torch
def test_built_objects_are_instances_of_the_advertised_class():
    """The old module-level ``PI_STGNN`` wrapper returned something it was not."""
    import torch

    model = build_pi_stgnn(in_dim=8, hidden_dim=16, n_nodes=4, n_layers=1)
    loss = build_adr_pde_loss()

    assert isinstance(model, get_pi_stgnn_class())
    assert isinstance(loss, get_adr_pde_loss_class())
    assert isinstance(model, torch.nn.Module)
    assert isinstance(loss, torch.nn.Module)


@needs_torch
def test_stgnn_forward_and_backward():
    import torch

    torch.manual_seed(0)
    n_batch, n_nodes, n_times, n_feat = 2, 4, 6, 8
    model = build_pi_stgnn(in_dim=n_feat, hidden_dim=16, n_nodes=n_nodes, n_layers=2)

    x = torch.randn(n_batch, n_nodes, n_times, n_feat)
    adjacency = torch.softmax(torch.randn(n_nodes, n_nodes), dim=-1)
    conc, emission, tau = model(x, adjacency, adjacency)

    assert conc.shape == (n_batch, n_nodes, n_times)
    assert emission.shape == (n_batch, n_nodes, n_times)
    assert tau.shape == (n_batch, n_nodes)
    # Softplus heads are non-negative; the lifetime head is 1 + 23 * sigmoid.
    assert (conc >= 0).all() and (emission >= 0).all()
    assert ((tau >= 1.0) & (tau <= 24.0)).all()

    (conc.mean() + emission.mean() + tau.mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@needs_torch
def test_pde_loss_is_zero_for_a_steady_empty_field():
    """No concentration, no emission, no tendency -> no residual."""
    import torch

    n_batch, n_nodes, n_times = 1, 3, 4
    loss = build_adr_pde_loss()
    zeros = torch.zeros(n_batch, n_nodes, n_times)
    value = loss(
        C_hat=zeros,
        S_hat=zeros,
        tau_eff=torch.full((n_batch, n_nodes), 10.0),
        dC_dt=zeros,
        dist_matrix=torch.full((n_nodes, n_nodes), 10.0),
        A_geo=torch.zeros(n_nodes, n_nodes),
        A_wind=torch.zeros(n_nodes, n_nodes),
        u10=torch.zeros(n_nodes),
        v10=torch.zeros(n_nodes),
        ws=torch.full((n_batch, n_nodes, n_times), 3.0),
        blh=torch.full((n_batch, n_nodes, n_times), 500.0),
    )
    assert float(value) == pytest.approx(0.0)


@needs_torch
def test_pde_advection_reaches_the_downwind_receptor():
    """Under the default ``upwind`` convention, the receptor feels the plume.

    ``A_wind[i, j]`` weights ``j`` downwind of ``i``, so summing over ``j``
    without transposing charges the advection to the *source* and leaves the
    downwind receptor with exactly zero -- which is what the loss used to do.
    """
    import torch

    builder = PhysicsGraphBuilder(
        stations=["W", "E"], latitudes=[51.5, 51.5], longitudes=[-0.30, -0.10]
    )
    A_wind = torch.tensor(
        builder.build_wind_graph(
            u10=np.array([5.0, 5.0]), v10=np.array([0.0, 0.0]), ws=np.array([5.0, 5.0])
        ),
        dtype=torch.float32,
    )
    dist = torch.tensor(builder.dist_matrix, dtype=torch.float32)

    n_batch, n_nodes, n_times = 1, 2, 2
    concentration = torch.zeros(n_batch, n_nodes, n_times)
    concentration[:, 0, :] = 100.0  # the whole plume sits at the western station

    def advection_at(direction: str) -> torch.Tensor:
        loss = get_adr_pde_loss_class()(direction=direction)
        A_dir = A_wind.transpose(-2, -1) if loss.direction == "upwind" else A_wind
        operator = A_dir.unsqueeze(0) / (torch.clamp(dist * 1000.0, min=500.0) + 1e-5)
        column = concentration[:, :, 0].unsqueeze(-1)
        return torch.sum(operator * (column - column.transpose(1, 2)), dim=-1) * 5.0

    upwind = advection_at("upwind")
    assert float(upwind[0, 1]) != pytest.approx(0.0)  # receptor transports
    assert float(upwind[0, 0]) == pytest.approx(0.0)  # source does not

    outflow = advection_at("outflow")
    assert float(outflow[0, 0]) != pytest.approx(0.0)
    assert float(outflow[0, 1]) == pytest.approx(0.0)


@needs_torch
def test_pde_loss_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="must be 'upwind' or 'outflow'"):
        build_adr_pde_loss(direction="crosswind")


@needs_torch
def test_pde_loss_penalises_an_unexplained_tendency():
    """A concentration rising with no source and no transport must cost something."""
    import torch

    n_batch, n_nodes, n_times = 1, 3, 4
    loss = build_adr_pde_loss()
    common = dict(
        tau_eff=torch.full((n_batch, n_nodes), 10.0),
        dist_matrix=torch.full((n_nodes, n_nodes), 10.0),
        A_geo=torch.zeros(n_nodes, n_nodes),
        A_wind=torch.zeros(n_nodes, n_nodes),
        u10=torch.zeros(n_nodes),
        v10=torch.zeros(n_nodes),
        ws=torch.full((n_batch, n_nodes, n_times), 3.0),
        blh=torch.full((n_batch, n_nodes, n_times), 500.0),
    )
    value = loss(
        C_hat=torch.full((n_batch, n_nodes, n_times), 20.0),
        S_hat=torch.zeros(n_batch, n_nodes, n_times),
        dC_dt=torch.full((n_batch, n_nodes, n_times), 50.0),
        **common,
    )
    assert float(value) > 0.0
