"""Dynamic Wind-Guided Dual-Graph Topology Construction for Atmospheric Networks.

Constructs static geospatial graphs A_geo and dynamic directed advection graphs
A_wind(t) for irregular air quality monitoring networks.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def haversine_pairwise(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Compute pairwise Great Circle distance in km between coordinate arrays."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def bearing_pairwise(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Compute azimuth bearing in radians (0 = North, pi/2 = East) from station 1 to station 2."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    theta = np.arctan2(y, x)
    return (theta + 2.0 * np.pi) % (2.0 * np.pi)


class PhysicsGraphBuilder:
    """Dual-Graph Topology Builder for Spatio-Temporal Graph Neural Networks."""

    def __init__(
        self,
        stations: list[str],
        latitudes: np.ndarray | list[float],
        longitudes: np.ndarray | list[float],
        cities: list[str] | None = None,
        k_nn: int = 8,
        sigma_dist_km: float = 45.0,
        max_dist_cutoff_km: float = 120.0,
    ) -> None:
        self.stations = list(stations)
        self.n_nodes = len(self.stations)
        self.lats = np.asarray(latitudes, dtype=np.float64)
        self.lons = np.asarray(longitudes, dtype=np.float64)
        self.cities = list(cities) if cities is not None else ["" for _ in self.stations]

        # Mismatched lengths used to surface as an IndexError from deep inside
        # build_static_graph, long after the coordinates that caused it.
        for name, got in (
            ("latitudes", self.lats.shape[0]),
            ("longitudes", self.lons.shape[0]),
            ("cities", len(self.cities)),
        ):
            if got != self.n_nodes:
                raise ValueError(f"{name} has {got} entries but {self.n_nodes} stations were given")
        self.k_nn = k_nn
        self.sigma_dist_km = sigma_dist_km
        self.max_dist_cutoff_km = max_dist_cutoff_km

        lat_i, lat_j = np.meshgrid(self.lats, self.lats, indexing="ij")
        lon_i, lon_j = np.meshgrid(self.lons, self.lons, indexing="ij")
        self.dist_matrix = haversine_pairwise(lat_i, lon_i, lat_j, lon_j)
        self.bearing_matrix = bearing_pairwise(lat_i, lon_i, lat_j, lon_j)

    def build_static_graph(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct sparse symmetric static spatial graph A_geo with k-NN and cluster connectivity.

        Returns:
            A_geo: [N, N] spatial adjacency matrix.
            edge_index: [2, E] edge indices.
            edge_weight: [E] edge weights.
        """
        A_dist = np.exp(-(self.dist_matrix**2) / (2.0 * (self.sigma_dist_km**2)))
        np.fill_diagonal(A_dist, 0.0)

        # 1. k-NN Sparsification
        A_knn = np.zeros_like(A_dist)
        for i in range(self.n_nodes):
            sorted_indices = np.argsort(self.dist_matrix[i])
            knn_indices = [
                idx
                for idx in sorted_indices
                if idx != i and self.dist_matrix[i, idx] <= self.max_dist_cutoff_km
            ][: self.k_nn]
            A_knn[i, knn_indices] = A_dist[i, knn_indices]

        # Symmetrize
        A_geo = np.maximum(A_knn, A_knn.T)

        # 2. Intracluster connectivity enhancement
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if (
                    self.cities[i]
                    and self.cities[i] == self.cities[j]
                    and self.dist_matrix[i, j] <= 50.0
                ):
                    boosted = max(
                        A_geo[i, j], np.exp(-(self.dist_matrix[i, j] ** 2) / (2.0 * (20.0**2)))
                    )
                    A_geo[i, j] = boosted
                    A_geo[j, i] = boosted

        src, dst = np.where(A_geo > 0.01)
        edge_index = np.vstack([src, dst])
        edge_weight = A_geo[src, dst]

        return A_geo, edge_index, edge_weight

    def build_wind_graph(
        self,
        u10: np.ndarray,
        v10: np.ndarray,
        ws: np.ndarray,
        tau_adv_hours: float = 3.0,
        alpha: float = 2.0,
    ) -> np.ndarray:
        """Construct dynamic wind-guided directed advection graph A_wind(t).

        Args:
            u10: [N] eastward wind component at each station (m/s).
            v10: [N] northward wind component at each station (m/s).
            ws: [N] wind speed at each station (m/s).
            tau_adv_hours: Advection characteristic time scale (default 3.0h).
            alpha: Directional alignment power exponent.

        Returns:
            A_wind: [N, N] directed advection adjacency matrix.
        """
        phi_wind = (np.arctan2(u10, v10) + 2.0 * np.pi) % (2.0 * np.pi)
        delta_angle = self.bearing_matrix - phi_wind[:, None]

        # Alignment cosine (1 if downwind, 0 if perpendicular or upwind)
        alignment = np.maximum(0.0, np.cos(delta_angle)) ** alpha
        sigma_adv = np.maximum(15.0, ws[:, None] * (tau_adv_hours * 3.6) + 10.0)

        A_wind = np.exp(-(self.dist_matrix**2) / (2.0 * (sigma_adv**2))) * alignment
        np.fill_diagonal(A_wind, 0.0)
        A_wind[self.dist_matrix > self.max_dist_cutoff_km] = 0.0

        return A_wind
