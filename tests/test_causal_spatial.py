"""Spatial diagnostics for a donor pool.

The haversine and bearing primitives are checked against values a spherical
Earth has to give (a meridian degree, a parallel degree, the quarter and
half circumference). The summary is checked on a constructed triangle of
donors -- one due north, one due east, one due south of the treated unit --
where every distance, bearing, sector and the centroid offset is known by
construction, plus the honest treatment of a donor whose fix is missing.
"""

import numpy as np
import pandas as pd

from normet import bearing, donor_spatial_summary, haversine, spatial_summary
from normet.causal import scm_diagnostics
from normet.causal.scm import scm

# A degree of meridian and, at 51.5 degrees north, a degree of parallel, on
# the R = 6371 km sphere the primitives use.
KM_PER_DEG_N = np.pi / 180.0 * 6371.0
KM_PER_DEG_E = np.pi / 180.0 * 6371.0 * np.cos(np.radians(51.5))

CUTOFF = "2023-05-01"
TREATED = "T"


def _triangle_panel() -> pd.DataFrame:
    """T at the origin of the layout; N due north, S due south, and E placed so
    its forward azimuth lies mid-sector (~108 degrees, in [90, 135)): a
    same-latitude east leg's initial bearing is a fraction of a degree *north*
    of 90, which would fall in the NE bin. GHOST is named but unfixed."""
    fixes = {
        "T": (51.5, 0.0),
        "N": (52.0, 0.0),
        "E": (51.4, 0.5),
        "S": (51.0, 0.0),
    }
    rows = []
    for date in pd.date_range("2023-01-01", periods=3, freq="MS"):
        for unit, (lat, lon) in fixes.items():
            rows.append({"date": date, "ID": unit, "value": 1.0, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def _triangle_table() -> pd.DataFrame:
    return donor_spatial_summary(
        _triangle_panel(), ["N", "E", "S", "GHOST"], TREATED, unit_col="ID"
    )


# ------------------------------------------------------------------ primitives
def test_haversine_coincides_with_itself():
    assert haversine(51.5, -0.1, 51.5, -0.1) == 0.0


def test_haversine_quarter_meridian_is_a_quarter_circumference():
    assert np.isclose(haversine(0.0, 0.0, 90.0, 0.0), np.pi / 2.0 * 6371.0, rtol=1e-9)


def test_haversine_antipodes_are_a_half_circumference():
    assert np.isclose(haversine(0.0, 0.0, 0.0, 180.0), np.pi * 6371.0, rtol=1e-9)


def test_haversine_one_degree_of_parallel_at_51_5_north():
    # The great circle, not the parallel's arc: the reference is the spherical
    # law of cosines, which the haversine must agree with to rounding.
    phi, dlam = np.radians(51.5), np.radians(1.0)
    sigma = np.arccos(np.sin(phi) ** 2 + np.cos(phi) ** 2 * np.cos(dlam))
    assert np.isclose(haversine(51.5, 0.0, 51.5, 1.0), 6371.0 * sigma, rtol=1e-9)
    # And within the parallel's own arc length, which a degree of parallel
    # is to ~0.001 %: the flat-earth reading is still the right magnitude.
    assert np.isclose(haversine(51.5, 0.0, 51.5, 1.0), KM_PER_DEG_E, rtol=1e-4)


def test_haversine_is_symmetric():
    a, b = haversine(48.8, 2.3, 51.5, -0.1), haversine(51.5, -0.1, 48.8, 2.3)
    assert np.isclose(a, b, rtol=1e-12)


def test_haversine_is_elementwise():
    d = haversine(np.array([51.5, 51.5]), np.array([0.0, 0.0]), 52.0, 0.0)
    assert d.shape == (2,)
    assert np.allclose(d, 0.5 * KM_PER_DEG_N)


def test_bearing_dues_on_the_compass():
    # Meridians and the equator are great circles, so the dues there are exact;
    # a same-latitude leg away from the equator curves, and its *initial*
    # bearing is a fraction of a degree off the parallel's 90.
    assert np.isclose(bearing(51.5, 0.0, 52.5, 0.0), 0.0, atol=1e-9)
    assert np.isclose(bearing(51.5, 0.0, 50.5, 0.0), 180.0, atol=1e-9)
    assert np.isclose(bearing(0.0, 0.0, 0.0, 1.0), 90.0, atol=1e-9)
    assert np.isclose(bearing(0.0, 0.0, 0.0, -1.0), 270.0, atol=1e-9)


def test_bearing_diagonal_is_the_forward_azimuth():
    # From the equator to (45, 45): y = sin(45)cos(45), x = sin(45), so the
    # forward azimuth is arctan(cos 45) -- a fact the great circle keeps.
    assert np.isclose(
        bearing(0.0, 0.0, 45.0, 45.0),
        np.degrees(np.arctan2(np.cos(np.radians(45.0)), 1.0)),
        atol=1e-9,
    )


def test_bearing_is_always_in_0_360_and_elementwise():
    b = bearing(np.array([51.5, 51.5]), np.array([0.0, 0.0]), 50.5, -1.0)
    assert b.shape == (2,)
    assert np.all((b >= 0.0) & (b < 360.0))


# ------------------------------------------------------------- donor summary
def test_donor_summary_places_the_triangle_and_keeps_the_missing_one():
    table = _triangle_table()

    assert list(table.index) == ["N", "E", "S", "GHOST"]
    assert list(table.columns) == ["distance_km", "bearing_deg", "sector"]
    # N and S sit on the treated unit's meridian, so their legs are exact; the
    # east leg is checked against the primitive itself, and on its magnitude.
    assert np.isclose(table.loc["N", "distance_km"], 0.5 * KM_PER_DEG_N, rtol=1e-9)
    assert np.isclose(table.loc["E", "distance_km"], haversine(51.5, 0.0, 51.4, 0.5), rtol=1e-9)
    assert 34.0 < table.loc["E", "distance_km"] < 38.0
    assert np.isclose(table.loc["S", "distance_km"], 0.5 * KM_PER_DEG_N, rtol=1e-9)
    assert np.isclose(table.loc["N", "bearing_deg"], 0.0, atol=1e-9)
    assert 100.0 < table.loc["E", "bearing_deg"] < 115.0
    assert np.isclose(table.loc["S", "bearing_deg"], 180.0, atol=1e-9)
    assert (table.loc[["N", "E", "S"], "sector"] == ["N", "E", "S"]).all()
    # The unfixed donor is read as missing, not dropped: the table is a
    # reading of the pool as named.
    assert np.isnan(table.loc["GHOST", "distance_km"])
    assert np.isnan(table.loc["GHOST", "bearing_deg"])
    assert table.loc["GHOST", "sector"] == "?"


def test_donor_summary_reads_one_fix_per_unit_as_the_median():
    df = _triangle_panel()
    # One drifting fix on N's rows must not move the station: the median of
    # the unit's fixes is its position.
    df.loc[(df["ID"] == "N") & (df["date"] == df["date"].iloc[0]), "lat"] += 1e-6
    assert np.isclose(
        donor_spatial_summary(df, ["N"], TREATED, unit_col="ID").loc["N", "distance_km"],
        0.5 * KM_PER_DEG_N,
        rtol=1e-9,
    )


def test_donor_summary_carries_the_weights_when_given():
    w = pd.Series({"N": 0.5, "E": 0.5})
    table = donor_spatial_summary(
        _triangle_panel(), ["N", "E", "S"], TREATED, unit_col="ID", weights=w
    )
    assert np.isclose(table.loc["N", "weight"], 0.5)
    assert np.isclose(table.loc["E", "weight"], 0.5)
    assert np.isnan(table.loc["S", "weight"])
    assert "weight" not in _triangle_table().columns


def test_donor_summary_raises_when_the_treated_unit_is_unfixed():
    panel = _triangle_panel()
    df = panel.drop(panel.index[panel["ID"] == TREATED])
    try:
        donor_spatial_summary(df, ["N"], TREATED, unit_col="ID")
        raise AssertionError("expected a ValueError for an unfixed treated unit")
    except ValueError:
        pass


def test_donor_summary_raises_when_the_coordinate_columns_are_absent():
    try:
        donor_spatial_summary(
            _triangle_panel().drop(columns=["lon"]), ["N"], TREATED, unit_col="ID"
        )
        raise AssertionError("expected a ValueError for an absent coordinate column")
    except ValueError:
        pass


# ------------------------------------------------------------ spatial summary
def test_spatial_summary_centroid_of_a_triangle_is_the_mean_offset():
    table = _triangle_table()
    s = spatial_summary(table, n_donors=4)

    # The centroid as an averaged (east, north) offset vector of the placed
    # donors, read back from the table's own distances and bearings.
    placed = table.dropna(subset=["distance_km"])
    rad = np.deg2rad(placed["bearing_deg"].to_numpy(dtype=float))
    d = placed["distance_km"].to_numpy(dtype=float)
    exp = np.hypot(np.mean(d * np.sin(rad)), np.mean(d * np.cos(rad)))
    assert np.isclose(s["centroid_offset_km"], float(exp), rtol=1e-12)
    assert 100.0 < s["centroid_bearing_deg"] < 115.0
    assert s["sector_coverage"] == [("N", 1), ("E", 1), ("S", 1)]
    assert np.isclose(s["sector_fraction"], 3 / 8.0)
    assert s["n_donors_placed"] == 3
    assert s["n_donors_missing"] == 1


def test_spatial_summary_of_a_single_donor_is_that_donor():
    s = spatial_summary(_triangle_table().loc[["N"]])
    assert np.isclose(s["centroid_offset_km"], 0.5 * KM_PER_DEG_N, rtol=1e-9)
    assert np.isclose(s["centroid_bearing_deg"], 0.0, atol=1e-9)
    assert s["sector_coverage"] == [("N", 1)]


def test_spatial_summary_of_an_unplaced_pool_is_all_missing():
    table = _triangle_table().loc[["GHOST"]]
    s = spatial_summary(table, n_donors=1)
    assert np.isnan(s["centroid_offset_km"])
    assert np.isnan(s["centroid_bearing_deg"])
    assert s["sector_coverage"] == []
    assert s["sector_fraction"] == 0.0
    assert s["n_donors_placed"] == 0
    assert s["n_donors_missing"] == 1


def test_sector_labels_follow_the_bearing_not_the_alphabet():
    # W (270) must be listed after E (90) in bearing order, not by name.
    table = _triangle_table().loc[["E", "S"]]
    s = spatial_summary(table)
    assert [name for name, _ in s["sector_coverage"]] == ["E", "S"]


# ---------------------------------------------------- scm_diagnostics wiring
def _fit_with_coordinates() -> tuple[dict, pd.DataFrame]:
    # 120 pre-cutoff days, 40 post, per unit, with the triangle's coordinates.
    dates = pd.date_range("2023-01-01", periods=160, freq="D")
    rows = []
    for i, date in enumerate(dates):
        for unit, (lat, lon) in (
            ("T", (51.5, 0.0)),
            ("N", (52.0, 0.0)),
            ("E", (51.5, 0.5)),
            ("S", (51.0, 0.0)),
        ):
            v = 10.0 + 0.1 * i + (10.0 if (unit == "T" and date >= pd.Timestamp(CUTOFF)) else 0.0)
            rows.append({"date": date, "ID": unit, "value": v, "lat": lat, "lon": lon})
    panel = pd.DataFrame(rows)
    result = scm(
        df=panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit=TREATED,
        cutoff_date=CUTOFF,
        donors=["N", "E", "S"],
    )
    return result, panel


def test_scm_diagnostics_appends_the_spatial_section_when_the_panel_has_fixes():
    result, panel = _fit_with_coordinates()
    out = scm_diagnostics(
        result,
        CUTOFF,
        panel=panel,
        treated_unit=TREATED,
        unit_col="ID",
        lat_col="lat",
        lon_col="lon",
    )
    assert "spatial" in out
    assert list(out["spatial"]["donors"].index) == ["N", "E", "S"]
    # Donors default to the fit's weights' index, so the weight column is there.
    assert "weight" in out["spatial"]["donors"].columns
    assert out["spatial"]["n_donors_placed"] == 3
    assert out["spatial"]["n_donors_missing"] == 0
    # The section agrees with calling the summary directly.
    direct = donor_spatial_summary(
        panel, ["N", "E", "S"], TREATED, unit_col="ID", weights=result["weights"]
    )
    pd.testing.assert_frame_equal(out["spatial"]["donors"], direct)


def test_scm_diagnostics_without_a_panel_is_what_it_was_before():
    result, panel = _fit_with_coordinates()
    out = scm_diagnostics(result, CUTOFF)
    assert "spatial" not in out
    # The panel carries fixes but no treated unit was named: still no section.
    out = scm_diagnostics(result, CUTOFF, panel=panel, unit_col="ID", lat_col="lat", lon_col="lon")
    assert "spatial" not in out
