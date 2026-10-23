"""Data Studio — fetch air-quality + meteorology and build a model-ready table.

Two-step workflow in the house style (parameter panel left, result tabs right):

1. **Find stations** — pick a network, tick pollutants, list the sites that
   measure them, browse/search the station table and pick one.
2. **Fetch & merge** — download the hourly measurements for every ticked
   pollutant at that site, fetch hourly meteorology for the site coordinates
   (Open-Meteo ERA5 archive by default — no API key; Copernicus CDS
   optionally), and outer-join everything on the hourly timestamp into the
   wide ``date + pollutants + met`` table the modelling steps expect.

The merged table can be saved as CSV or sent straight to the main window.

Data sources
------------
Three families behind one network picker:

- the six **UK networks** (AURN, AQE, SAQN, WAQN, NI, LMAM) via
  :mod:`normet.io.ukaq` — openair ``.RData`` archives, ~1500 stations, whole
  calendar years; no key.
- the **EEA** (European Environment Agency) reporting network via
  :mod:`normet.io.eea` — the ArcGIS station layer for listing and the
  agency's public per-station Parquet blobs for the series; no key.
- **OpenAQ** via :mod:`normet.io.openaq` — a free API key (``OPENAQ_API_KEY``
  or the field in the panel), any country it hosts.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from normet import style as S

from ._clock import TaskClock
from ._widgets import CanvasTab, NoWheelComboBox, hint_label, run_button
from .workers import TaskRunner

log = logging.getLogger(__name__)

# Column names as they appear in BOTH the archives and the metadata
# ``variable`` column — the two agree exactly, so no name mapping is needed.
# NOx is "NOXasNO2" here, not "NOX": that is the archive's own name for it
# and what the merged table's column will be called.
POLLUTANTS = ["PM2.5", "PM10", "NO2", "NOXasNO2", "NO", "O3", "SO2", "CO"]
DEFAULT_POLLUTANTS = {"PM2.5", "NO2", "O3"}

# The six UK networks: picker label -> normet.io.ukaq source key. AURN first
# so it stays the default.
NETWORKS: dict[str, str] = {
    "AURN — UK national network": "aurn",
    "AQE — Air Quality England": "aqe",
    "SAQN — Scotland": "saqn",
    "WAQN — Wales": "waqn",
    "NI — Northern Ireland": "ni",
    "LMAM — DEFRA locally managed": "local",
}

# The non-UK networks: one entry per family, picked the same way in the
# Network combo. Each carries a provider and any argument the picker needs.
EEA_NETWORK = "EEA — European Environment Agency (all of Europe)"
OPENAQ_NETWORK = "OpenAQ (most of the world, needs a free key)"

# Provider keys: dispatch the find/fetch to the right adapter module.
PROV_UK, PROV_EEA, PROV_OPENAQ = "uk", "eea", "openaq"

# normet notation -> OpenAQ v3 parameter slug. Only the parameters the
# adapter's `parameters_id` table can resolve — OpenAQ v3 does not serve
# NO or NOx separately, so those tick them off in the other networks only.
_OPENAQ_SLUG = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "O3": "o3",
    "SO2": "so2",
    "CO": "co",
}

MET_OPEN_METEO = "Open-Meteo (ERA5, no key needed)"
MET_CDS = "Copernicus CDS (needs ~/.cdsapirc)"


def _network_meta(text: str) -> tuple[str, str | None]:
    """(provider, ukaq source) for a Network combo text; source is UK-only."""
    if text in NETWORKS:
        return PROV_UK, NETWORKS[text]
    if text == EEA_NETWORK:
        return PROV_EEA, None
    if text == OPENAQ_NETWORK:
        return PROV_OPENAQ, None
    raise ValueError(f"Unknown network {text!r}")


def _find_stations(
    provider: str, source: str | None, country: str, pollutants: list[str], api_key: str | None
) -> pd.DataFrame:
    """One row per station measuring any of *pollutants*, for *provider*.

    Columns (identical across providers): ``site``, ``code``, ``site_type``,
    ``pollutants`` (which of the ticked ones it reports), ``n``, ``from``/
    ``to`` (the covered period; blank when the source does not state one),
    ``lat``, ``lon``. The picker and the fetch both read off these columns.
    """
    if provider == PROV_UK:
        return _find_stations_uk(pollutants, source or "aurn")
    if provider == PROV_EEA:
        return _find_stations_eea(country, pollutants)
    if provider == PROV_OPENAQ:
        return _find_stations_openaq(country, pollutants, api_key)
    raise ValueError(f"Unknown provider {provider!r}")


def _find_stations_uk(pollutants: list[str], source: str) -> pd.DataFrame:
    """One row per station in *source* measuring any of *pollutants*.

    Columns: ``site``, ``code``, ``site_type``, ``pollutants`` (which of the
    ticked ones it reports), ``from``/``to`` (the period the archive covers,
    from the network metadata), ``lat``, ``lon``.

    ``end_date`` is blank in the metadata for a station still operating, so
    a missing value is shown as today rather than as an empty cell — an
    empty "to" would otherwise read as "no data".
    """
    from normet import list_ukaq_stations

    meta = list_ukaq_stations(source, all_variables=True)
    if meta.empty:
        return pd.DataFrame(
            columns=["site", "code", "site_type", "pollutants", "n", "from", "to", "lat", "lon"]
        )

    wanted = set(pollutants)
    sub = meta[meta["variable"].astype(str).isin(wanted)]
    log.info("%s: %d station-species rows match %s", source.upper(), len(sub), sorted(wanted))
    if sub.empty:
        return pd.DataFrame(
            columns=["site", "code", "site_type", "pollutants", "n", "from", "to", "lat", "lon"]
        )

    # Parse the two date columns once for the whole frame, not per group:
    # doing it inside the loop made pandas re-infer the format on every one
    # of ~1500 groups and emit a UserWarning each time. The metadata uses
    # ISO dates, so the format is stated rather than guessed.
    sub = sub.copy()
    for col in ("start_date", "end_date"):
        sub[col] = (
            pd.to_datetime(sub[col], format="%Y-%m-%d", errors="coerce")
            if col in sub.columns
            else pd.NaT
        )

    today = pd.Timestamp.today().normalize()
    rows = []
    for code, g in sub.groupby("code", sort=False):
        have = [p for p in pollutants if p in set(g["variable"].astype(str))]
        starts = g["start_date"]
        ends = g["end_date"]
        t0 = starts.min() if starts.notna().any() else pd.NaT
        # Any still-open series means the station is still reporting.
        t1 = today if ends.isna().any() else ends.max()
        rows.append(
            {
                "site": str(g["site"].iloc[0]),
                "code": str(code),
                "site_type": str(g["site_type"].iloc[0]) if "site_type" in g else "",
                "pollutants": ", ".join(have),
                "n": len(have),
                "from": t0.strftime("%Y-%m-%d") if pd.notna(t0) else "",
                "to": t1.strftime("%Y-%m-%d") if pd.notna(t1) else "",
                "lat": g["latitude"].iloc[0] if "latitude" in g else None,
                "lon": g["longitude"].iloc[0] if "longitude" in g else None,
            }
        )
    df = pd.DataFrame(rows).sort_values(["n", "site"], ascending=[False, True])
    return df.reset_index(drop=True)


def _find_stations_eea(country: str, pollutants: list[str]) -> pd.DataFrame:
    """EEA stations from the ArcGIS layer, in the shared picker columns.

    The per-pollutant ``from``/``to`` coverage is not per-station but is what
    the layer's popup states for each pollutant, so the picker shows the
    union across the ticked pollutants it reports.
    """
    from normet.io.eea import list_eea_stations

    df = list_eea_stations(country=country, pollutants=pollutants)
    if df.empty:
        return df
    import json

    spans = df["links"].map(
        lambda s: {k: v for k, v in json.loads(s).items() if k in set(pollutants)}
    )
    lo = spans.map(lambda m: min((v["from"] for v in m.values()), default=None))
    hi = spans.map(lambda m: max((v["to"] for v in m.values()), default=None))
    # The layer dates its blobs by year; the full dates let the picker reuse
    # its usual "from–to" snap logic unchanged.
    df["from"] = [f"{y:04d}-01-01" if y is not None else "" for y in lo]
    df["to"] = [f"{y:04d}-12-31" if y is not None else "" for y in hi]
    return df.drop(columns=["links"])


def _find_stations_openaq(country: str, pollutants: list[str], api_key: str | None) -> pd.DataFrame:
    """OpenAQ locations for *country*, filtered to the ticked parameters.

    The ticked slugs go to the server as a union filter, so a big country is
    narrowed before paging rather than fetched whole and cut down client-side.
    """
    from normet.io.openaq import openaq_locations

    slugs = [s for s in (_OPENAQ_SLUG.get(p) for p in pollutants) if s]
    unserved = [p for p in pollutants if p not in _OPENAQ_SLUG]
    if unserved:
        log.info("OpenAQ does not serve %s — skipped", ", ".join(unserved))
    locs = (
        openaq_locations(country=country or None, parameters=slugs, limit=5000, api_key=api_key)
        if slugs
        else pd.DataFrame()
    )
    if locs.empty:
        return pd.DataFrame(
            columns=["site", "code", "site_type", "pollutants", "n", "from", "to", "lat", "lon"]
        )
    slug_by_notation = _OPENAQ_SLUG
    rows = []
    for _, r in locs.iterrows():
        have = [p for p in pollutants if slug_by_notation.get(p) in set(r.get("parameters", []))]
        if not have:
            continue
        rows.append(
            {
                "site": str(r.get("name") or r.get("id")),
                "code": str(r.get("id")),
                "site_type": str(r.get("provider") or ""),
                "pollutants": ", ".join(have),
                "n": len(have),
                "from": "",
                "to": str(r.get("last_updated") or "")[:10],
                "lat": r.get("lat"),
                "lon": r.get("lon"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["n", "site"], ascending=[False, True]).reset_index(drop=True)


def _fetch_uk(
    site: dict, pollutants: list[str], t0: pd.Timestamp, t1: pd.Timestamp, source: str
) -> pd.DataFrame:
    """UK networks via the openair archives — one file per station-year."""
    from normet import fetch_ukaq_measurements

    years = list(range(t0.year, t1.year + 1))
    log.info(
        "Fetching %s at %s (%s) for %d-%d…",
        ", ".join(pollutants),
        site["site"],
        site["code"],
        years[0],
        years[-1],
    )
    aq = fetch_ukaq_measurements(
        site["code"], years, source=source, pollutant=pollutants, on_missing="warn"
    )
    if aq.empty:
        raise RuntimeError(
            f"No data returned for {site['site']} ({site['code']}) in "
            f"{years[0]}–{years[-1]} — try a different station or date range."
        )
    aq["date"] = pd.to_datetime(aq["date"], utc=True).dt.tz_localize(None)
    return aq[(aq["date"] >= t0) & (aq["date"] <= t1 + pd.Timedelta(hours=23, minutes=59))]


def _fetch_eea(
    site: dict, pollutants: list[str], t0: pd.Timestamp, t1: pd.Timestamp, country: str
) -> pd.DataFrame:
    """EEA via one public Parquet blob per station and pollutant.

    Each blob is the station's whole record (2013 to now), so a single
    download per pollutant is the cheapest pull; the range is trimmed off
    here, the same way the UK path trims its station-year files.
    """
    from normet.io.eea import fetch_eea_station

    wide: dict[str, pd.Series] = {}
    for pol in pollutants:
        try:
            one = fetch_eea_station(station=site["code"], country=country, pollutant=pol)
        except RuntimeError:
            log.info("EEA: no %s blob for %s — skipped", pol, site["site"])
            continue
        if one.empty:
            continue
        s = pd.to_datetime(one["date"], utc=True).dt.tz_localize(None)
        wide[pol] = pd.Series(one["value"].to_numpy(), index=s)
    if not wide:
        raise RuntimeError(
            f"EEA returned no blobs for {site['site']} ({site['code']}) — "
            "try a different station or pollutant."
        )
    aq = pd.DataFrame(wide)
    aq = aq.reset_index().rename(columns={"index": "date"})
    return aq[(aq["date"] >= t0) & (aq["date"] <= t1 + pd.Timedelta(hours=23, minutes=59))]


def _fetch_openaq(
    site: dict, pollutants: list[str], t0: pd.Timestamp, t1: pd.Timestamp, api_key: str | None
) -> pd.DataFrame:
    """OpenAQ via the per-sensor measurements endpoint (keyed)."""
    from normet.io.openaq import fetch_openaq_measurements

    wide: dict[str, pd.Series] = {}
    for pol in pollutants:
        slug = _OPENAQ_SLUG.get(pol)
        if slug is None:
            log.info("OpenAQ does not serve %s — skipped at %s", pol, site["site"])
            continue
        one = fetch_openaq_measurements(
            location_id=int(site["code"]),
            parameter=slug,
            date_from=t0,
            date_to=t1,
            api_key=api_key,
        )
        if one.empty:
            continue
        d = pd.to_datetime(one["date"], utc=True)
        wide[pol] = pd.Series(one["value"].to_numpy(), index=d.dt.tz_localize(None))
    if not wide:
        raise RuntimeError(
            f"OpenAQ returned nothing for {site['site']} ({site['code']}) — "
            "check the key and date range, or try a different location."
        )
    aq = pd.DataFrame(wide)
    aq = aq.reset_index().rename(columns={"index": "date"})
    return aq


def _fetch_and_merge(
    site: dict,
    pollutants: list[str],
    date_from: str,
    date_to: str,
    met_source: str,
    provider: str,
    source: str | None,
    country: str,
    api_key: str | None,
) -> pd.DataFrame:
    """Download AQ + met for one site and outer-join on the hourly timestamp.

    *provider* selects the air-quality adapter; the meteorology half is
    shared (the station carries coordinates for every provider).
    """
    t0 = pd.Timestamp(date_from)
    t1 = pd.Timestamp(date_to)
    aq: pd.DataFrame
    if provider == PROV_UK:
        aq = _fetch_uk(site, pollutants, t0, t1, source or "aurn")
    elif provider == PROV_EEA:
        aq = _fetch_eea(site, pollutants, t0, t1, country)
    elif provider == PROV_OPENAQ:
        aq = _fetch_openaq(site, pollutants, t0, t1, api_key)
    else:
        raise ValueError(f"Unknown provider {provider!r}")

    pol_cols = [c for c in pollutants if c in aq.columns]
    if not pol_cols or aq.empty:
        raise RuntimeError(
            f"No air-quality data came back for {site['site']} in "
            f"{t0:%Y-%m-%d} to {t1:%Y-%m-%d} — try different pollutants or dates."
        )
    dropped = [p for p in pollutants if p not in pol_cols]
    if dropped:
        log.info("%s does not report %s — skipped", site["site"], ", ".join(dropped))
    # Already wide (one column per species); collapse any duplicate hours.
    merged = aq.groupby("date")[pol_cols].mean().sort_index()

    if met_source == MET_OPEN_METEO:
        from normet import fetch_openmeteo_timeseries

        met = fetch_openmeteo_timeseries(
            sites={site["site"]: (float(site["lat"]), float(site["lon"]))},
            date_from=date_from,
            date_to=date_to,
        )
        met = met.drop(columns=["site", "lat", "lon"]).set_index("date")
        merged = merged.join(met, how="left")
    elif met_source == MET_CDS:
        from normet import fetch_era5_timeseries

        met = fetch_era5_timeseries(
            sites={site["site"]: (float(site["lat"]), float(site["lon"]))},
            date_from=date_from,
            date_to=date_to,
        )
        met["date"] = pd.to_datetime(met["date"])
        if met["date"].dt.tz is not None:
            met["date"] = met["date"].dt.tz_localize(None)
        met = met.drop(columns=[c for c in ("site", "lat", "lon") if c in met], errors="ignore")
        merged = merged.join(met.set_index("date"), how="left")

    merged = merged.reset_index().rename(columns={"index": "date"})
    merged.insert(1, "site", site["site"])
    merged["lat"] = site["lat"]
    merged["lon"] = site["lon"]
    return merged


class DataWindow(QMainWindow):
    """ "Get data" window: air-quality networks + reanalysis met, merged.

    The UK networks, the EEA reporting network and OpenAQ all run through
    the same pick-a-station → fetch & merge flow.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Normet — Data Studio (AQ + met)")
        self.resize(1240, 820)
        self._main = parent
        self.stations: pd.DataFrame | None = None
        self.merged: pd.DataFrame | None = None

        self.runner = TaskRunner(self)
        self.runner.started.connect(self._task_started)
        self.runner.finished.connect(self._task_finished)
        self.task_clock = TaskClock(self, self.runner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_panel())
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 4, 4, 0)
        rv.addWidget(self._build_action_bar())
        rv.addWidget(self._build_tabs(), 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 880])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setMaximumWidth(220)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Abandon the running download.")
        self.cancel_btn.clicked.connect(self._abandon)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage(
            "Pick a network, tick pollutants, click 🔍 Find stations, "
            "pick a site, then ▶ Fetch & merge."
        )
        self._sync_enabled()

    # ------------------------------------------------------------- left panel
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)

        aq_box = QGroupBox("Air quality")
        av = QVBoxLayout(aq_box)
        av.addWidget(QLabel("Network"))
        self.net_combo = NoWheelComboBox()
        self.net_combo.addItems(list(NETWORKS) + [EEA_NETWORK, OPENAQ_NETWORK])
        self.net_combo.setToolTip(
            "Air-quality source to search.\n"
            "The six UK networks (AURN national + the regional networks);\n"
            "the EEA reporting network for all of Europe; or OpenAQ for\n"
            "most of the world (needs a free API key, see below)."
        )
        self.net_combo.currentIndexChanged.connect(self._network_changed)
        av.addWidget(self.net_combo)

        # Context fields shown only where the network needs one. EEA is a
        # single reporting network keyed by country; OpenAQ needs its key.
        ctx_box = QGroupBox("Source settings")
        cv = QFormLayout(ctx_box)
        cv.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.country_edit = QLineEdit()
        self.country_edit.setPlaceholderText("ISO-2, e.g. GB, DE, FR, PL, CN")
        self.country_edit.setMaximumWidth(180)
        cv.addRow("Country", self.country_edit)
        self.openaq_key = QLineEdit()
        self.openaq_key.setPlaceholderText("free key from explore.openaq.org (else OPENAQ_API_KEY)")
        self.openaq_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openaq_key.textEdited.connect(lambda *_: self._clear_key_tip())
        cv.addRow("OpenAQ key", self.openaq_key)
        self.openaq_hint = hint_label("", small=True)
        cv.addRow(self.openaq_hint)
        self.ctx_box = ctx_box
        av.addWidget(ctx_box)
        # Only the visibility rules here: the station table does not exist yet,
        # so the full _network_changed would run before its widgets.
        self._refresh_context()

        av.addWidget(QLabel("Pollutants"))
        self.pol_list = QListWidget()
        self.pol_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.pol_list.setMaximumHeight(150)
        self.pol_list.setToolTip("Each ticked pollutant becomes one column of the merged table.")
        for pol in POLLUTANTS:
            item = QListWidgetItem(pol)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if pol in DEFAULT_POLLUTANTS else Qt.CheckState.Unchecked
            )
            self.pol_list.addItem(item)
        av.addWidget(self.pol_list)
        self.find_btn = run_button(
            "🔍  Find stations",
            "List every site in the chosen network measuring the ticked\npollutants, on the right.",
        )
        self.find_btn.clicked.connect(self._run_find_stations)
        av.addWidget(self.find_btn)
        self.station_hint = hint_label("No station chosen yet", small=False)
        av.addWidget(self.station_hint)
        v.addWidget(aq_box)

        rng_box = QGroupBox("Date range")
        rf = QFormLayout(rng_box)
        rf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        today = QDate.currentDate()
        self.date_from = QDateEdit(QDate(today.year() - 3, 1, 1))
        self.date_to = QDateEdit(today.addDays(-3))
        for de in (self.date_from, self.date_to):
            de.setCalendarPopup(True)
            de.setDisplayFormat("yyyy-MM-dd")
        rf.addRow("From", self.date_from)
        rf.addRow("To", self.date_to)
        rf.addRow(
            hint_label(
                "The archives run from each station's opening date (see the\n"
                "from/to columns); selecting a station snaps the range to it.\n"
                "Whole years are downloaded, then trimmed to this range."
            )
        )
        v.addWidget(rng_box)

        met_box = QGroupBox("Meteorology")
        mf = QFormLayout(met_box)
        mf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.met_combo = NoWheelComboBox()
        self.met_combo.addItems([MET_OPEN_METEO, MET_CDS])
        self.met_combo.setToolTip(
            "Hourly meteorology at the station coordinates.\n"
            "Open-Meteo serves the ERA5 archive without registration;\n"
            "the Copernicus CDS needs an account and ~/.cdsapirc."
        )
        mf.addRow("Source", self.met_combo)
        mf.addRow(
            hint_label(
                "Adds t2m, d2m, rh2m, sp, tcc, tp, ssrd, ws, wd, u10, v10 —\n"
                "the predictors the Train step auto-recognises as met."
            )
        )
        v.addWidget(met_box)

        self.fetch_btn = run_button(
            "▶  Fetch && merge",
            "Download the pollutant series and the meteorology, and join them\ninto one hourly table ready for training.",
        )
        self.fetch_btn.clicked.connect(self._run_fetch)
        v.addWidget(self.fetch_btn)

        v.addStretch(1)
        self._run_buttons = [self.find_btn, self.fetch_btn]

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 0, 4, 0)
        self.save_btn = QPushButton("💾  Save merged CSV…")
        self.save_btn.clicked.connect(self._save_csv)
        self.save_btn.setEnabled(False)
        self.send_btn = QPushButton("⬆  Send to main window")
        self.send_btn.setToolTip("Load the merged table into the main window, ready for training.")
        self.send_btn.clicked.connect(self._send_to_main)
        self.send_btn.setEnabled(False)
        h.addWidget(self.save_btn)
        h.addWidget(self.send_btn)
        h.addStretch(1)
        return bar

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        # Pages on the window's own ground, as the toolbars above their figures are: a
        # framed pane drew them lighter, and each toolbar as a grey band across it.
        self.tabs.setDocumentMode(True)

        st_tab = QWidget()
        sl = QVBoxLayout(st_tab)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("e.g. Manchester or MAN3")
        self.search_edit.textChanged.connect(self._filter_stations)
        search_row.addWidget(self.search_edit)
        sl.addLayout(search_row)
        self.station_table = QTableWidget()
        self.station_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.station_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.station_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.station_table.itemSelectionChanged.connect(self._station_selected)
        sl.addWidget(self.station_table, 1)
        sl.addWidget(
            hint_label("Click 🔍 Find stations to fill this table, then select the site to use.")
        )
        self.tabs.addTab(st_tab, "Stations")

        self.tab_preview = CanvasTab("Fetch data to preview the merged table here.")
        self.tabs.addTab(self.tab_preview, "Preview")

        pv_tab = QWidget()
        pl = QVBoxLayout(pv_tab)
        self.preview_summary = QLabel("No data fetched yet.")
        self.preview_summary.setWordWrap(True)
        pl.addWidget(self.preview_summary)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        pl.addWidget(self.preview_table, 1)
        self.tabs.addTab(pv_tab, "Table")
        return self.tabs

    # ---------------------------------------------------------------- helpers
    def _checked_pollutants(self) -> list[str]:
        return [
            self.pol_list.item(i).text()
            for i in range(self.pol_list.count())
            if self.pol_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _sync_enabled(self) -> None:
        busy = self.runner.busy
        self.find_btn.setEnabled(not busy)
        self.fetch_btn.setEnabled(not busy and self._selected_station() is not None)
        has_data = self.merged is not None
        self.save_btn.setEnabled(has_data)
        self.send_btn.setEnabled(has_data and self._main is not None)
        self.cancel_btn.setEnabled(busy)

    def _task_started(self, name: str) -> None:
        for b in self._run_buttons:
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _task_finished(self) -> None:
        self._sync_enabled()

    def _abandon(self) -> None:
        if self.runner.busy:
            self.runner.abandon()
            self.statusBar().showMessage("Download abandoned.")

    def _show_error(self, tb: str) -> None:
        QMessageBox.critical(
            self, "Download failed", tb.splitlines()[-1] if tb else "Unknown error"
        )
        log.error("%s", tb)

    # ---------------------------------------------------------------- actions
    def _provider(self) -> tuple[str, str | None, str]:
        """(provider, ukaq source, country) for the current network text."""
        prov, src = _network_meta(self.net_combo.currentText())
        country = self.country_edit.text().strip().upper()
        return prov, src, country

    def _openaq_key(self) -> str | None:
        key = self.openaq_key.text().strip()
        return key or (os.environ.get("OPENAQ_API_KEY") or None)

    def _refresh_context(self) -> None:
        """Show only the source fields the chosen network needs."""
        prov = _network_meta(self.net_combo.currentText())[0]
        self.country_edit.setVisible(prov in (PROV_EEA, PROV_OPENAQ))
        self.openaq_key.setVisible(prov == PROV_OPENAQ)
        self.openaq_hint.setText(
            "Set OPENAQ_API_KEY or paste a free key to search OpenAQ."
            if prov == PROV_OPENAQ
            else ""
        )
        self.ctx_box.setVisible(prov != PROV_UK)

    def _clear_key_tip(self) -> None:
        if self.openaq_hint and self.openaq_key.text().strip():
            self.openaq_hint.setText("")

    def _network_changed(self) -> None:
        """A station list belongs to one network; drop it when that changes."""
        self.stations = None
        self.station_table.clearContents()
        self.station_table.setRowCount(0)
        self.station_hint.setText("No station chosen yet")
        self._refresh_context()
        self.statusBar().showMessage(
            f"Network set to {self.net_combo.currentText()} — click 🔍 Find stations."
        )
        self._sync_enabled()

    def _run_find_stations(self) -> None:
        pollutants = self._checked_pollutants()
        if not pollutants:
            QMessageBox.information(self, "No pollutants", "Tick at least one pollutant.")
            return
        prov, src, country = self._provider()
        if prov in (PROV_EEA, PROV_OPENAQ) and not country:
            QMessageBox.information(
                self,
                "No country",
                "Type an ISO-2 country code first (GB, DE, FR, …)."
                if prov == PROV_EEA
                else "Type an ISO-2 country code first (GB, US, BR, …).",
            )
            return
        key = self._openaq_key()
        if prov == PROV_OPENAQ and not key:
            QMessageBox.information(
                self,
                "No OpenAQ key",
                "OpenAQ needs a free API key — paste one above or set OPENAQ_API_KEY. "
                "Register at https://explore.openaq.org/register.",
            )
            return
        self.runner.submit(
            "find stations",
            _find_stations,
            self._stations_done,
            self._show_error,
            prov,
            src,
            country,
            pollutants,
            key,
        )

    def _stations_done(self, df: pd.DataFrame) -> None:
        self.stations = df
        self._fill_station_table(df)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            f"{len(df)} sites measure the ticked pollutants — search and select one."
        )

    def _fill_station_table(self, df: pd.DataFrame) -> None:
        cols = ["site", "code", "site_type", "pollutants", "from", "to", "lat", "lon"]
        self.station_table.clear()
        self.station_table.setRowCount(len(df))
        self.station_table.setColumnCount(len(cols))
        self.station_table.setHorizontalHeaderLabels(
            ["Site", "Code", "Type", "Pollutants", "From", "To", "Lat", "Lon"]
        )
        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                val = row[col]
                if col in ("lat", "lon") and pd.notna(val):
                    val = f"{float(val):.4f}"
                item = QTableWidgetItem(str(val))
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, int(row.name))
                self.station_table.setItem(r, c, item)
        self.station_table.resizeColumnsToContents()

    def _filter_stations(self, text: str) -> None:
        if self.stations is None:
            return
        text = text.strip().lower()
        df = self.stations
        if text:
            match = df["site"].astype(str).str.lower().str.contains(text, na=False)
            match |= df["code"].astype(str).str.lower().str.contains(text, na=False)
            # Site type is searchable too: "rural" or "traffic" is often what
            # you actually want to filter on across ~1500 stations.
            match |= df["site_type"].astype(str).str.lower().str.contains(text, na=False)
            df = df[match]
        self._fill_station_table(df)

    def _selected_station(self) -> dict | None:
        if self.stations is None:
            return None
        items = self.station_table.selectedItems()
        if not items:
            return None
        idx = self.station_table.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if idx is None or idx not in self.stations.index:
            return None
        return self.stations.loc[idx].to_dict()

    def _station_selected(self) -> None:
        st = self._selected_station()
        if st:
            cov = f"{st.get('from', '')} → {st.get('to', '')}" if st.get("to") else "unknown"
            code = f" ({st['code']})" if st.get("code") else ""
            stype = f"\ntype: {st['site_type']}" if st.get("site_type") else ""
            self.station_hint.setText(
                f"Selected: {st['site']}{code}{stype}\n"
                f"measures: {st['pollutants']}\ndata held: {cov}"
            )
            self.station_hint.setStyleSheet("")
            # Snap the pickers to the period the archive covers. Unlike the
            # old SOS path there is no rolling window to fight, so the whole
            # record is offered rather than the last ~6 months -- but capped
            # at 5 years so a first click does not queue a 25-year download.
            if st.get("from") and st.get("to"):
                lo = QDate.fromString(str(st["from"]), "yyyy-MM-dd")
                hi = QDate.fromString(str(st["to"]), "yyyy-MM-dd")
                if lo.isValid() and hi.isValid():
                    # Open-Meteo's archive lags a few days behind real time.
                    hi = min(hi, QDate.currentDate().addDays(-5))
                    self.date_from.setDate(max(lo, hi.addYears(-5)))
                    self.date_to.setDate(hi)
        self._sync_enabled()

    def _run_fetch(self) -> None:
        st = self._selected_station()
        if st is None:
            QMessageBox.information(
                self, "No station", "Find stations and select one in the table first."
            )
            return
        pollutants = self._checked_pollutants()
        d_from = self.date_from.date().toString("yyyy-MM-dd")
        d_to = self.date_to.date().toString("yyyy-MM-dd")
        if pd.Timestamp(d_from) >= pd.Timestamp(d_to):
            QMessageBox.information(self, "Bad range", "'From' must be before 'To'.")
            return
        met = self.met_combo.currentText()
        prov, src, country = self._provider()
        self.runner.submit(
            f"fetch {st['site']}",
            _fetch_and_merge,
            self._fetch_done,
            self._show_error,
            st,
            pollutants,
            d_from,
            d_to,
            met,
            prov,
            src,
            country,
            self._openaq_key(),
        )

    def _fetch_done(self, df: pd.DataFrame) -> None:
        self.merged = df
        n_nan = int(df.isna().sum().sum())
        pol_cols = [c for c in df.columns if c in POLLUTANTS]
        met_cols = [c for c in df.columns if c in ("t2m", "ws", "wd", "sp", "rh2m")]
        self.preview_summary.setText(
            f"{len(df):,} hourly rows × {df.shape[1]} columns   |   "
            f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}   |   "
            f"pollutants: {', '.join(pol_cols)}   |   missing cells: {n_nan:,}"
        )
        self._fill_preview_table(df)
        self._draw_preview(df, pol_cols)
        self.tabs.setCurrentWidget(self.tab_preview)
        self._sync_enabled()
        verdict_met = "with met" if met_cols else "WITHOUT met"
        self.statusBar().showMessage(
            f"Merged table ready ({verdict_met}) — save it or send it to the main window."
        )

    def _fill_preview_table(self, df: pd.DataFrame, max_rows: int = 300) -> None:
        head = df.head(max_rows)
        self.preview_table.clear()
        self.preview_table.setRowCount(len(head))
        self.preview_table.setColumnCount(df.shape[1])
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(head)):
            for c in range(df.shape[1]):
                val = head.iat[r, c]
                if isinstance(val, float):
                    val = f"{val:.4g}"
                self.preview_table.setItem(r, c, QTableWidgetItem(str(val)))
        self.preview_table.resizeColumnsToContents()

    def _draw_preview(self, df: pd.DataFrame, pol_cols: list[str]) -> None:
        import matplotlib.pyplot as plt

        n = max(1, len(pol_cols))
        fig, axes = plt.subplots(n, 1, figsize=(8, 1.5 * n + 0.4), sharex=True, squeeze=False)
        d = df.set_index("date")
        shown = []
        for ax, pol in zip(axes.ravel(), pol_cols or [None], strict=False):
            if pol is None:
                ax.axis("off")
                continue
            S.line(
                ax,
                d.index,
                d[pol],
                lw=0.5 if len(d) > 1500 else 0.9,
                color=S.PALETTE[len(shown) % 4],
            )
            cov = d[pol].notna().mean() * 100
            ax.set_title(f"{S.formula(pol)} — {cov:.0f} % coverage")
            ax.set_ylabel(S.formula(pol))
            ax.margins(x=0)
            S.grid(ax)
            shown.append(ax)
        if shown and isinstance(d.index, pd.DatetimeIndex):
            S.format_time_axis(shown[-1])
        S.outer_ticks(shown)
        fig.align_ylabels(shown)
        self.tab_preview.show_result(
            fig,
            verdict=(
                "ok",
                f"Fetched {len(df):,} hourly rows for {df['site'].iloc[0]}.",
            )
            if len(df)
            else ("warn", "Empty result."),
            lines=[
                "Columns follow the modelling convention: met variables are "
                "auto-ticked as features for training."
            ],
        )

    def _save_csv(self) -> None:
        if self.merged is None:
            return
        site = str(self.merged["site"].iloc[0]).replace(" ", "_") if len(self.merged) else "data"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save merged CSV", f"normet_{site}.csv", "CSV (*.csv)"
        )
        if path:
            self.merged.to_csv(path, index=False)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
        # macOS returns focus to the parent (main) window after the file
        # dialog closes; keep this window in front.
        self.raise_()
        self.activateWindow()

    def _send_to_main(self) -> None:
        if self.merged is None or self._main is None:
            return
        self._main.ingest_dataframe(
            self.merged.copy(), f"Data Studio: {self.merged['site'].iloc[0]}"
        )
        self._main.raise_()
        self._main.activateWindow()
        self.statusBar().showMessage("Sent to the main window — train a model there.")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.runner.shutdown()
        super().closeEvent(event)
