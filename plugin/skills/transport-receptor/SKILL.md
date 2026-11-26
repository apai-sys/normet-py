---
name: transport-receptor
description: >-
  Work with air-mass trajectories and receptor attribution in normet: add
  meteorological physics and memory features, build trajectory-based receptor
  maps (concentration-weighted and potential-source contribution), cluster
  trajectories, and invert footprints across several receptors. Use when the
  user asks where pollution came from, wants a source-receptor map, or has
  HYSPLIT trajectories to analyse.
---

# Trajectories and Receptor Attribution

normet reads HYSPLIT trajectory dumps (`.tdump`), builds receptor maps from
them, and can invert a set of receptors for their footprints.

## 1. Physics and memory features

- `normet_make_met_features` adds physically meaningful columns to a table:
  ventilation, stagnation, recirculation, mixing depth and short memory terms
  of the weather.
- Use it when the raw meteorology is too coarse for the model, then run the
  de-weathering with the enriched table.

## 2. Receptor maps

- A trajectory dump gives, for each receptor time and each back-trajectory
  point, a position and a meteorological state. Two maps are built from them:
  - **Concentration-weighted trajectories (WCWT)**: the observed
    concentration at the receptor time spread along its back-trajectory.
  - **Potential source contribution (WPSCF)**: the share of trajectories
    passing a grid cell above a concentration threshold.
- Aggregate over a grid with a stated resolution, keep the receptor times that
  have a complete trajectory, and say how many were used.

## 3. Clustering

- Cluster trajectories (for example by their coordinates and length) to name
  the air-mass sectors; report each cluster's share of the receptor times and
  its mean concentration.
- The number of clusters is the analyst's decision: propose a count from the
  elbow or the silhouette, and say what each cluster represents.

## 4. Multi-site inversion

- `normet_invert_multisite` fits several receptors at once, so a source region
  must explain all of them rather than one site's record.
- Report the fitted footprint, the number of receptors, the grid resolution
  and the residuals per site.
- A single-site inversion is weaker evidence: say so.

Quote numbers only as the tools give them.
