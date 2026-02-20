# Bragg peak over time notebook

This notebook uses **only** code from `scripts/analysis_for_aps_08-ide-2025-1006.py`.

## How to run

1. **From terminal:**
   ```bash
   cd /path/to/007_APS_NBH
   jupyter notebook notebooks/bragg_peak_over_time.ipynb
   ```

2. **From Cursor / VS Code:**
   Open `notebooks/bragg_peak_over_time.ipynb`. Use the same Python env as the analysis script (h5py, matplotlib, scipy). Run cells top to bottom.

3. **Paths:** In the first code cell, set `RESULTS_HDF` and `NPZ_PATH`. Only the HDF keys used in the notebook are read (no full 14GB load).

## What it loads

- `xpcs/temporal_mean/scattering_2d` – one 2D image (mean over time)
- `xpcs/spatial_mean/intensity_vs_time` – time series
- q/phi maps from the npz (same as the analysis script's bragg-peak commands)

You can change parameters (e.g. `brightest_percentile`) and re-run cells below to iterate.
