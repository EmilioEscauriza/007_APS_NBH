# ttc_fit_transport_coefficient.py

Fit experimental TTC arrays with the transport-coefficient model (He et al. 2024/2025 PNAS).

## Purpose

Loads a saved `.npy` TTC array from `data/ttc_arrays/<POSITION_NAME>/<SAMPLE_ID>/` and fits either:
- **Laminar (homodyne)** model
- **Heterodyne (two-component)** model

## Configuration

Set these variables at the top of the script:
- `SAMPLE_ID`: e.g. "A073"
- `MASK_N`: mask number
- `POSITION_NAME`: e.g. "A4"
- `TIME_RANGE`: `"full"` or `[index_start, index_end, stride]` (e.g. `[1000, 3000, 2]`)

## Usage

```bash
python ttc_fit_transport_coefficient.py
```

## Model

Uses `c2_laminar` and `c2_heterodyne` from `ttc_plots_transport_coefficient.py`. See that script's explanation for the formulas.

## Output

- Fitted parameters
- Comparison plots of data vs model
