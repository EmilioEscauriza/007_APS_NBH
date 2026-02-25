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

**Heterodyne (time-dependent):** The fit uses the same functional forms as the plot script (forward time $t$):
- $J(t) = J_0\,(1 - e^{-t/\tau_J})$, $\tau_J = \texttt{j\_tau\_frac} \cdot t_{\max}$
- $x_s(t) = 0.25 + 0.45\,(1 - e^{-t/\tau_{x_s}})$, $\tau_{x_s} = \texttt{xs\_tau\_frac} \cdot t_{\max}$
- $v(t) = v_{\text{mean}}\,(1 + a\,\cos(2\pi t/T))$, clipped to stay positive ($a$ = v_amp_frac, $T$ = v_period_s)

Fitted 7-parameter vector: `[J0, j_tau_frac, v_mean, v_amp_frac, v_period_s, xs_tau_frac, beta]`. For stripes-only (no ramp), set `CONSTANT_J_AND_XS = True` to fit 4 params: J0, v0, x_s_const, beta.

## Output

- Fitted parameters
- Comparison plots of data vs model
