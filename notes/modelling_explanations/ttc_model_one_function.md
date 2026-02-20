# ttc_model_one_function.py

Interactive TTC model GUI with two frequencies.

## Purpose

Build synthetic TTC maps using a simple two-frequency model with sliders for interactive exploration.

## Model

Generates TTC maps with:
- Direct (delay) component: amplitude $A_d$, frequency $\omega_d$
- Indirect (mean-time) component: amplitude $A_i$, frequency $\omega_i$

## Key functions

- `symmetrize(C)`: Mirror along $t_1=t_2$ diagonal
- `clip_percentile(C, p_hi)`: Clip values for display
- `make_ttc_model_gui_two_freqs(...)`: Main GUI builder

## Parameters

- `n`: number of time points (default 450)
- `tmax`: time axis scaling in seconds (default 4800)
- `Ad`, `Ai`: amplitudes for delay and indirect components
- `wd`, `wi`: angular frequencies (rad/s)

## Usage

```bash
python ttc_model_one_function.py
```

Uses macOS backend by default; change `mpl.use("macosx")` if needed.
