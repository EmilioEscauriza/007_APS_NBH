# synthetic_data.py

Interactive TTC toy model with sliders (currently commented out).

## Purpose

Generate synthetic two-time correlation maps using a simple analytic model with two frequency components.

## Model (from comments)

Coordinates:
- $U = (t_1 + t_2)/2$ (mean time)
- $V = t_2 - t_1$ (lag/delay)

Correlation:
$$C(t_1,t_2) = C_0 + A \cos(\omega V) + m \cos(\omega U + \phi) \cos(\omega_m V)$$

where $\omega_m = \omega / 2$.

## Status

Code is currently commented out. Uncomment and run with PyQt6 for interactive GUI.

## Requirements

```bash
pip install PyQt6
```
