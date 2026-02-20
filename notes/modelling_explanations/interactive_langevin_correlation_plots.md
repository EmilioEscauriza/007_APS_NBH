# interactive_langevin_correlation_plots.py

Interactive damped Langevin oscillator with mixed TTC (stripes + chequer).

## Model

Euler–Maruyama integration:
$$d\theta = v \, dt$$
$$dv = (-\gamma v - \omega_0^2 \theta) \, dt + \sigma \, dW$$

## Plots (top row)

1. **Time trace:** $I(t)$ from $\theta(t)$
2. **$g_2(\tau)$:** $\langle I(t) I(t+\tau) \rangle / \langle I \rangle^2$
3. **Two-time correlation map (TTC):**
   - `TTC = mix * TTC_stripes + (1-mix) * TTC_chequer`
   - TTC_stripes: depends on lag $|t_2-t_1|$ → diagonal stripes
   - TTC_chequer: outer product $I(t_1)I(t_2)$ → chequer texture
   - Mix slider blends them to reproduce "stripes + chequer"

## Layout

- **Bottom left:** equations box
- **Bottom right:** sliders

## Requirements

```bash
pip install PyQt6
```

## Usage

```bash
python interactive_langevin_correlation_plots.py
```
