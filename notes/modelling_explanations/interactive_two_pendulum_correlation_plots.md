# interactive_two_pendulum_correlation_plots.py

Interactive 2-pendulum correlation GUI with sliders + equations box.

## Top row

1. **Time traces:** $I_1(t)$, $I_2(t)$
2. **Cross-correlation:** $g_2^{12}(\tau)$
3. **Two-time cross-correlation map:** (stripe-forming)

## Bottom row

- **Left:** equations box
- **Right:** sliders

## Model

Uses `make_pendulum_signal(t, theta0, omega, gamma, phi, noise_sigma, seed)` for each pendulum.

Each pendulum:
$$\theta(t) = \theta_0 \exp(-\gamma t) \cos(\omega t + \phi) + \text{noise}$$

Cross-correlation between the two pendulums reveals phase relationships and produces stripe patterns in the two-time map.

## Requirements

```bash
pip install PyQt6
```

## Usage

```bash
python interactive_two_pendulum_correlation_plots.py
```
