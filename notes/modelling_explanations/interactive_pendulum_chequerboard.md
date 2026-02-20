# interactive_pendulum_chequerboard.py

Interactive version of the original single-pendulum correlation figure.

## Layout

- **Left:** $g_2(\tau) = \langle I(t) I(t+\tau) \rangle / \langle I \rangle^2$
- **Right:** $\text{TTC}(t_1,t_2) = I(t_1) I(t_2) / \langle I \rangle^2$ (outer-product map)

## Purpose

A simple "pendulum-as-a-signal" demo, not a full XPCS simulator. Shows:
- How a damped oscillator signal produces autocorrelation
- How the outer-product construction creates a "chequerboard" pattern in the TTC

## Key function

`make_pendulum_signal(t, theta0, ...)`: Generates damped oscillation signal with optional noise.

## Requirements

```bash
pip install PyQt6
```

## Usage

```bash
python interactive_pendulum_chequerboard.py
```
