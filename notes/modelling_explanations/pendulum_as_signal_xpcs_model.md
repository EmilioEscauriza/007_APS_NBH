# pendulum_as_signal_xpcs_model.py

Interactive "pendulum-as-a-signal" XPCS-like toy model.

## Model

$$\theta_p(t) = \theta_0 \exp(-\gamma t) \cos(\omega t + \phi_p) + \eta_p(t)$$

$$I_p(t) = I_0 [1 + a \cdot \theta_p(t)]$$

## Key point for XPCS relevance

TTC is computed with a pixel/ROI ensemble average ($p$ = "pixels"):

$$C(t_1,t_2) = \frac{\langle I_p(t_1) I_p(t_2) \rangle_p}{\langle I_p(t_1) \rangle_p \langle I_p(t_2) \rangle_p}$$

This is the standard two-time ($g_2$) normalization used in XPCS — it largely removes frame-to-frame mean intensity changes and highlights speckle-like fluctuations.

## Layout

- **Top left:** $g_2(\tau) = \langle I(t) I(t+\tau) \rangle_{p,t} / \langle I \rangle^2$
- **Top right:** TTC map $C(t_1,t_2)$ (square)
- **Bottom left:** equations box
- **Bottom right:** sliders + buttons

## Requirements

```bash
pip install PyQt6
```

## Usage

```bash
python pendulum_as_signal_xpcs_model.py
```
