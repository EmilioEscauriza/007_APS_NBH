# three_TTC_demo.py

Interactive Langevin oscillator demo showing **three** "two-time" constructions side-by-side for the SAME simulation data.

## Top row (all square)

1. **Outer-product (single trajectory):**
   $$M_{\text{outer}}(t_1,t_2) = \frac{x_1(t_1) x_1(t_2)}{\sqrt{\langle x_1^2(t_1) \rangle \langle x_1^2(t_2) \rangle}}$$
   Rank-1 look, chequer-prone.

2. **Ensemble two-time correlation (normalized):**
   $$C(t_1,t_2) = \langle x(t_1) x(t_2) \rangle_{\text{traj}}$$
   $$C_{\text{norm}}(t_1,t_2) = \frac{C}{\sqrt{C(t_1,t_1) C(t_2,t_2)}}$$
   Stripe-prone.

3. **Lag-based "stripe TTC":**
   $$G(\tau) = \langle x(t) x(t+\tau) \rangle_{t,\text{traj}}$$
   $$M_{\text{lag}}(t_1,t_2) = G(|t_2-t_1|) / G(0)$$
   Forced stripes.

## Bottom row

- **Left:** equations/definitions box
- **Right:** sliders + buttons

## Dynamics (semi-implicit Euler–Maruyama)

$$dv = \left(-\frac{\gamma}{m} v - \frac{k}{m} x\right) dt + \sigma \, dW$$
$$x_{n+1} = x_n + v_{n+1} \, dt$$

with $\sigma = \sqrt{2 \gamma k_B T}/m$

## Requirements

```bash
pip install PyQt6
```

## Usage

```bash
python three_TTC_demo.py
```
