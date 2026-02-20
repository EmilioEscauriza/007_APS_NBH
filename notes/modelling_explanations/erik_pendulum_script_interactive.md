# erik_pendulum_script_interactive.py

Interactive ensemble Langevin oscillator (fixed ensemble size, with in-place text box).

## Layout

**Top row (2 panels):**
- **Left:** multi-tau autocorrelation $g(\tau)$ from $x(t)$
- **Right:** normalized two-time map $C_{\text{norm}}(t_1,t_2)$, titled "Damped Langevin: C(t1,t2)" (square aspect)

**Bottom row:**
- **Left:** equations/text box (in-layout)
- **Right:** sliders

## Dynamics (semi-implicit Euler–Maruyama)

$$dv = \left(-\frac{\gamma}{m} v - \frac{k}{m} x\right) dt + \sigma \, dW$$
$$x_{n+1} = x_n + v_{n+1} \, dt$$

## Notes

- `Ntraj` is FIXED (no slider), set to match the scale of earlier GUIs
- Guardrail prevents `Nt` from becoming too large ($Nt \times Nt$ map cost)

## Requirements

```bash
pip install PyQt6
```

## Usage

```bash
python erik_pendulum_script_interactive.py
```
