"""
Interactive ensemble Langevin oscillator (fixed ensemble size, with in-place text box).

TOP ROW (2 panels):
  LEFT : multi-tau autocorrelation g(τ) from x(t)
  RIGHT: normalized two-time map Cnorm(t1,t2), titled "Damped Langevin: C(t1,t2)"
         (square aspect on screen)

BOTTOM ROW:
  LEFT : equations/text box (in-layout, not overlaid)
  RIGHT: sliders

Dynamics (semi-implicit Euler–Maruyama):
    dv = (-(gamma/m) v - (k/m) x) dt + sigma dW
    x_{n+1} = x_n + v_{n+1} dt

Notes:
- Ntraj is FIXED (no slider), set to match the scale of your earlier GUIs.
- Guardrail prevents Nt from becoming too large (Nt×Nt map cost).

Requires:
  pip install PyQt6

Run:
  python langevin_ensemble_gui_fixedNtraj.py
"""

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# -------------------------
# Fixed ensemble size
# -------------------------
NTRAJ = 1024  # "same scale" as earlier GUIs; change here if desired


# -------------------------
# Simulation + correlations
# -------------------------

def simulate_ensemble_langevin(
    m: float,
    k: float,
    gamma: float,
    kB: float,
    T: float,
    tmax: float,
    dt: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        t : (Nt,)
        x : (NTRAJ, Nt)
    """
    rng = np.random.default_rng(int(seed))

    t = np.arange(0.0, tmax + dt, dt)
    Nt = len(t)

    # canonical equilibrium initial conditions
    x0 = rng.normal(0.0, np.sqrt(kB * T / k), size=NTRAJ)
    v0 = rng.normal(0.0, np.sqrt(kB * T / m), size=NTRAJ)

    x = np.zeros((NTRAJ, Nt), dtype=float)
    v = np.zeros((NTRAJ, Nt), dtype=float)
    x[:, 0] = x0
    v[:, 0] = v0

    sigma = np.sqrt(2.0 * gamma * kB * T / (m**2))
    sqrt_dt = np.sqrt(dt)

    for n in range(Nt - 1):
        dW = rng.normal(0.0, 1.0, size=NTRAJ) * sqrt_dt
        v[:, n + 1] = v[:, n] + (-(gamma / m) * v[:, n] - (k / m) * x[:, n]) * dt + sigma * dW
        x[:, n + 1] = x[:, n] + v[:, n + 1] * dt

    return t, x


def two_time_cnorm(x: np.ndarray) -> np.ndarray:
    """
    Normalized two-time correlation:
      C(t1,t2) = <x(t1)x(t2)>_traj
      Cnorm = C / sqrt(C(t1,t1) C(t2,t2))
    """
    C = (x.T @ x) / float(x.shape[0])
    diag = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / (diag[:, None] * diag[None, :])


# -------------------------
# Multi-tau autocorrelation
# -------------------------

def multitau_autocorr_ensemble(x: np.ndarray, dt: float, m: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-tau style autocorrelation using time + trajectory averaging.

    Computes:
      G(τ) = < x(t) x(t+τ) >_{t,traj}
      g(τ) = G(τ) / G(0)

    Returns:
      tau_seconds, g_norm
    """
    m = int(m)
    if m < 4:
        raise ValueError("multi-tau m must be >= 4")
    if m % 2 != 0:
        m += 1  # enforce even

    y = x
    dt_eff = float(dt)

    taus = []
    gvals = []

    level = 0
    while y.shape[1] > (m + 1):
        Nt = y.shape[1]

        lag_list = np.arange(0, m) if level == 0 else np.arange(m // 2, m)

        for lag in lag_list:
            a = y[:, : Nt - lag]
            b = y[:, lag: Nt]
            gvals.append(np.mean(a * b))
            taus.append(lag * dt_eff)

        # safe downsample (drop odd point)
        Nt_even = Nt - (Nt % 2)
        y = y[:, :Nt_even]
        y = 0.5 * (y[:, 0::2] + y[:, 1::2])
        dt_eff *= 2.0
        level += 1
        if level > 60:
            break

    taus = np.asarray(taus, dtype=float)
    gvals = np.asarray(gvals, dtype=float)
    order = np.argsort(taus)
    taus = taus[order]
    gvals = gvals[order]

    g0 = gvals[0] if np.isfinite(gvals[0]) and abs(gvals[0]) > 1e-20 else 1.0
    return taus, gvals / g0


# -------------------------
# GUI
# -------------------------

def main():
    # Defaults (interactive-safe)
    p0 = dict(
        m=10.0,
        k=1.0,
        gamma=0.1,
        kB=1.0,      # fixed in this GUI, but kept in the equations box
        T=1.0,
        tmax=40.0,
        dt=0.02,
        seed=1,      # fixed (if you want a slider, say so)
        mt_m=16,
    )

    # ---- Layout: 2x2 grid ----
    # Top row: plots
    # Bottom row: text box (left) + sliders (right)
    fig = plt.figure(figsize=(16.0, 8.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 0.78],
        width_ratios=[1.0, 1.0],
        left=0.05, right=0.985, top=0.93, bottom=0.07,
        wspace=0.25, hspace=0.30,
    )

    ax_g2 = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.set_axis_off()

    ax_slider_area = fig.add_subplot(gs[1, 1])
    ax_slider_area.set_axis_off()

    # ---- Text box (in-place) ----
    eq_text = (
        r"$dv = (-(\gamma/m)v - (k/m)x)\,dt + \sigma\,dW$" "\n"
        r"$x_{n+1}=x_n+v_{n+1}dt$ (semi-implicit)" "\n\n"
        r"$\sigma=\sqrt{2\gamma k_B T}/m$" "\n\n"
        r"$C(t_1,t_2)=\langle x(t_1)x(t_2)\rangle_{\rm traj}$" "\n"
        r"$C_{\rm norm}=\frac{C}{\sqrt{C(t_1,t_1)C(t_2,t_2)}}$" "\n\n"
        r"$g(\tau)=\langle x(t)x(t+\tau)\rangle/\langle x^2\rangle$" "\n\n"
        rf"Fixed: $N_{{traj}}={NTRAJ}$, $k_B={p0['kB']}$, seed={p0['seed']}"
    )
    ax_text.text(
        0.02, 0.98, eq_text,
        va="top", ha="left",
        fontsize=11,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", alpha=0.95),
        transform=ax_text.transAxes,
    )
    ax_text.set_title("Equations / definitions", loc="left", pad=6)

    # ---- TOP LEFT: multi-tau g(τ) ----
    (line_g2,) = ax_g2.plot([], [], lw=2)
    ax_g2.set_xscale("log")
    ax_g2.set_xlabel(r"lag $\tau$ (s)")
    ax_g2.set_ylabel(r"$g(\tau)$")
    ax_g2.set_title("Multi-tau autocorrelation")
    ax_g2.grid(True, alpha=0.3)

    # ---- TOP RIGHT: normalized two-time map (square) ----
    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    im = ax_map.imshow(
        np.zeros((10, 10)),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=[0, 1, 0, 1],
        aspect="equal",
    )
    ax_map.set_title("Damped Langevin: C(t1,t2)")
    ax_map.set_xlabel("t2 (s)")
    ax_map.set_ylabel("t1 (s)")
    ax_map.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("Cnorm")

    # ---- Sliders region (inside bottom-right cell) ----
    gs_sl = gs[1, 1].subgridspec(nrows=3, ncols=6, wspace=0.35, hspace=0.95)

    def slider_ax(r, c, colspan=2, pad=0.02):
        ax = fig.add_subplot(gs_sl[r, c:c + colspan])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    # Slider definitions (valinit MUST be keyword)
    s_m = Slider(slider_ax(0, 0), "m", 0.5, 50.0, valinit=p0["m"], valfmt="%.2f")
    s_k = Slider(slider_ax(0, 2), "k", 0.1, 20.0, valinit=p0["k"], valfmt="%.2f")
    s_gamma = Slider(slider_ax(0, 4), "γ", 0.0, 5.0, valinit=p0["gamma"], valfmt="%.3f")

    s_T = Slider(slider_ax(1, 0), "T", 0.05, 5.0, valinit=p0["T"], valfmt="%.2f")
    s_tmax = Slider(slider_ax(1, 2), "tmax (s)", 5.0, 200.0, valinit=p0["tmax"], valfmt="%.1f")
    s_dt = Slider(slider_ax(1, 4), "dt (s)", 0.005, 0.10, valinit=p0["dt"], valfmt="%.3f")

    s_mtm = Slider(slider_ax(2, 0), "multi-tau m", 8, 32, valinit=p0["mt_m"], valfmt="%.0f")

    sliders = (s_m, s_k, s_gamma, s_T, s_tmax, s_dt, s_mtm)
    for s in sliders:
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # Buttons anchored at the very bottom (outside gridspec, but not over the plots)
    ax_reset = fig.add_axes([0.055, 0.012, 0.09, 0.045])
    ax_save = fig.add_axes([0.150, 0.012, 0.12, 0.045])
    b_reset = Button(ax_reset, "Reset")
    b_save = Button(ax_save, "Save PNG")

    busy = {"flag": False}

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            # Guardrail on Nt (Nt×Nt map cost)
            tmax = float(s_tmax.val)
            dt = float(s_dt.val)
            Nt = int(np.floor(tmax / dt)) + 1
            if Nt > 3500:
                # keep dt fixed, shrink tmax
                tmax = 3500 * dt
                Nt = int(np.floor(tmax / dt)) + 1

            t, x = simulate_ensemble_langevin(
                m=float(s_m.val),
                k=float(s_k.val),
                gamma=float(s_gamma.val),
                kB=float(p0["kB"]),
                T=float(s_T.val),
                tmax=tmax,
                dt=dt,
                seed=int(p0["seed"]),
            )

            # LEFT: multi-tau
            taus, g = multitau_autocorr_ensemble(x, dt=dt, m=int(round(s_mtm.val)))
            # drop tau=0 for log-x
            if len(taus) > 1 and taus[0] == 0.0:
                taus_plot = taus[1:]
                g_plot = g[1:]
            else:
                taus_plot = taus
                g_plot = g

            line_g2.set_data(taus_plot, g_plot)
            if len(taus_plot) > 1:
                ax_g2.set_xlim(max(taus_plot.min(), 1e-6), taus_plot.max())
            y0, y1 = float(np.min(g_plot)), float(np.max(g_plot))
            pad = 0.08 * (y1 - y0 + 1e-12)
            ax_g2.set_ylim(y0 - pad, y1 + pad)

            omega0 = np.sqrt(float(s_k.val) / float(s_m.val))
            ax_g2.set_title(f"Multi-tau autocorrelation (ω0={omega0:.3f} rad/s)")

            # RIGHT: Cnorm map (square aspect)
            Cn = two_time_cnorm(x)
            im.set_data(Cn)
            im.set_extent([t[0], t[-1], t[0], t[-1]])
            im.set_clim(-1, 1)
            ax_map.set_xlim(t[0], t[-1])
            ax_map.set_ylim(t[0], t[-1])
            ax_map.set_aspect("equal", adjustable="box")

            fig.canvas.draw_idle()
        finally:
            busy["flag"] = False

    for s in sliders:
        s.on_changed(update)

    def on_reset(_event):
        for s in sliders:
            s.reset()
        update()

    def on_save(_event):
        fname = "langevin_fixedNtraj_g2_and_cnorm.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    main()