"""
Interactive damped Langevin oscillator with mixed TTC (stripes + chequer).

Model (Euler–Maruyama):
    dθ = v dt
    dv = (-γ v - ω0^2 θ) dt + σ dW

Plots (top):
  1) Time trace: I(t) from θ(t)
  2) g2(τ) = < I(t) I(t+τ) > / <I>^2
  3) Two-time correlation map (TTC):
       TTC = mix * TTC_stripes + (1-mix) * TTC_chequer
       - TTC_stripes: depends on lag |t2-t1| -> diagonal stripes
       - TTC_chequer: outer product I(t1)I(t2) -> chequer texture
       - mix slider blends them so you can reproduce "stripes + chequer"

Bottom:
  Left  : equations box
  Right : sliders

Requires:
  pip install PyQt6
Run:
  python langevin_gui.py
"""

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")  # requires PyQt6
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# -------------------------
# Langevin simulator
# -------------------------

def simulate_langevin_theta(
    t: np.ndarray,
    theta0: float,
    v0: float,
    f0_hz: float,
    gamma: float,      # 1/s
    sigma: float,      # noise strength in dv equation
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euler–Maruyama for second-order Langevin oscillator:
        dθ = v dt
        dv = (-γ v - ω0^2 θ) dt + σ dW
    """
    dt = float(t[1] - t[0])
    omega0 = 2 * np.pi * f0_hz
    rng = np.random.default_rng(int(seed))

    N = len(t)
    theta = np.empty(N, dtype=float)
    v = np.empty(N, dtype=float)
    theta[0] = theta0
    v[0] = v0

    dW = rng.normal(0.0, np.sqrt(dt), size=N - 1)

    for n in range(N - 1):
        theta[n + 1] = theta[n] + v[n] * dt
        v[n + 1] = v[n] + (-gamma * v[n] - (omega0**2) * theta[n]) * dt + sigma * dW[n]

    return theta, v


def make_intensity_from_theta(theta: np.ndarray, a: float) -> np.ndarray:
    """
    Map θ(t) to positive intensity-like I(t).
    For Langevin, RMS normalization is more stable than max normalization.
    """
    theta_norm = theta / (np.std(theta) + 1e-12)
    I = 1.0 + a * theta_norm
    return np.clip(I, 1e-6, None)


# -------------------------
# Correlations
# -------------------------

def g2_from_intensity(I: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """g2(k)=<I(t)I(t+k)>/<I>^2 for k>=0."""
    Imean = np.mean(I)
    N = len(I)
    max_lag = max(1, min(int(max_lag), N - 2))

    g2 = np.empty(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        g2[k] = np.mean(I[: N - k] * I[k:N]) / (Imean**2)

    return np.arange(max_lag + 1), g2


def ttc_stripes_from_lagcorr(I: np.ndarray) -> np.ndarray:
    """
    Stripe-forming TTC based on lag-correlation:
      C(k)=<I(t)I(t+k)>/<I>^2
      TTC[i,j]=C(|j-i|)
    """
    Imean = np.mean(I)
    N = len(I)

    C = np.empty(N, dtype=float)
    for k in range(N):
        C[k] = np.mean(I[: N - k] * I[k:N]) / (Imean**2)

    idx = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    return C[idx]


def ttc_chequer_outer(I: np.ndarray) -> np.ndarray:
    """Chequer-ish TTC from instantaneous product."""
    Imean = np.mean(I)
    return np.outer(I, I) / (Imean**2)


# -------------------------
# GUI
# -------------------------

def main():
    # Sampling (keep fixed for responsiveness)
    dt = 0.01
    T_total = 50.0
    t = np.arange(0.0, T_total, dt)

    # Initial parameters
    p0 = dict(
        theta0=0.30,
        v0=0.00,
        f0=0.60,        # Hz
        gamma=0.05,     # 1/s  (lower => more visible stripes)
        sigma=0.40,     # noise strength
        seed=1,
        a=0.55,         # intensity mapping strength (too large can saturate)
        mix=0.75,       # 1=stripes, 0=chequer
        trace_seconds=18.0,
        max_lag_seconds=10.0,
        ttc_window_seconds=20.0,
    )

    # ---- Layout ----
    fig = plt.figure(figsize=(16.6, 8.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 0.80],
        left=0.05, right=0.985, top=0.93, bottom=0.07,
        wspace=0.28, hspace=0.30,
    )

    ax_trace = fig.add_subplot(gs[0, 0])
    ax_g2 = fig.add_subplot(gs[0, 1])
    ax_ttc = fig.add_subplot(gs[0, 2])

    # Bottom row split: equations (left) + sliders (right spanning 2 cols)
    gs_bottom = gs[1, :].subgridspec(nrows=1, ncols=3, width_ratios=[1.10, 1.0, 1.0], wspace=0.25)
    ax_eq = fig.add_subplot(gs_bottom[0, 0])
    ax_sliders_container = fig.add_subplot(gs_bottom[0, 1:])
    ax_sliders_container.axis("off")

    # Slider grid inside the right bottom region
    gs_sl = gs_bottom[0, 1:].subgridspec(nrows=5, ncols=6, wspace=0.30, hspace=0.70)

    def slider_ax(r, c, colspan=2, pad=0.02):
        """Create slider axis and slightly shrink laterally to avoid label/value collision."""
        ax = fig.add_subplot(gs_sl[r, c:c + colspan])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    # ---- Equations box ----
    ax_eq.set_axis_off()
    eq_text = (
        r"$d\theta = v\,dt$" "\n"
        r"$dv = (-\gamma v - \omega_0^2\theta)\,dt + \sigma\,dW$" "\n\n"
        r"$I(t)=1+a\,\theta(t)/\mathrm{std}(\theta)$" "\n\n"
        r"$g_2(\tau)=\langle I(t)I(t+\tau)\rangle/\langle I\rangle^2$" "\n\n"
        r"$\mathrm{TTC}_{\mathrm{stripe}}(t_1,t_2)=g_2(|t_2-t_1|)$" "\n"
        r"$\mathrm{TTC}_{\mathrm{cheq}}(t_1,t_2)=I(t_1)I(t_2)/\langle I\rangle^2$" "\n\n"
        r"$\mathrm{TTC} = \mathrm{mix}\cdot \mathrm{stripe} + (1-\mathrm{mix})\cdot \mathrm{cheq}$" "\n\n"
        r"$\omega_0=2\pi f_0$"
    )
    ax_eq.text(
        0.02, 0.98, eq_text,
        va="top", ha="left",
        fontsize=11,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        transform=ax_eq.transAxes,
    )
    ax_eq.set_title("Langevin model + TTC construction", loc="left", pad=6)

    # ---- Artists ----
    line_I, = ax_trace.plot([], [], linewidth=2, label=r"$I(t)$")
    ax_trace.set_title("Time trace (single trajectory)")
    ax_trace.set_xlabel("time (s)")
    ax_trace.set_ylabel("intensity (a.u.)")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(frameon=False)

    line_g2, = ax_g2.plot([], [], linewidth=2)
    ax_g2.set_title(r"$g_2(\tau)$")
    ax_g2.set_xlabel(r"lag $\tau$ (s)")
    ax_g2.set_ylabel(r"$g_2(\tau)$")
    ax_g2.grid(True, alpha=0.3)

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")
    im = ax_ttc.imshow(
        np.zeros((10, 10)),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        extent=[0, 1, 0, 1],
    )
    ax_ttc.set_aspect("equal", adjustable="box")
    ax_ttc.set_title("Two-time map (stripes + chequer)")
    ax_ttc.set_xlabel(r"$t_2$ (s)")
    ax_ttc.set_ylabel(r"$t_1$ (s)")
    cbar = fig.colorbar(im, ax=ax_ttc, fraction=0.046, pad=0.04)
    cbar.set_label("normalized corr.")

    # ---- Sliders ----
    s_f0 = Slider(slider_ax(0, 0), "f₀ (Hz)", 0.05, 2.00, valinit=p0["f0"], valfmt="%.3f")
    s_gamma = Slider(slider_ax(0, 2), "γ (1/s)", 0.00, 2.00, valinit=p0["gamma"], valfmt="%.3f")
    s_sigma = Slider(slider_ax(0, 4), "σ", 0.00, 3.00, valinit=p0["sigma"], valfmt="%.3f")

    s_theta0 = Slider(slider_ax(1, 0), "θ₀", -1.00, 1.00, valinit=p0["theta0"], valfmt="%.3f")
    s_v0 = Slider(slider_ax(1, 2), "v₀", -3.00, 3.00, valinit=p0["v0"], valfmt="%.3f")
    s_a = Slider(slider_ax(1, 4), "a", 0.00, 1.50, valinit=p0["a"], valfmt="%.2f")

    s_mix = Slider(slider_ax(2, 0), "mix", 0.0, 1.0, valinit=p0["mix"], valfmt="%.2f")
    s_seed = Slider(slider_ax(2, 2), "seed", 0, 999, valinit=p0["seed"], valfmt="%.0f")

    s_trace = Slider(slider_ax(3, 0), "trace (s)", 5.0, float(T_total), valinit=p0["trace_seconds"], valfmt="%.1f")
    s_lag = Slider(slider_ax(3, 2), "max lag (s)", 0.5, float(T_total) / 2, valinit=p0["max_lag_seconds"], valfmt="%.1f")
    s_ttc = Slider(slider_ax(3, 4), "TTC (s)", 2.0, float(T_total), valinit=p0["ttc_window_seconds"], valfmt="%.1f")

    for s in (s_f0, s_gamma, s_sigma, s_theta0, s_v0, s_a, s_mix, s_seed, s_trace, s_lag, s_ttc):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # Buttons
    ax_reset = fig.add_axes([0.055, 0.012, 0.09, 0.045])
    ax_save = fig.add_axes([0.150, 0.012, 0.10, 0.045])
    b_reset = Button(ax_reset, "Reset")
    b_save = Button(ax_save, "Save PNG")

    busy = {"flag": False}

    def compute_all(p):
        theta, v = simulate_langevin_theta(
            t=t,
            theta0=p["theta0"],
            v0=p["v0"],
            f0_hz=p["f0"],
            gamma=p["gamma"],
            sigma=p["sigma"],
            seed=p["seed"],
        )
        I = make_intensity_from_theta(theta, p["a"])

        # Trace window
        Nt = int(p["trace_seconds"] / dt)
        Nt = max(8, min(Nt, len(t)))
        tt = t[:Nt]
        It = I[:Nt]

        # g2
        max_lag = int(p["max_lag_seconds"] / dt)
        lags, g2 = g2_from_intensity(I, max_lag=max_lag)
        tau = lags * dt

        # TTC window + mixed TTC
        Nw = int(p["ttc_window_seconds"] / dt)
        Nw = max(8, min(Nw, len(t)))
        tw = t[:Nw]
        Iw = I[:Nw]

        TTC_stripe = ttc_stripes_from_lagcorr(Iw)
        TTC_cheq = ttc_chequer_outer(Iw)
        TTC = p["mix"] * TTC_stripe + (1.0 - p["mix"]) * TTC_cheq

        return tt, It, tau, g2, tw, TTC

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            p = dict(
                theta0=float(s_theta0.val),
                v0=float(s_v0.val),
                f0=float(s_f0.val),
                gamma=float(s_gamma.val),
                sigma=float(s_sigma.val),
                seed=int(round(s_seed.val)),
                a=float(s_a.val),
                mix=float(s_mix.val),
                trace_seconds=float(s_trace.val),
                max_lag_seconds=float(s_lag.val),
                ttc_window_seconds=float(s_ttc.val),
            )

            tt, It, tau, g2, tw, TTC = compute_all(p)

            # Trace
            line_I.set_data(tt, It)
            ax_trace.set_xlim(tt[0], tt[-1])
            ymin, ymax = It.min(), It.max()
            pad = 0.05 * (ymax - ymin + 1e-12)
            ax_trace.set_ylim(ymin - pad, ymax + pad)

            # g2
            line_g2.set_data(tau, g2)
            ax_g2.set_xlim(tau[0], tau[-1])
            y2min, y2max = g2.min(), g2.max()
            pad2 = 0.08 * (y2max - y2min + 1e-12)
            ax_g2.set_ylim(y2min - pad2, y2max + pad2)

            # TTC
            im.set_data(TTC)
            im.set_extent([tw[0], tw[-1], tw[0], tw[-1]])

            vmin = np.percentile(TTC, 1)
            vmax = np.percentile(TTC, 99)
            if vmax <= vmin:
                vmax = vmin + 1e-6
            im.set_clim(vmin, vmax)

            fig.canvas.draw_idle()
        finally:
            busy["flag"] = False

    # Slider callbacks
    for s in (s_f0, s_gamma, s_sigma, s_theta0, s_v0, s_a, s_mix, s_seed, s_trace, s_lag, s_ttc):
        s.on_changed(update)

    def on_reset(_event):
        for s in (s_f0, s_gamma, s_sigma, s_theta0, s_v0, s_a, s_mix, s_seed, s_trace, s_lag, s_ttc):
            s.reset()
        update()

    def on_save(_event):
        fname = "langevin_gui_mixed.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    main()