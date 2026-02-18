"""
Interactive version of the original single-pendulum correlation figure.

LEFT  : g2(τ) = <I(t) I(t+τ)> / <I>^2
RIGHT : TTC(t1,t2) = I(t1) I(t2) / <I>^2  (outer-product map)

This is a simple "pendulum-as-a-signal" demo, not a full XPCS simulator.

Requires (recommended on macOS for widgets):
  pip install PyQt6

Run:
  python pendulum_gui_single.py
"""

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")  # requires PyQt6; avoids macosx/tk widget issues
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# -------------------------
# Signal + correlations
# -------------------------

def make_pendulum_signal(
    t: np.ndarray,
    theta0: float,
    omega: float,          # rad/s
    gamma: float,          # 1/s
    phi: float,
    noise_sigma: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    theta = theta0 * np.exp(-gamma * t) * np.cos(omega * t + phi)
    if noise_sigma > 0:
        theta = theta + rng.normal(0.0, noise_sigma, size=t.shape)
    return theta


def make_intensity_from_theta(theta: np.ndarray, a: float) -> np.ndarray:
    theta_norm = theta / (np.max(np.abs(theta)) + 1e-12)
    I = 1.0 + a * theta_norm
    return np.clip(I, 1e-6, None)


def g2_from_intensity(I: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    Imean = np.mean(I)
    N = len(I)
    max_lag = max(1, min(int(max_lag), N - 2))
    g2 = np.empty(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        g2[k] = np.mean(I[: N - k] * I[k:N]) / (Imean**2)
    return np.arange(max_lag + 1), g2


def two_time_correlation(I: np.ndarray) -> np.ndarray:
    Imean = np.mean(I)
    return np.outer(I, I) / (Imean**2)


# -------------------------
# GUI
# -------------------------

def main():
    # Fixed sampling for responsiveness
    dt = 0.02
    T_total = 30.0
    t = np.arange(0.0, T_total, dt)

    # Initial params
    p0 = dict(
        theta0=0.35,
        f=0.60,              # Hz (we slider frequency, convert to omega)
        gamma=0.03,          # 1/s
        phi=0.40,            # rad
        noise_sigma=0.01,
        seed=1,
        intensity_a=0.90,
        max_lag_seconds=10.0,
        ttc_window_seconds=20.0,
    )

    # ---- Layout: top plots, bottom controls ----
    fig = plt.figure(figsize=(14.8, 7.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 0.70],
        left=0.06, right=0.985, top=0.93, bottom=0.08,
        wspace=0.25, hspace=0.32,
    )

    ax_g2 = fig.add_subplot(gs[0, 0])
    ax_ttc = fig.add_subplot(gs[0, 1])

    # Bottom: equations box (left) + sliders (right)
    gs_bottom = gs[1, :].subgridspec(nrows=1, ncols=3, width_ratios=[1.05, 1.0, 1.0], wspace=0.25)
    ax_eq = fig.add_subplot(gs_bottom[0, 0])
    ax_sliders_container = fig.add_subplot(gs_bottom[0, 1:])
    ax_sliders_container.axis("off")

    gs_sl = gs_bottom[0, 1:].subgridspec(nrows=4, ncols=6, wspace=0.30, hspace=0.75)

    def slider_ax(r, c, colspan=2, pad=0.02):
        ax = fig.add_subplot(gs_sl[r, c:c + colspan])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    # ---- Equations box ----
    ax_eq.set_axis_off()
    eq_text = (
        r"$\theta(t)=\theta_0 e^{-\gamma t}\cos(\omega t+\phi)$" "\n\n"
        r"$I(t)=1+a\,\frac{\theta(t)}{\max|\theta|}$" "\n\n"
        r"$g_2(\tau)=\frac{\langle I(t)I(t+\tau)\rangle}{\langle I\rangle^2}$" "\n\n"
        r"$\mathrm{TTC}(t_1,t_2)=\frac{I(t_1)I(t_2)}{\langle I\rangle^2}$" "\n\n"
        r"$\omega=2\pi f$"
    )
    ax_eq.text(
        0.02, 0.98, eq_text,
        va="top", ha="left",
        fontsize=11,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        transform=ax_eq.transAxes,
    )
    ax_eq.set_title("Equations", loc="left", pad=6)

    # ---- Plot artists ----
    (line_g2,) = ax_g2.plot([], [], linewidth=2)
    ax_g2.set_title(r"$g_2(\tau)=\langle I(t)I(t+\tau)\rangle/\langle I\rangle^2$")
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
    ax_ttc.set_title(r"TTC: $I(t_1)I(t_2)/\langle I\rangle^2$")
    ax_ttc.set_xlabel(r"$t_2$ (s)")
    ax_ttc.set_ylabel(r"$t_1$ (s)")
    ax_ttc.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax_ttc, fraction=0.046, pad=0.04)
    cbar.set_label("normalized correlation")

    # ---- Sliders ----
    s_f = Slider(slider_ax(0, 0), "f (Hz)", 0.05, 2.0, valinit=p0["f"], valfmt="%.3f")
    s_gamma = Slider(slider_ax(0, 2), "γ (1/s)", 0.0, 0.3, valinit=p0["gamma"], valfmt="%.3f")
    s_phi = Slider(slider_ax(0, 4), "φ (rad)", 0.0, 2*np.pi, valinit=p0["phi"], valfmt="%.2f")

    s_theta0 = Slider(slider_ax(1, 0), "θ₀", 0.01, 1.0, valinit=p0["theta0"], valfmt="%.3f")
    s_noise = Slider(slider_ax(1, 2), "noise σ", 0.0, 0.2, valinit=p0["noise_sigma"], valfmt="%.3f")
    s_seed = Slider(slider_ax(1, 4), "seed", 0, 999, valinit=p0["seed"], valfmt="%.0f")

    s_a = Slider(slider_ax(2, 0), "a", 0.0, 1.5, valinit=p0["intensity_a"], valfmt="%.2f")
    s_lag = Slider(slider_ax(2, 2), "max lag (s)", 1.0, T_total, valinit=p0["max_lag_seconds"], valfmt="%.1f")
    s_ttc = Slider(slider_ax(2, 4), "TTC (s)", 2.0, T_total, valinit=p0["ttc_window_seconds"], valfmt="%.1f")

    for s in (s_f, s_gamma, s_phi, s_theta0, s_noise, s_seed, s_a, s_lag, s_ttc):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # Buttons
    ax_reset = fig.add_axes([0.06, 0.012, 0.09, 0.045])
    ax_save = fig.add_axes([0.155, 0.012, 0.11, 0.045])
    b_reset = Button(ax_reset, "Reset")
    b_save = Button(ax_save, "Save PNG")

    busy = {"flag": False}

    def compute(p):
        omega = 2 * np.pi * p["f"]
        theta = make_pendulum_signal(
            t,
            theta0=p["theta0"],
            omega=omega,
            gamma=p["gamma"],
            phi=p["phi"],
            noise_sigma=p["noise_sigma"],
            seed=p["seed"],
        )
        I = make_intensity_from_theta(theta, a=p["intensity_a"])

        max_lag = int(p["max_lag_seconds"] / dt)
        lags, g2 = g2_from_intensity(I, max_lag=max_lag)
        tau = lags * dt

        Nw = int(p["ttc_window_seconds"] / dt)
        Nw = max(2, min(Nw, len(I)))
        Iw = I[:Nw]
        tw = t[:Nw]
        TTC = two_time_correlation(Iw)

        return tau, g2, tw, TTC

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            p = dict(
                theta0=float(s_theta0.val),
                f=float(s_f.val),
                gamma=float(s_gamma.val),
                phi=float(s_phi.val),
                noise_sigma=float(s_noise.val),
                seed=int(round(s_seed.val)),
                intensity_a=float(s_a.val),
                max_lag_seconds=float(s_lag.val),
                ttc_window_seconds=float(s_ttc.val),
            )

            tau, g2, tw, TTC = compute(p)

            # g2
            line_g2.set_data(tau, g2)
            ax_g2.set_xlim(tau[0], tau[-1])
            y0, y1 = float(g2.min()), float(g2.max())
            pad = 0.08 * (y1 - y0 + 1e-12)
            ax_g2.set_ylim(y0 - pad, y1 + pad)

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

    for s in (s_f, s_gamma, s_phi, s_theta0, s_noise, s_seed, s_a, s_lag, s_ttc):
        s.on_changed(update)

    def on_reset(_event):
        for s in (s_f, s_gamma, s_phi, s_theta0, s_noise, s_seed, s_a, s_lag, s_ttc):
            s.reset()
        update()

    def on_save(_event):
        fname = "pendulum_correlations_gui.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    main()