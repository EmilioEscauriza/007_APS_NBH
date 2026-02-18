"""
Interactive "pendulum-as-a-signal" XPCS-like toy model

Model:
  θ_p(t) = θ0 * exp(-γ t) * cos(ω t + φ_p) + η_p(t)
  I_p(t) = I0 * [1 + a * θ_p(t)]

Key point for XPCS relevance:
- TTC is computed with a pixel/ROI ensemble average (p = "pixels"):

    C(t1,t2) = < I_p(t1) I_p(t2) >_p  / ( <I_p(t1)>_p <I_p(t2)>_p )

This is the standard two-time (two-time g2) normalization used in XPCS, it largely removes
frame-to-frame mean intensity changes and highlights speckle-like fluctuations.

Layout:
  TOP LEFT  : g2(τ) = <I(t) I(t+τ)>_{p,t} / <I>^2
  TOP RIGHT : TTC map C(t1,t2) (square)
  BOTTOM LEFT  : equations box
  BOTTOM RIGHT : sliders + buttons

Requires:
  pip install PyQt6

Run:
  python pendulum_xpcs_ttc_gui.py
"""

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def simulate_theta_pixels(
    t: np.ndarray,
    theta0: float,
    omega: float,
    gamma: float,
    phi0: float,
    phase_spread: float,
    noise_sigma: float,
    Npix: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate θ_p(t) for p=1..Npix.

    phase_spread controls heterogeneity across pixels:
      φ_p = φ0 + uniform(-phase_spread/2, +phase_spread/2)

    If phase_spread ~ 0, all pixels oscillate in phase -> TTC can look more chequer-like.
    Larger phase_spread makes the pixel ensemble behave more like XPCS ROIs (stripe-like TTC).
    """
    rng = np.random.default_rng(int(seed))

    # per-pixel phases
    phi_p = phi0 + rng.uniform(-0.5 * phase_spread, 0.5 * phase_spread, size=Npix)

    # deterministic part: (Npix, Nt)
    env = theta0 * np.exp(-gamma * t)  # (Nt,)
    theta_det = env[None, :] * np.cos(omega * t[None, :] + phi_p[:, None])

    if noise_sigma > 0:
        theta_det = theta_det + rng.normal(0.0, noise_sigma, size=theta_det.shape)

    return theta_det


def make_intensity_from_theta(theta_pix: np.ndarray, I0: float, a: float) -> np.ndarray:
    """
    I_p(t) = I0 * [1 + a * θ_p(t)]
    Clamp to positive.
    """
    I = I0 * (1.0 + a * theta_pix)
    return np.clip(I, 1e-12, None)


def ttc_xpcs_like(I_pix: np.ndarray) -> np.ndarray:
    """
    XPCS-relevant two-time normalization using pixel (ROI) ensemble:

      C(t1,t2) = < I_p(t1) I_p(t2) >_p / ( <I_p(t1)>_p <I_p(t2)>_p )

    I_pix: shape (Npix, Nt)
    """
    Npix = I_pix.shape[0]
    num = (I_pix.T @ I_pix) / float(Npix)  # (Nt, Nt)
    mu = np.mean(I_pix, axis=0)            # (Nt,)
    den = mu[:, None] * mu[None, :]
    return num / (den + 1e-18)


def g2_from_pixels(I_pix: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    g2(k) = < I_p(t) I_p(t+k) >_{p,t} / <I>^2
    where <I> is mean over p,t.
    """
    Npix, Nt = I_pix.shape
    max_lag = max(1, min(int(max_lag), Nt - 2))
    Imean = float(np.mean(I_pix))

    g2 = np.empty(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        g2[k] = np.mean(I_pix[:, : Nt - k] * I_pix[:, k:Nt]) / (Imean**2)

    return np.arange(max_lag + 1), g2


def main():
    # Fixed sampling for responsiveness
    dt = 0.02
    T_total = 30.0
    t = np.arange(0.0, T_total, dt)

    # Defaults
    p0 = dict(
        theta0=0.08,
        f=0.60,               # Hz
        gamma=0.03,           # 1/s
        phi0=0.40,            # rad
        phase_spread=np.pi,   # rad (0..2π)
        noise_sigma=0.01,
        I0=1.0,
        a=1.0,
        Npix=256,             # "ROI size" (ensemble)
        seed=1,
        ttc_window_seconds=20.0,
        max_lag_seconds=10.0,
    )

    # ---- Layout: 2x2 grid, bottom-left text, bottom-right sliders ----
    fig = plt.figure(figsize=(16.0, 8.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 0.78],
        width_ratios=[1.0, 1.0],
        left=0.05, right=0.985, top=0.93, bottom=0.07,
        wspace=0.25, hspace=0.30,
    )

    ax_g2 = fig.add_subplot(gs[0, 0])
    ax_ttc = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.set_axis_off()
    ax_slider_area = fig.add_subplot(gs[1, 1])
    ax_slider_area.set_axis_off()

    # ---- Equations box (in place, not overlaid) ----
    eq_text = (
        r"$\theta_p(t)=\theta_0 e^{-\gamma t}\cos(\omega t+\phi_p)+\eta_p(t)$" "\n"
        r"$I_p(t)=I_0[1+a\,\theta_p(t)]$" "\n\n"
        r"$C(t_1,t_2)=\frac{\langle I_p(t_1)I_p(t_2)\rangle_p}{\langle I_p(t_1)\rangle_p\langle I_p(t_2)\rangle_p}$" "\n"
        r"$g_2(\tau)=\frac{\langle I_p(t)I_p(t+\tau)\rangle_{p,t}}{\langle I\rangle^2}$" "\n\n"
        r"Here p indexes ROI pixels (ensemble)."
    )
    ax_text.text(
        0.02, 0.98, eq_text,
        va="top", ha="left",
        fontsize=11,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", alpha=0.95),
        transform=ax_text.transAxes,
    )
    ax_text.set_title("Toy model and XPCS-style TTC definition", loc="left", pad=6)

    # ---- TOP LEFT: g2 ----
    (line_g2,) = ax_g2.plot([], [], lw=2)
    ax_g2.set_title(r"$g_2(\tau)$ from ROI ensemble")
    ax_g2.set_xlabel(r"lag $\tau$ (s)")
    ax_g2.set_ylabel(r"$g_2(\tau)$")
    ax_g2.grid(True, alpha=0.3)

    # ---- TOP RIGHT: TTC (square) ----
    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    im = ax_ttc.imshow(
        np.zeros((10, 10)),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=[0, 1, 0, 1],
        aspect="equal",
    )
    ax_ttc.set_title(r"Two-time map (XPCS-style): $C(t_1,t_2)$")
    ax_ttc.set_xlabel(r"$t_2$ (s)")
    ax_ttc.set_ylabel(r"$t_1$ (s)")
    ax_ttc.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax_ttc, fraction=0.046, pad=0.04)
    cbar.set_label("C")

    # ---- Sliders inside bottom-right cell ----
    gs_sl = gs[1, 1].subgridspec(nrows=4, ncols=6, wspace=0.35, hspace=0.95)

    def slider_ax(r, c, colspan=2, pad=0.02):
        ax = fig.add_subplot(gs_sl[r, c:c + colspan])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    s_theta0 = Slider(slider_ax(0, 0), "θ0", 0.0, 0.5, valinit=p0["theta0"], valfmt="%.3f")
    s_f      = Slider(slider_ax(0, 2), "f (Hz)", 0.05, 2.0, valinit=p0["f"], valfmt="%.3f")
    s_gamma  = Slider(slider_ax(0, 4), "γ (1/s)", 0.0, 0.3, valinit=p0["gamma"], valfmt="%.3f")

    s_phi0   = Slider(slider_ax(1, 0), "φ0", 0.0, 2*np.pi, valinit=p0["phi0"], valfmt="%.2f")
    s_spread = Slider(slider_ax(1, 2), "phase spread", 0.0, 2*np.pi, valinit=p0["phase_spread"], valfmt="%.2f")
    s_noise  = Slider(slider_ax(1, 4), "noise σ", 0.0, 0.2, valinit=p0["noise_sigma"], valfmt="%.3f")

    s_I0     = Slider(slider_ax(2, 0), "I0", 0.1, 10.0, valinit=p0["I0"], valfmt="%.2f")
    s_a      = Slider(slider_ax(2, 2), "a", 0.0, 5.0, valinit=p0["a"], valfmt="%.2f")
    s_Npix   = Slider(slider_ax(2, 4), "ROI pixels", 16, 1024, valinit=p0["Npix"], valfmt="%.0f")

    s_ttcwin = Slider(slider_ax(3, 0), "TTC win (s)", 2.0, T_total, valinit=p0["ttc_window_seconds"], valfmt="%.1f")
    s_lagmax = Slider(slider_ax(3, 2), "max lag (s)", 0.5, T_total, valinit=p0["max_lag_seconds"], valfmt="%.1f")
    s_seed   = Slider(slider_ax(3, 4), "seed", 0, 999, valinit=p0["seed"], valfmt="%.0f")

    sliders = (s_theta0, s_f, s_gamma, s_phi0, s_spread, s_noise, s_I0, s_a, s_Npix, s_ttcwin, s_lagmax, s_seed)
    for s in sliders:
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # Buttons
    ax_reset = fig.add_axes([0.055, 0.012, 0.09, 0.045])
    ax_save  = fig.add_axes([0.150, 0.012, 0.12, 0.045])
    b_reset = Button(ax_reset, "Reset")
    b_save  = Button(ax_save, "Save PNG")

    busy = {"flag": False}

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            theta0 = float(s_theta0.val)
            omega = 2 * np.pi * float(s_f.val)
            gamma = float(s_gamma.val)
            phi0 = float(s_phi0.val)
            phase_spread = float(s_spread.val)
            noise_sigma = float(s_noise.val)
            I0 = float(s_I0.val)
            a = float(s_a.val)
            Npix = int(round(s_Npix.val))
            seed = int(round(s_seed.val))

            ttc_window_seconds = float(s_ttcwin.val)
            max_lag_seconds = float(s_lagmax.val)

            # window sizes
            Nw = int(ttc_window_seconds / dt)
            Nw = max(10, min(Nw, len(t)))
            tw = t[:Nw]

            max_lag = int(max_lag_seconds / dt)
            max_lag = max(1, min(max_lag, len(t) - 2))

            # simulate ROI pixels
            theta_pix = simulate_theta_pixels(
                t=t,
                theta0=theta0,
                omega=omega,
                gamma=gamma,
                phi0=phi0,
                phase_spread=phase_spread,
                noise_sigma=noise_sigma,
                Npix=Npix,
                seed=seed,
            )
            I_pix = make_intensity_from_theta(theta_pix, I0=I0, a=a)

            # g2
            lags, g2 = g2_from_pixels(I_pix, max_lag=max_lag)
            tau = lags * dt
            line_g2.set_data(tau, g2)
            ax_g2.set_xlim(tau[0], tau[-1])
            y0, y1 = float(np.min(g2)), float(np.max(g2))
            pad = 0.08 * (y1 - y0 + 1e-12)
            ax_g2.set_ylim(y0 - pad, y1 + pad)

            # TTC (XPCS-style, ROI ensemble)
            TTC = ttc_xpcs_like(I_pix[:, :Nw])

            im.set_data(TTC)
            im.set_extent([tw[0], tw[-1], tw[0], tw[-1]])
            ax_ttc.set_xlim(tw[0], tw[-1])
            ax_ttc.set_ylim(tw[0], tw[-1])
            ax_ttc.set_aspect("equal", adjustable="box")

            vmin = np.percentile(TTC, 1)
            vmax = np.percentile(TTC, 99)
            if vmax <= vmin:
                vmax = vmin + 1e-6
            im.set_clim(vmin, vmax)

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
        fname = "pendulum_xpcs_like_ttc.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    main()