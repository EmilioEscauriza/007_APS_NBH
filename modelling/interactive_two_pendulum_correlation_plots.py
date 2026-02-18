"""
Interactive 2-pendulum correlation GUI with sliders + equations box.

Top row:
  (1) Time traces I1(t), I2(t)
  (2) Cross-correlation g2_12(τ)
  (3) Two-time cross-correlation map (stripe-forming)

Bottom row:
  Left  : equations box
  Right : sliders
"""

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")  # requires: pip install PyQt6
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# -------------------------
# Core model + correlations
# -------------------------

def make_pendulum_signal(t, theta0, omega, gamma, phi, noise_sigma, seed):
    rng = np.random.default_rng(int(seed))
    theta = theta0 * np.exp(-gamma * t) * np.cos(omega * t + phi)
    if noise_sigma > 0:
        theta = theta + rng.normal(0.0, noise_sigma, size=t.shape)
    return theta


def make_intensity_from_theta(theta, a):
    theta_norm = theta / (np.max(np.abs(theta)) + 1e-12)
    I = 1.0 + a * theta_norm
    return np.clip(I, 1e-6, None)


def cross_g2(I1, I2, max_lag):
    """g2_12(k) = <I1(t) I2(t+k)> / (<I1><I2>) for k in [-max_lag, +max_lag]."""
    I1m = np.mean(I1)
    I2m = np.mean(I2)
    N = min(len(I1), len(I2))
    I1 = I1[:N]
    I2 = I2[:N]

    lags = np.arange(-max_lag, max_lag + 1)
    g12 = np.empty_like(lags, dtype=float)

    for idx, k in enumerate(lags):
        if k >= 0:
            a = I1[: N - k]
            b = I2[k:N]
        else:
            kk = -k
            a = I1[kk:N]
            b = I2[: N - kk]
        g12[idx] = np.mean(a * b) / (I1m * I2m)

    return lags, g12


def two_time_cross_from_lagcorr(I1, I2):
    """
    Stripe-forming TTC cross map:
      C12(d)=<I1(t)I2(t+d)>/(<I1><I2>)  for d=j-i
      TTC[i,j]=C12(j-i)  -> constant along diagonals
    """
    N = min(len(I1), len(I2))
    I1 = I1[:N]
    I2 = I2[:N]
    I1m = np.mean(I1)
    I2m = np.mean(I2)

    lags = np.arange(-(N - 1), N)  # d = j - i
    C = np.empty_like(lags, dtype=float)

    for idx, d in enumerate(lags):
        if d >= 0:
            a = I1[: N - d]
            b = I2[d:N]
        else:
            dd = -d
            a = I1[dd:N]
            b = I2[: N - dd]
        C[idx] = np.mean(a * b) / (I1m * I2m)

    dmat = np.subtract.outer(np.arange(N), np.arange(N))  # i - j
    dmat = -dmat  # j - i
    TTC = C[dmat + (N - 1)]
    return TTC


# -------------------------
# GUI
# -------------------------

def main():
    # ==========
    # Sampling
    # ==========
    dt = 0.02
    T_total = 60.0
    t = np.arange(0.0, T_total, dt)

    # ==========
    # Initial parameters
    # ==========
    p0 = dict(
        theta0=0.35,
        f1=0.60,           # Hz
        f2=0.63,           # Hz
        gamma=0.005,       # 1/s
        phi1=0.10,         # rad
        phi2=1.20,         # rad
        noise=0.01,
        seed1=1,
        seed2=2,
        a1=0.90,
        a2=0.90,
        trace_seconds=25.0,
        max_lag_seconds=15.0,
        ttc_window_seconds=35.0,
    )

    # ==========
    # Layout
    # ==========
    fig = plt.figure(figsize=(16.2, 8.4))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 0.78],
        left=0.05, right=0.985, top=0.93, bottom=0.07,
        wspace=0.28, hspace=0.30,
    )

    ax_trace = fig.add_subplot(gs[0, 0])
    ax_g2 = fig.add_subplot(gs[0, 1])
    ax_ttc = fig.add_subplot(gs[0, 2])

    # Bottom row: equations box (left) + sliders (right, spanning 2 columns)
    gs_bottom = gs[1, :].subgridspec(nrows=1, ncols=3, width_ratios=[1.05, 1.0, 1.0], wspace=0.25)
    ax_eq = fig.add_subplot(gs_bottom[0, 0])
    ax_sliders_container = fig.add_subplot(gs_bottom[0, 1:])
    ax_sliders_container.axis("off")

    # Slider grid inside the right bottom region
    gs_sl = gs_bottom[0, 1:].subgridspec(nrows=5, ncols=6, wspace=0.35, hspace=0.80)

    def slider_ax(r, c, colspan=2, pad=0.02):
        """Create a slider axis and shrink it laterally to prevent label/value overlap."""
        ax = fig.add_subplot(gs_sl[r, c:c+colspan])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    # ==========
    # Equations box
    # ==========
    ax_eq.set_axis_off()
    eq_text = (
        r"$\theta_1(t)=\theta_0 e^{-\gamma t}\cos(\omega_1 t+\phi_1)$" "\n"
        r"$\theta_2(t)=\theta_0 e^{-\gamma t}\cos(\omega_2 t+\phi_2)$" "\n\n"
        r"$I_k(t)=1+a_k\,\frac{\theta_k(t)}{\max|\theta_k|}$" "\n\n"
        r"$g_{2,12}(\tau)=\frac{\langle I_1(t)\,I_2(t+\tau)\rangle}{\langle I_1\rangle\langle I_2\rangle}$" "\n\n"
        r"$C_{12}(t_1,t_2)=g_{2,12}(t_2-t_1)\ \ (\mathrm{diagonal\ stripes})$" "\n\n"
        r"$\omega_k=2\pi f_k,\quad T_{\mathrm{beat}}\approx 1/|f_2-f_1|$"
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

    # ==========
    # Artists (top row)
    # ==========
    line_I1, = ax_trace.plot([], [], linewidth=2, label=r"$I_1(t)$")
    line_I2, = ax_trace.plot([], [], linewidth=2, label=r"$I_2(t)$")
    ax_trace.set_title("Time traces")
    ax_trace.set_xlabel("time (s)")
    ax_trace.set_ylabel("intensity (a.u.)")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.legend(frameon=False)

    line_g2, = ax_g2.plot([], [], linewidth=2)
    ax_g2.axvline(0.0, linewidth=1, alpha=0.3)
    ax_g2.set_title(r"$g_{2,12}(\tau)$")
    ax_g2.set_xlabel(r"lag $\tau$ (s)")
    ax_g2.set_ylabel(r"$g_{2,12}(\tau)$")
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
    ax_ttc.set_title("Two-time map")
    ax_ttc.set_xlabel(r"$t_2$ (s)")
    ax_ttc.set_ylabel(r"$t_1$ (s)")
    cbar = fig.colorbar(im, ax=ax_ttc, fraction=0.046, pad=0.04)
    cbar.set_label("normalized corr.")

    # ==========
    # Sliders (compact labels)
    # ==========
    s_f1 = Slider(slider_ax(0, 0), "f₁ (Hz)", 0.10, 2.00, valinit=p0["f1"], valfmt="%.3f")
    s_f2 = Slider(slider_ax(0, 2), "f₂ (Hz)", 0.10, 2.00, valinit=p0["f2"], valfmt="%.3f")
    s_gamma = Slider(slider_ax(0, 4), "γ", 0.0, 0.20, valinit=p0["gamma"], valfmt="%.4f")

    s_phi1 = Slider(slider_ax(1, 0), "φ₁", 0.0, 2*np.pi, valinit=p0["phi1"], valfmt="%.2f")
    s_phi2 = Slider(slider_ax(1, 2), "φ₂", 0.0, 2*np.pi, valinit=p0["phi2"], valfmt="%.2f")
    s_theta0 = Slider(slider_ax(1, 4), "θ₀", 0.01, 1.00, valinit=p0["theta0"], valfmt="%.3f")

    s_a1 = Slider(slider_ax(2, 0), "a₁", 0.0, 1.5, valinit=p0["a1"], valfmt="%.2f")
    s_a2 = Slider(slider_ax(2, 2), "a₂", 0.0, 1.5, valinit=p0["a2"], valfmt="%.2f")
    s_noise = Slider(slider_ax(2, 4), "σ", 0.0, 0.2, valinit=p0["noise"], valfmt="%.3f")

    s_seed1 = Slider(slider_ax(3, 0), "seed₁", 0, 999, valinit=p0["seed1"], valfmt="%.0f")
    s_seed2 = Slider(slider_ax(3, 2), "seed₂", 0, 999, valinit=p0["seed2"], valfmt="%.0f")

    s_trace = Slider(slider_ax(4, 0), "trace (s)", 5.0, 60.0, valinit=p0["trace_seconds"], valfmt="%.1f")
    s_lag = Slider(slider_ax(4, 2), "lag (s)", 1.0, 40.0, valinit=p0["max_lag_seconds"], valfmt="%.1f")
    s_ttc = Slider(slider_ax(4, 4), "TTC (s)", 5.0, 60.0, valinit=p0["ttc_window_seconds"], valfmt="%.1f")

    # Smaller slider text helps a lot on laptops
    for s in (s_f1, s_f2, s_gamma, s_phi1, s_phi2, s_theta0, s_a1, s_a2, s_noise, s_seed1, s_seed2, s_trace, s_lag, s_ttc):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # Buttons
    ax_reset = fig.add_axes([0.055, 0.012, 0.09, 0.045])
    ax_save = fig.add_axes([0.150, 0.012, 0.09, 0.045])
    b_reset = Button(ax_reset, "Reset")
    b_save = Button(ax_save, "Save PNG")

    # ==========
    # Compute + update
    # ==========
    busy = {"flag": False}

    def compute(p):
        omega1 = 2 * np.pi * p["f1"]
        omega2 = 2 * np.pi * p["f2"]

        th1 = make_pendulum_signal(t, p["theta0"], omega1, p["gamma"], p["phi1"], p["noise"], p["seed1"])
        th2 = make_pendulum_signal(t, p["theta0"], omega2, p["gamma"], p["phi2"], p["noise"], p["seed2"])
        I1 = make_intensity_from_theta(th1, p["a1"])
        I2 = make_intensity_from_theta(th2, p["a2"])

        Nt = int(p["trace_seconds"] / dt)
        Nt = max(8, min(Nt, len(t)))
        tt = t[:Nt]
        I1t = I1[:Nt]
        I2t = I2[:Nt]

        max_lag = int(p["max_lag_seconds"] / dt)
        max_lag = max(1, min(max_lag, len(I1) - 2))
        lags, g12 = cross_g2(I1, I2, max_lag)
        tau = lags * dt

        Nw = int(p["ttc_window_seconds"] / dt)
        Nw = max(8, min(Nw, len(t)))
        tw = t[:Nw]
        TTC = two_time_cross_from_lagcorr(I1[:Nw], I2[:Nw])

        return tt, I1t, I2t, tau, g12, tw, TTC

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            p = dict(
                theta0=s_theta0.val,
                f1=s_f1.val,
                f2=s_f2.val,
                gamma=s_gamma.val,
                phi1=s_phi1.val,
                phi2=s_phi2.val,
                noise=s_noise.val,
                seed1=int(round(s_seed1.val)),
                seed2=int(round(s_seed2.val)),
                a1=s_a1.val,
                a2=s_a2.val,
                trace_seconds=s_trace.val,
                max_lag_seconds=s_lag.val,
                ttc_window_seconds=s_ttc.val,
            )

            tt, I1t, I2t, tau, g12, tw, TTC = compute(p)

            line_I1.set_data(tt, I1t)
            line_I2.set_data(tt, I2t)
            ax_trace.set_xlim(tt[0], tt[-1])
            ymin = min(I1t.min(), I2t.min())
            ymax = max(I1t.max(), I2t.max())
            pad = 0.05 * (ymax - ymin + 1e-12)
            ax_trace.set_ylim(ymin - pad, ymax + pad)

            line_g2.set_data(tau, g12)
            ax_g2.set_xlim(tau[0], tau[-1])
            y2min, y2max = g12.min(), g12.max()
            pad2 = 0.08 * (y2max - y2min + 1e-12)
            ax_g2.set_ylim(y2min - pad2, y2max + pad2)

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

    for s in (s_f1, s_f2, s_gamma, s_phi1, s_phi2, s_theta0, s_a1, s_a2, s_noise, s_seed1, s_seed2, s_trace, s_lag, s_ttc):
        s.on_changed(update)

    def on_reset(_event):
        for s in (s_f1, s_f2, s_gamma, s_phi1, s_phi2, s_theta0, s_a1, s_a2, s_noise, s_seed1, s_seed2, s_trace, s_lag, s_ttc):
            s.reset()
        update()

    def on_save(_event):
        fname = "pendulum_gui.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    main()