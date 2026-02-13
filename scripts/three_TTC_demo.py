"""
Interactive Langevin oscillator demo showing THREE "two-time" constructions side-by-side
for the SAME simulation data (same x(t)).

Top row (all square):
  1) Outer-product (single trajectory):
       M_outer(t1,t2) = x1(t1) x1(t2) / sqrt(<x1^2(t1)><x1^2(t2)>)  (rank-1 look, chequer-prone)

  2) Ensemble two-time correlation (normalized):
       C(t1,t2)    = < x(t1) x(t2) >_traj
       Cnorm(t1,t2)= C / sqrt(C(t1,t1) C(t2,t2))                    (stripe-prone)

  3) Lag-based "stripe TTC" from the ensemble lag correlation:
       G(τ) = < x(t) x(t+τ) >_{t,traj}
       M_lag(t1,t2) = G(|t2-t1|) / G(0)                             (forced stripes)

Bottom row:
  Left  : equations/definitions box (in-layout, not overlaid)
  Right : sliders + buttons

Dynamics (semi-implicit Euler–Maruyama):
    dv = (-(gamma/m) v - (k/m) x) dt + sigma dW
    x_{n+1} = x_n + v_{n+1} dt
  with sigma = sqrt(2*gamma*kB*T)/m

Requires:
  pip install PyQt6

Run:
  python langevin_three_ttc_gui.py
"""

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# -------------------------
# Fixed ensemble size
# -------------------------
NTRAJ = 1024  # fixed, matches the "scale" of the earlier GUIs


# -------------------------
# Simulation
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

    # equilibrium initial conditions
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
        x[:, n + 1] = x[:, n] + v[:, n + 1] * dt  # semi-implicit

    return t, x


# -------------------------
# Maps
# -------------------------

def map_outer_singletraj_norm(x1: np.ndarray) -> np.ndarray:
    """
    "Normal TTC equation form" many people start with for a single trajectory,
    but normalized like a correlation coefficient so color scale is meaningful:

      M[i,j] = x1[i] x1[j] / sqrt( (x1[i]^2)(x1[j]^2) )  = sign(x1[i] x1[j])

    That extreme normalization is too aggressive, so instead we normalize using
    the RMS of the full series (typical practical choice):

      M = outer(x1, x1) / <x1^2>
    """
    denom = np.mean(x1**2) + 1e-12
    return np.outer(x1, x1) / denom


def map_ensemble_cnorm(x: np.ndarray) -> np.ndarray:
    """
    C(t1,t2)=<x(t1)x(t2)>_traj
    Cnorm = C / sqrt(C(t1,t1)C(t2,t2))
    """
    C = (x.T @ x) / float(x.shape[0])
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / (d[:, None] * d[None, :])


def lagcorr_ensemble(x: np.ndarray) -> np.ndarray:
    """
    G(k) = < x(t) x(t+k) >_{t,traj}
    Returns G for k=0..Nt-1
    """
    Ntraj, Nt = x.shape
    G = np.empty(Nt, dtype=float)
    for k in range(Nt):
        a = x[:, : Nt - k]
        b = x[:, k: Nt]
        G[k] = np.mean(a * b)
    return G


def map_stripe_from_lag(G: np.ndarray) -> np.ndarray:
    """
    M_lag[i,j] = G(|i-j|) / G(0)
    """
    G0 = G[0] if abs(G[0]) > 1e-20 else 1.0
    idx = np.abs(np.subtract.outer(np.arange(len(G)), np.arange(len(G))))
    return G[idx] / G0


# -------------------------
# GUI
# -------------------------

def main():
    p0 = dict(
        m=10.0,
        k=1.0,
        gamma=0.1,
        kB=1.0,      # fixed
        T=1.0,
        tmax=30.0,
        dt=0.02,
        seed=1,
    )

    # Layout: 2 rows x 3 cols
    # Top: 3 maps
    # Bottom: text (spans 1 col) + sliders (span 2 cols)
    fig = plt.figure(figsize=(18.0, 9.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 0.82],
        left=0.04, right=0.99, top=0.93, bottom=0.07,
        wspace=0.22, hspace=0.30,
    )

    ax_outer = fig.add_subplot(gs[0, 0])
    ax_cnorm = fig.add_subplot(gs[0, 1])
    ax_lag   = fig.add_subplot(gs[0, 2])

    ax_text  = fig.add_subplot(gs[1, 0])
    ax_text.set_axis_off()

    ax_slider_area = fig.add_subplot(gs[1, 1:])
    ax_slider_area.set_axis_off()

    # Text box (in-place)
    eq_text = (
        r"$dv = (-(\gamma/m)v - (k/m)x)\,dt + \sigma\,dW$" "\n"
        r"$x_{n+1}=x_n+v_{n+1}dt$ (semi-implicit)" "\n\n"
        r"$\sigma=\sqrt{2\gamma k_B T}/m$" "\n\n"
        r"Outer (single traj): $M_{\rm outer}=x_1(t_1)x_1(t_2)/\langle x_1^2\rangle$" "\n"
        r"Ensemble: $C_{\rm norm}=\frac{\langle x(t_1)x(t_2)\rangle}{\sqrt{C(t_1,t_1)C(t_2,t_2)}}$" "\n"
        r"Lag-stripe: $M_{\rm lag}(t_1,t_2)=G(|t_2-t_1|)/G(0)$" "\n\n"
        rf"Fixed: $N_{{traj}}={NTRAJ}$, $k_B={p0['kB']}$"
    )
    ax_text.text(
        0.02, 0.98, eq_text,
        va="top", ha="left",
        fontsize=11,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", alpha=0.95),
        transform=ax_text.transAxes,
    )
    ax_text.set_title("Why stripes vs chequer", loc="left", pad=6)

    # Heatmap artists
    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    im_outer = ax_outer.imshow(np.zeros((10, 10)), origin="lower", cmap=cmap, interpolation="nearest",
                               extent=[0, 1, 0, 1], aspect="equal")
    im_cnorm = ax_cnorm.imshow(np.zeros((10, 10)), origin="lower", cmap=cmap, interpolation="nearest",
                               extent=[0, 1, 0, 1], aspect="equal", vmin=-1, vmax=1)
    im_lag   = ax_lag.imshow(np.zeros((10, 10)), origin="lower", cmap=cmap, interpolation="nearest",
                             extent=[0, 1, 0, 1], aspect="equal", vmin=-1, vmax=1)

    for ax in (ax_outer, ax_cnorm, ax_lag):
        ax.set_xlabel("t2 (s)")
        ax.set_ylabel("t1 (s)")
        ax.set_aspect("equal", adjustable="box")

    ax_outer.set_title("Single-trajectory outer product (chequer-prone)")
    ax_cnorm.set_title("Ensemble two-time correlation (stripe-prone)")
    ax_lag.set_title("Lag-only stripe map (forced stripes)")

    c0 = fig.colorbar(im_outer, ax=ax_outer, fraction=0.046, pad=0.03)
    c0.set_label("value")
    c1 = fig.colorbar(im_cnorm, ax=ax_cnorm, fraction=0.046, pad=0.03)
    c1.set_label("Cnorm")
    c2 = fig.colorbar(im_lag, ax=ax_lag, fraction=0.046, pad=0.03)
    c2.set_label("G(|Δt|)/G(0)")

    # Sliders inside bottom-right cell
    gs_sl = gs[1, 1:].subgridspec(nrows=3, ncols=6, wspace=0.35, hspace=0.95)

    def slider_ax(r, c, colspan=2, pad=0.02):
        ax = fig.add_subplot(gs_sl[r, c:c + colspan])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    s_m     = Slider(slider_ax(0, 0), "m",     0.5, 50.0, valinit=p0["m"],     valfmt="%.2f")
    s_k     = Slider(slider_ax(0, 2), "k",     0.1, 20.0, valinit=p0["k"],     valfmt="%.2f")
    s_gamma = Slider(slider_ax(0, 4), "γ",     0.0, 5.0,  valinit=p0["gamma"], valfmt="%.3f")

    s_T     = Slider(slider_ax(1, 0), "T",     0.05, 5.0, valinit=p0["T"],     valfmt="%.2f")
    s_tmax  = Slider(slider_ax(1, 2), "tmax",  5.0, 200.0,valinit=p0["tmax"],  valfmt="%.1f")
    s_dt    = Slider(slider_ax(1, 4), "dt",    0.005,0.10,valinit=p0["dt"],    valfmt="%.3f")

    s_seed  = Slider(slider_ax(2, 0), "seed",  0, 999,    valinit=p0["seed"],  valfmt="%.0f")

    sliders = (s_m, s_k, s_gamma, s_T, s_tmax, s_dt, s_seed)
    for s in sliders:
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # Buttons
    ax_reset = fig.add_axes([0.045, 0.012, 0.09, 0.045])
    ax_save  = fig.add_axes([0.140, 0.012, 0.12, 0.045])
    b_reset = Button(ax_reset, "Reset")
    b_save  = Button(ax_save,  "Save PNG")

    busy = {"flag": False}

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            m = float(s_m.val)
            k = float(s_k.val)
            gamma = float(s_gamma.val)
            T = float(s_T.val)
            dt = float(s_dt.val)
            tmax = float(s_tmax.val)
            seed = int(round(s_seed.val))

            # Guardrail: now we build 3 Nt×Nt maps, keep it smaller
            Nt = int(np.floor(tmax / dt)) + 1
            if Nt > 2000:
                tmax = 2000 * dt
                Nt = int(np.floor(tmax / dt)) + 1

            t, x = simulate_ensemble_langevin(
                m=m, k=k, gamma=gamma, kB=p0["kB"], T=T,
                tmax=tmax, dt=dt, seed=seed
            )

            # pick a single trajectory for the outer-product map
            x1 = x[0, :]

            M_outer = map_outer_singletraj_norm(x1)
            Cn = map_ensemble_cnorm(x)
            G = lagcorr_ensemble(x)
            M_lag = map_stripe_from_lag(G)

            # Update images + extents
            for im, M in ((im_outer, M_outer), (im_cnorm, Cn), (im_lag, M_lag)):
                im.set_data(M)
                im.set_extent([t[0], t[-1], t[0], t[-1]])

            # Color scaling choices:
            # - outer product has wider distribution, use robust limits
            v0 = np.percentile(M_outer, 1)
            v1 = np.percentile(M_outer, 99)
            if v1 <= v0:
                v1 = v0 + 1e-6
            im_outer.set_clim(v0, v1)

            # Cnorm and lag-map naturally in ~[-1,1] for stable systems
            im_cnorm.set_clim(-1, 1)
            im_lag.set_clim(-1, 1)

            for ax in (ax_outer, ax_cnorm, ax_lag):
                ax.set_xlim(t[0], t[-1])
                ax.set_ylim(t[0], t[-1])
                ax.set_aspect("equal", adjustable="box")

            omega0 = np.sqrt(k / m)
            ax_cnorm.set_title(f"Ensemble two-time correlation (ω0={omega0:.3f} rad/s)")
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
        fname = "langevin_three_ttc.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    main()