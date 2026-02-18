from __future__ import annotations

import numpy as np
import matplotlib as mpl
mpl.use("macosx")  # set before pyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def symmetrize(C: np.ndarray) -> np.ndarray:
    # Mirror along t1=t2: C + C^T - diag(diag(C))
    C = np.asarray(C, dtype=np.float64)
    return C + C.T - np.diag(np.diag(C))


def clip_percentile(C: np.ndarray, p_hi: float) -> np.ndarray:
    lo, hi = np.percentile(C, [0.0, float(p_hi)])
    return np.clip(C, lo, hi)


def make_ttc_model_gui_two_freqs(
    *,
    n: int = 450,
    tmax: float = 4800.0,      # seconds (just sets axis scaling)
    C0: float = 0.0,
    Ad: float = 1.0,
    Ai: float = 1.0,
    wd: float = 2 * np.pi / 200.0,  # rad/s (period 200 s)
    wi: float = 2 * np.pi / 120.0,  # rad/s (period 120 s)
    clip_hi_percentile: float = 99.5,
    cmap: str = "plasma",
):
    """
    Model (2 frequencies total):
      C(t1,t2)= C0 + Ad*cos(wd*(t2-t1)) + Ai*cos(wi*t1)*cos(wi*t2)

    Always symmetrized for display.
    """
    # time axes
    t = np.linspace(0.0, float(tmax), int(n))
    t1, t2 = np.meshgrid(t, t, indexing="xy")  # t1 along x, t2 along y

    def build_C(C0_, Ad_, Ai_, wd_, wi_):
        C = (
            C0_
            + Ad_ * np.cos(wd_ * (t2 - t1))
            + Ai_ * (np.cos(wi_ * t1) * np.cos(wi_ * t2))
        )
        return symmetrize(C)

    # initial
    C = build_C(C0, Ad, Ai, wd, wi)
    Cplot = clip_percentile(C, clip_hi_percentile)

    # ---- figure layout ----
    fig = plt.figure(figsize=(11.5, 6.5))
    ax_img = fig.add_axes([0.07, 0.20, 0.58, 0.74])  # left big image
    ax_txt = fig.add_axes([0.68, 0.70, 0.30, 0.24])  # text panel
    ax_txt.axis("off")

    im = ax_img.imshow(
        Cplot,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=[t.min(), t.max(), t.min(), t.max()],
        aspect="auto",
    )
    ax_img.set_aspect("equal")
    ax_img.set_xlabel("t₁ (s)")
    ax_img.set_ylabel("t₂ (s)")
    ax_img.set_title("Toy TTC model (always symmetrized)")

    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.02)

    # ---- sliders ----
    # Slider axes [left, bottom, width, height]
    ax_C0 = fig.add_axes([0.73, 0.60, 0.22, 0.03])
    ax_Ad = fig.add_axes([0.73, 0.55, 0.22, 0.03])
    ax_Ai = fig.add_axes([0.73, 0.50, 0.22, 0.03])

    ax_Pd = fig.add_axes([0.73, 0.42, 0.22, 0.03])  # period for wd
    ax_Pi = fig.add_axes([0.73, 0.37, 0.22, 0.03])  # period for wi

    ax_clip = fig.add_axes([0.73, 0.29, 0.22, 0.03])

    s_C0 = Slider(ax_C0, "C0", valmin=-2.0, valmax=2.0, valinit=C0)
    s_Ad = Slider(ax_Ad, "Ad", valmin=0.0, valmax=3.0, valinit=Ad)
    s_Ai = Slider(ax_Ai, "Ai", valmin=0.0, valmax=3.0, valinit=Ai)

    # Period sliders are more intuitive than omega
    # Guard min period to avoid crazy high freq aliasing on coarse grids
    Pd0 = 2 * np.pi / wd if wd > 0 else 200.0
    Pi0 = 2 * np.pi / wi if wi > 0 else 120.0

    s_Pd = Slider(ax_Pd, "Pd (s)", valmin=10.0, valmax=2000.0, valinit=Pd0)
    s_Pi = Slider(ax_Pi, "Pi (s)", valmin=10.0, valmax=2000.0, valinit=Pi0)

    s_clip = Slider(ax_clip, "clip %", valmin=90.0, valmax=100.0, valinit=clip_hi_percentile)

    # ---- reset button ----
    ax_reset = fig.add_axes([0.80, 0.22, 0.12, 0.05])
    b_reset = Button(ax_reset, "Reset")

    def update(_=None):
        C0_ = float(s_C0.val)
        Ad_ = float(s_Ad.val)
        Ai_ = float(s_Ai.val)

        Pd_ = float(s_Pd.val)
        Pi_ = float(s_Pi.val)
        wd_ = 2 * np.pi / Pd_
        wi_ = 2 * np.pi / Pi_

        clip_ = float(s_clip.val)

        Cnew = build_C(C0_, Ad_, Ai_, wd_, wi_)
        Cnew = clip_percentile(Cnew, clip_)

        im.set_data(Cnew)

        # update text panel
        ax_txt.clear()
        ax_txt.axis("off")
        ax_txt.text(
            0.0, 1.0,
            "Model:\n"
            "C = C0 + Ad cos(wd(t2-t1)) + Ai cos(wi t1)cos(wi t2)\n\n"
            f"Pd = {Pd_:.2f} s   (wd = {wd_:.4g} rad/s)\n"
            f"Pi = {Pi_:.2f} s   (wi = {wi_:.4g} rad/s)\n\n"
            f"Ad/Ai = {(Ad_/Ai_ if Ai_ > 0 else np.inf):.3g}\n"
            f"clip = {clip_:.2f}%",
            va="top",
            fontsize=10,
        )

        fig.canvas.draw_idle()

    def on_reset(_event):
        s_C0.reset()
        s_Ad.reset()
        s_Ai.reset()
        s_Pd.reset()
        s_Pi.reset()
        s_clip.reset()
        update()

    for s in (s_C0, s_Ad, s_Ai, s_Pd, s_Pi, s_clip):
        s.on_changed(update)
    b_reset.on_clicked(on_reset)

    # initial text fill
    update()
    plt.show()


if __name__ == "__main__":
    make_ttc_model_gui_two_freqs(
        n=450,
        tmax=4800.0,
        C0=0.0,
        Ad=1.0,
        Ai=1.0,
        wd=2 * np.pi / 200.0,
        wi=2 * np.pi / 120.0,
        clip_hi_percentile=99.5,
    )