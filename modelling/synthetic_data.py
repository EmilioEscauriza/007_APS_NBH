# import numpy as np
# import matplotlib as mpl
#
# mpl.use("QtAgg")  # requires PyQt6; change to "macosx" if you prefer
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
#
#
# def make_ttc_product_gui_twofreq_half(
#     *,
#     n: int = 900,
#     dt_s: float = 1.0,
#     clip_percentile: float | None = 99.5,
#     cmap: str = "plasma",
# ):
#     """
#     Interactive TTC toy model with sliders.
#
#     Governing equations
#     -------------------
#       U = (t1 + t2)/2
#       V = (t2 - t1)
#
#       C(t1,t2) = C0
#                + A * cos(omega * V)
#                + m * cos(omega * U + phi) * cos(omega_m * V)
#
#       omega_m = omega / 2
#     """
#
#     # -------------------------
#     # grid
#     # -------------------------
#     t = np.arange(n, dtype=np.float64) * float(dt_s)
#     T1, T2 = np.meshgrid(t, t, indexing="xy")
#     V = (T2 - T1)
#     U = 0.5 * (T1 + T2)
#
#     def model(C0: float, A: float, omega: float, m: float, phi: float) -> np.ndarray:
#         omega = float(omega)
#         omega_m = 0.5 * omega
#
#         stripes = float(A) * np.cos(omega * V)
#         prod = np.cos(omega * U + float(phi)) * np.cos(omega_m * V)
#         return float(C0) + stripes + float(m) * prod
#
#     def clip_for_display(C: np.ndarray) -> np.ndarray:
#         if clip_percentile is None:
#             return C
#         hi = np.percentile(C, float(clip_percentile))
#         lo = np.percentile(C, 0.0)
#         return np.clip(C, lo, hi)
#
#     # -------------------------
#     # defaults
#     # -------------------------
#     # Period ~ 250 s => omega ~ 2pi/250
#     p0 = dict(
#         C0=1.0,
#         A=0.6,
#         omega=2.0 * np.pi / 250.0,
#         m=0.6,
#         phi=0.0,
#     )
#
#     # -------------------------
#     # layout
#     # -------------------------
#     fig = plt.figure(figsize=(12.8, 7.2))
#     gs = fig.add_gridspec(
#         2, 1,
#         height_ratios=[1.0, 0.28],
#         left=0.06, right=0.985, top=0.93, bottom=0.09,
#         hspace=0.24,
#     )
#
#     ax_img = fig.add_subplot(gs[0, 0])
#     ax_sl = fig.add_subplot(gs[1, 0])
#     ax_sl.axis("off")
#
#     C = model(**p0)
#     Cplot = clip_for_display(C)
#
#     im = ax_img.imshow(
#         Cplot,
#         origin="lower",
#         cmap=cmap,
#         interpolation="nearest",
#         aspect="equal",
#         extent=[t[0], t[-1], t[0], t[-1]],
#     )
#     ax_img.set_xlabel(r"$t_1$ (s)")
#     ax_img.set_ylabel(r"$t_2$ (s)")
#     cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.03)
#     cbar.set_label("C(t₁,t₂) (a.u.)")
#
#     # slider grid
#     gs_sl = gs[1, 0].subgridspec(2, 4, wspace=0.40, hspace=1.05)
#
#     def slider_ax(r, c, pad=0.018):
#         ax = fig.add_subplot(gs_sl[r, c])
#         pos = ax.get_position()
#         ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
#         return ax
#
#     # Row 1: offsets/amplitudes
#     s_C0 = Slider(slider_ax(0, 0), "C0", 0.0, 3.0, valinit=p0["C0"], valfmt="%.3f")
#     s_A  = Slider(slider_ax(0, 1), "A", -2.0, 2.0, valinit=p0["A"], valfmt="%.3f")
#     s_m  = Slider(slider_ax(0, 2), "m", -2.0, 2.0, valinit=p0["m"], valfmt="%.3f")
#     s_phi = Slider(slider_ax(0, 3), "φ", -np.pi, np.pi, valinit=p0["phi"], valfmt="%.3f")
#
#     # Row 2: omega slider (shared) + readout slot
#     # Period range 50..600 s -> omega range [2pi/600, 2pi/50]
#     wmin = 2.0 * np.pi / 600.0
#     wmax = 2.0 * np.pi / 50.0
#     s_omega = Slider(slider_ax(1, 0), "ω (rad/s)", wmin, wmax, valinit=p0["omega"], valfmt="%.5f")
#
#     ax_info = fig.add_subplot(gs_sl[1, 1:])
#     ax_info.axis("off")
#
#     # Make slider text a bit smaller so it fits nicely
#     for s in (s_C0, s_A, s_m, s_phi, s_omega):
#         s.label.set_fontsize(9)
#         s.valtext.set_fontsize(9)
#
#     # buttons
#     ax_reset = fig.add_axes([0.06, 0.02, 0.08, 0.045])
#     b_reset = Button(ax_reset, "Reset")
#
#     ax_save = fig.add_axes([0.15, 0.02, 0.10, 0.045])
#     b_save = Button(ax_save, "Save PNG")
#
#     busy = {"flag": False}
#     info_text = ax_info.text(0.0, 0.65, "", fontsize=11, va="center", ha="left")
#
#     def update(_=None):
#         if busy["flag"]:
#             return
#         busy["flag"] = True
#         try:
#             omega = float(s_omega.val)
#             omega_m = 0.5 * omega
#             Tu = (2.0 * np.pi / omega) if omega > 0 else np.inf
#             Tv = (2.0 * np.pi / omega) if omega > 0 else np.inf
#             Tm = (2.0 * np.pi / omega_m) if omega_m > 0 else np.inf
#
#             C = model(
#                 C0=float(s_C0.val),
#                 A=float(s_A.val),
#                 omega=omega,
#                 m=float(s_m.val),
#                 phi=float(s_phi.val),
#             )
#             Cplot = clip_for_display(C)
#             im.set_data(Cplot)
#
#             vmin = float(np.min(Cplot))
#             vmax = float(np.max(Cplot))
#             if vmax <= vmin:
#                 vmax = vmin + 1e-12
#             im.set_clim(vmin, vmax)
#
#             ax_img.set_title(rf"TTC model | $\omega$={omega:.5f} rad/s, $\omega_m$={omega_m:.5f} rad/s")
#             info_text.set_text(
#                 f"Derived periods:\n"
#                 f"  T(ω)   = {Tu:.1f} s  (applies to A term and cos(ωU+φ))\n"
#                 f"  T(ω_m) = {Tm:.1f} s  (ω_m = ω/2, applies to cos(ω_m V))"
#             )
#
#             fig.canvas.draw_idle()
#         finally:
#             busy["flag"] = False
#
#     for s in (s_C0, s_A, s_m, s_phi, s_omega):
#         s.on_changed(update)
#
#     def on_reset(_event):
#         for s in (s_C0, s_A, s_m, s_phi, s_omega):
#             s.reset()
#         update()
#
#     def on_save(_event):
#         fname = "ttc_product_gui_twofreq_omega_m_half.png"
#         fig.savefig(fname, dpi=200)
#         print(f"Saved: {fname}")
#
#     b_reset.on_clicked(on_reset)
#     b_save.on_clicked(on_save)
#
#     update()
#     plt.show()
#
#
# if __name__ == "__main__":
#     make_ttc_product_gui_twofreq_half(n=900, dt_s=1.0, clip_percentile=99.5)

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")  # requires PyQt6; change to "macosx" if you prefer
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def make_I_and_ttc_product_gui_twofreq_half(
    *,
    n: int = 900,
    dt_s: float = 1.0,
    clip_percentile: float | None = 99.5,
    cmap: str = "plasma",
):
    """
    Interactive TTC toy model with sliders + a linked I(t) panel.

    Governing equations
    -------------------
      u = (t1 + t2)/2
      v = (t2 - t1)

      C(t1,t2) = C0
               + A * cos(omega * v)
               + m * cos(omega * u + phi) * cos((omega/2) * v)

    Linked intensity-like trace (for intuition only)
    ------------------------------------------------
      I(t) = C0 + aI*cos(omega*t) + mI*cos((omega/2)*t + phi)

    where aI, mI are smooth functions of A, m (keeps sign, compresses range):
      aI = sign(A) * sqrt(|A|)
      mI = sign(m) * sqrt(|m|)
    """

    # -------------------------
    # grid
    # -------------------------
    t = np.arange(n, dtype=np.float64) * float(dt_s)
    T1, T2 = np.meshgrid(t, t, indexing="xy")
    v = (T2 - T1)
    u = 0.5 * (T1 + T2)

    def model(C0: float, A: float, omega: float, m: float, phi: float) -> np.ndarray:
        omega = float(omega)
        omega_m = 0.5 * omega
        stripes = float(A) * np.cos(omega * v)
        prod = np.cos(omega * u + float(phi)) * np.cos(omega_m * v)
        return float(C0) + stripes + float(m) * prod

    def model_I(C0: float, A: float, omega: float, m: float, phi: float) -> np.ndarray:
        omega = float(omega)
        omega_m = 0.5 * omega
        aI = np.sign(A) * np.sqrt(abs(A))
        mI = np.sign(m) * np.sqrt(abs(m))
        return float(C0) + aI * np.cos(omega * t) + mI * np.cos(omega_m * t + float(phi))

    def clip_for_display(C: np.ndarray) -> np.ndarray:
        if clip_percentile is None:
            return C
        hi = np.percentile(C, float(clip_percentile))
        lo = np.percentile(C, 0.0)
        return np.clip(C, lo, hi)

    # -------------------------
    # defaults
    # -------------------------
    # Period ~ 250 s => omega ~ 2pi/250
    p0 = dict(
        C0=1.0,
        A=0.6,
        omega=2.0 * np.pi / 250.0,
        m=0.6,
        phi=0.0,
    )

    # -------------------------
    # layout (now 2 columns on top)
    # -------------------------
    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 0.28],
        width_ratios=[1.0, 1.05],
        left=0.06, right=0.985, top=0.93, bottom=0.09,
        wspace=0.30, hspace=0.24,
    )

    ax_I = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])
    ax_sl = fig.add_subplot(gs[1, :])
    ax_sl.axis("off")

    # initial
    I = model_I(**p0)
    C = model(**p0)
    Cplot = clip_for_display(C)

    # ---- Left: I(t)
    (line_I,) = ax_I.plot(t, I, lw=2.0)
    ax_I.set_xlabel("t (s)")
    ax_I.set_ylabel("I(t) (a.u.)")
    ax_I.grid(True, alpha=0.25)

    # ---- Right: TTC
    im = ax_img.imshow(
        Cplot,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        extent=[t[0], t[-1], t[0], t[-1]],
    )
    ax_img.set_xlabel(r"$t_1$ (s)")
    ax_img.set_ylabel(r"$t_2$ (s)")
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.03)
    cbar.set_label("C(t₁,t₂) (a.u.)")

    # -------------------------
    # slider grid (same style as yours)
    # -------------------------
    gs_sl = gs[1, :].subgridspec(2, 4, wspace=0.40, hspace=1.05)

    def slider_ax(r, c, pad=0.018):
        ax = fig.add_subplot(gs_sl[r, c])
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad, pos.y0, pos.width - 2 * pad, pos.height])
        return ax

    # Row 1: offsets/amplitudes
    s_C0 = Slider(slider_ax(0, 0), "C0", 0.0, 3.0, valinit=p0["C0"], valfmt="%.3f")
    s_A = Slider(slider_ax(0, 1), "A", -2.0, 2.0, valinit=p0["A"], valfmt="%.3f")
    s_m = Slider(slider_ax(0, 2), "m", -2.0, 2.0, valinit=p0["m"], valfmt="%.3f")
    s_phi = Slider(slider_ax(0, 3), "φ", -np.pi, np.pi, valinit=p0["phi"], valfmt="%.3f")

    # Row 2: omega slider + info
    wmin = 2.0 * np.pi / 600.0
    wmax = 2.0 * np.pi / 50.0
    s_omega = Slider(slider_ax(1, 0), "ω (rad/s)", wmin, wmax, valinit=p0["omega"], valfmt="%.5f")

    ax_info = fig.add_subplot(gs_sl[1, 1:])
    ax_info.axis("off")

    for s in (s_C0, s_A, s_m, s_phi, s_omega):
        s.label.set_fontsize(9)
        s.valtext.set_fontsize(9)

    # buttons
    ax_reset = fig.add_axes([0.06, 0.02, 0.08, 0.045])
    b_reset = Button(ax_reset, "Reset")

    ax_save = fig.add_axes([0.15, 0.02, 0.10, 0.045])
    b_save = Button(ax_save, "Save PNG")

    busy = {"flag": False}
    info_text = ax_info.text(0.0, 0.65, "", fontsize=11, va="center", ha="left")

    def update(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        try:
            omega = float(s_omega.val)
            omega_m = 0.5 * omega
            Tu = (2.0 * np.pi / omega) if omega > 0 else np.inf
            Tm = (2.0 * np.pi / omega_m) if omega_m > 0 else np.inf

            # TTC (exact same as your model)
            C = model(
                C0=float(s_C0.val),
                A=float(s_A.val),
                omega=omega,
                m=float(s_m.val),
                phi=float(s_phi.val),
            )
            Cplot = clip_for_display(C)
            im.set_data(Cplot)

            vmin = float(np.min(Cplot))
            vmax = float(np.max(Cplot))
            if vmax <= vmin:
                vmax = vmin + 1e-12
            im.set_clim(vmin, vmax)

            # I(t) (linked)
            I = model_I(
                C0=float(s_C0.val),
                A=float(s_A.val),
                omega=omega,
                m=float(s_m.val),
                phi=float(s_phi.val),
            )
            line_I.set_ydata(I)
            ax_I.relim()
            ax_I.autoscale_view()

            ax_img.set_title(rf"TTC model | $\omega$={omega:.5f} rad/s, $\omega/2$={omega_m:.5f} rad/s")
            info_text.set_text(
                f"Derived periods:\n"
                f"  T(ω)     = {Tu:.1f} s\n"
                f"  T(ω/2)   = {Tm:.1f} s"
            )

            fig.canvas.draw_idle()
        finally:
            busy["flag"] = False

    for s in (s_C0, s_A, s_m, s_phi, s_omega):
        s.on_changed(update)

    def on_reset(_event):
        for s in (s_C0, s_A, s_m, s_phi, s_omega):
            s.reset()
        update()

    def on_save(_event):
        fname = "I_and_ttc_product_gui_twofreq_half.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"Saved: {fname}")

    b_reset.on_clicked(on_reset)
    b_save.on_clicked(on_save)

    update()
    plt.show()


if __name__ == "__main__":
    make_I_and_ttc_product_gui_twofreq_half(n=800, dt_s=1.0, clip_percentile=99.5)