from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import h5py
import numpy as np

from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl
mpl.use("macosx")  # must be set before importing pyplot
import matplotlib.pyplot as plt

import re


# ============================================================
# File finding + loading
# ============================================================

def find_results_hdf(base_dir: Path, sample_id: str) -> Path:
    """
    Find the first file matching: <sample_id>_*_results.hdf
    Raises FileNotFoundError if none exist.
    """
    matches = sorted(base_dir.glob(f"{sample_id}_*_results.hdf"))
    if not matches:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")
    return matches[0]


@dataclass
class XPCSData:
    dynamic_roi_map: np.ndarray
    scattering_2d: np.ndarray
    ttc: np.ndarray
    g2: np.ndarray


def load_xpcs_arrays(
    sample_id: str,
    base_dir: Path,
    *,
    mask_n: int,
    scattering_first_frame_only: bool = True,
) -> XPCSData:
    """
    Load the 4 arrays you listed as NumPy arrays:

        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d   = f["xpcs/temporal_mean/scattering_2d"][...]
        ttc             = f["xpcs/twotime/correlation_map/c2_00{mask_n:03d}"][...]
        g2              = f["xpcs/twotime/normalized_g2"][...]

    Returns XPCSData(dataclass).
    """
    hdf_path = find_results_hdf(base_dir, sample_id)
    ttc_path = f"xpcs/twotime/correlation_map/c2_00{mask_n:03d}"

    with h5py.File(hdf_path, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]

        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        if scattering_first_frame_only and scattering_2d.ndim == 3:
            scattering_2d = scattering_2d[0, :, :]

        ttc = f[ttc_path][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]

    return XPCSData(
        dynamic_roi_map=dynamic_roi_map,
        scattering_2d=scattering_2d,
        ttc=ttc,
        g2=g2,
    )


def save_xpcs_npz(out_path: Path, data: XPCSData) -> Path:
    """Optional cache to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        dynamic_roi_map=data.dynamic_roi_map,
        scattering_2d=data.scattering_2d,
        ttc=data.ttc,
        g2=data.g2,
    )
    return out_path


# ============================================================
# TTC preprocessing + utilities
# ============================================================

def despike_patch_with_local_median(C: np.ndarray, center: Tuple[int, int], halfwidth: int = 1) -> np.ndarray:
    """
    Replace a small (2*halfwidth+1)^2 patch centered at `center` with the median
    of a slightly larger surrounding window (excluding the patch).
    """
    C = C.copy()
    n = C.shape[0]
    cy, cx = center

    y0 = max(cy - halfwidth, 0)
    y1 = min(cy + halfwidth + 1, n)
    x0 = max(cx - halfwidth, 0)
    x1 = min(cx + halfwidth + 1, n)

    pad = max(halfwidth + 2, 3)
    Y0 = max(cy - pad, 0)
    Y1 = min(cy + pad + 1, n)
    X0 = max(cx - pad, 0)
    X1 = min(cx + pad + 1, n)

    window = C[Y0:Y1, X0:X1].copy()

    py0, py1 = y0 - Y0, y1 - Y0
    px0, px1 = x0 - X0, x1 - X0
    window[py0:py1, px0:px1] = np.nan

    med = np.nanmedian(window)
    if np.isfinite(med):
        C[y0:y1, x0:x1] = med

    return C


def arrow_endpoint_to_edge(start_x: int, start_y: int, dx: int, dy: int, n: int) -> Tuple[int, int]:
    """
    For a direction (dx,dy) in {-1,0,1}, return the endpoint on the image edge
    when stepping from (start_x,start_y) until leaving bounds.
    """
    if dx == 0 and dy == 0:
        return start_x, start_y

    steps = []
    if dx > 0:
        steps.append(n - 1 - start_x)
    elif dx < 0:
        steps.append(start_x)

    if dy > 0:
        steps.append(n - 1 - start_y)
    elif dy < 0:
        steps.append(start_y)

    kmax = min(steps) if steps else 0
    return start_x + dx * kmax, start_y + dy * kmax


# ============================================================
# Lineouts (TTC + annotated figure)
# ============================================================

def plot_ttc_with_lineouts(
    data: XPCSData,
    start: int,
    *,
    clip_percentile: Optional[float] = 99.9,
    cmap: str = "plasma",
    add_antidiag_se: bool = True,
    despike_at_start: bool = True,
    despike_halfwidth: int = 1,
) -> None:
    """
    Left : symmetrized TTC with arrows for lineouts (to edges)
    Right: lineout curves

    - TTC is mirrored about x=y: C -> C + C.T - diag(diag(C))
    - Vertical lineout goes DOWN in time (toward x-axis) => decreasing t2 index
    - Adds anti-diagonal lineout from (start,start) toward SE: (x+, y-)
    """
    C = symmetrize_ttc(data.ttc)
    n = C.shape[0]
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"TTC must be square, got shape {C.shape}")
    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n-1}]")

    if despike_at_start:
        C = despike_patch_with_local_median(C, center=(start, start), halfwidth=despike_halfwidth)

    # Lineouts (anchored at (start,start))
    # Horizontal right: (x+, y)
    x_h = np.arange(start, n)
    y_h = C[start, start:n]

    # Vertical DOWN: (x, y-) -> decreasing row index
    x_v = np.arange(start, -1, -1)
    y_v = C[start::-1, start]

    # Main diagonal forward: (x+, y+)
    x_d = np.arange(start, n)
    y_d = np.diag(C)[start:n]

    lineouts = [
        dict(x=x_h, y=y_h, color="tab:blue",   label="horizontal (→)"),
        dict(x=x_v, y=y_v, color="tab:orange", label="vertical (↓)"),
        dict(x=x_d, y=y_d, color="tab:green",  label="diag x=y (↗)"),
    ]

    # Anti-diagonal SE (x+, y-) from (start,start): (start+k, start-k)
    if add_antidiag_se:
        kmax = min(n - 1 - start, start)
        xs = start + np.arange(0, kmax + 1)
        ys = start - np.arange(0, kmax + 1)
        y_ad = C[ys, xs]  # C[row=y, col=x]
        lineouts.append(dict(x=xs, y=y_ad, color="tab:purple", label="anti-diag (↘)"))

    Cplot = C.copy()
    if clip_percentile is not None:
        Cplot = clip_ttc(Cplot, p_hi=float(clip_percentile))

    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    # Left: TTC + arrows
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title("TTC with lineouts")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    arrows = [
        ("horizontal", (1, 0),  "tab:blue"),
        ("vertical",   (0, -1), "tab:orange"),  # down
        ("diag x=y",   (1, 1),  "tab:green"),
    ]
    if add_antidiag_se:
        arrows.append(("anti-diag", (1, -1), "tab:purple"))  # SE

    for _, (dx, dy), col in arrows:
        ex, ey = arrow_endpoint_to_edge(start, start, dx, dy, n)
        ax0.annotate(
            "",
            xy=(ex, ey),
            xytext=(start, start),
            arrowprops=dict(arrowstyle="->", lw=3, color=col),
        )

    ax0.plot([start], [start], marker="o", markersize=4, color="white", zorder=5)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # Right: lineout curves
    ax1 = fig.add_subplot(gs[1])
    for d in lineouts:
        ax1.plot(d["x"], d["y"], lw=2, color=d["color"], label=d["label"])
    ax1.set_title(f"Lineouts from t₁=t₂={start}")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("TTC value")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Fitting: Option 3 (baseline term) using grid over (omega, tau)
# ============================================================

def fit_damped_cosine_with_linear_baseline(
    t: np.ndarray,
    y: np.ndarray,
    omega_grid: np.ndarray,
    tau_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit:
      y(t)= C + b*t + exp(-t/tau)*(a*cos(omega*t) + s*sin(omega*t))

    For any fixed (omega, tau), (C, b, a, s) are solved by weighted least squares.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if weights is None:
        w = np.ones_like(t)
    else:
        w = np.asarray(weights, float)
        w = np.clip(w, 0.0, np.inf)

    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(w)
    t, y, w = t[m], y[m], w[m]

    sw = np.sqrt(w)

    best: Dict[str, Any] = {"sse": np.inf}

    for tau in tau_grid:
        tau = float(tau)
        if tau <= 0:
            continue
        e = np.exp(-t / tau)

        for omega in omega_grid:
            omega = float(omega)
            coswt = np.cos(omega * t)
            sinwt = np.sin(omega * t)

            # [1, t, e*cos, e*sin] * [C, b, a, s]^T
            A = np.column_stack([np.ones_like(t), t, e * coswt, e * sinwt])

            Aw = A * sw[:, None]
            yw = y * sw

            coeff, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
            C, b, a, s = coeff

            yhat = A @ coeff
            resid = y - yhat
            sse = float(np.sum(w * resid * resid))

            if sse < best["sse"]:
                R = float(np.hypot(a, s))
                # a*cos + s*sin = R*cos(ωt + φ) with φ = atan2(-s, a)
                phi = float(np.arctan2(-s, a))

                best = dict(
                    C=float(C),
                    b=float(b),
                    a=float(a),
                    s=float(s),
                    R=R,
                    phi=phi,
                    omega=omega,
                    tau=tau,
                    yhat=yhat,
                    sse=sse,
                )

    return best


def evaluate_model(t: np.ndarray, C: float, b: float, omega: float, tau: float, a: float, s: float) -> np.ndarray:
    t = np.asarray(t, float)
    e = np.exp(-t / tau)
    return C + b * t + e * (a * np.cos(omega * t) + s * np.sin(omega * t))


# ============================================================
# Extract + fit + plot (anti-diagonal through diagonal anchor)
# ============================================================

def extract_fit_antidiagonal_with_ttc_plot(
    sample_id: str,
    base_dir: Path,
    *,
    mask_n: int,
    start_time_idx: int,
    dt_s: float = 1.0,
    omega_min: float = 2 * np.pi / 500.0,
    omega_max: float = 2 * np.pi / 20.0,
    n_omega: int = 260,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    n_tau: int = 120,
    clip_hi_percentile: float = 99.9,
    despike_at_start: bool = True,
    despike_halfwidth: int = 1,
    use_weights: bool = True,
) -> Dict[str, Any]:
    """
    Extract the anti-diagonal lineout passing through (i,i) where i=start_time_idx:
      (t1, t2) = (i-k, i+k), k>=0, staying in-bounds

    Then fit Option 3:
      y(t)= C + b*t + exp(-t/tau)*(a*cos(omega*t) + s*sin(omega*t))

    Produces a 1x2 figure:
      Left: TTC with arrow along the extracted anti-diagonal
      Right: lineout + best-fit curve
    """
    data = load_xpcs_arrays(sample_id, base_dir, mask_n=mask_n)

    C = symmetrize_ttc(data.ttc)
    if despike_at_start:
        C = despike_patch_with_local_median(C, center=(start_time_idx, start_time_idx), halfwidth=despike_halfwidth)
    C = clip_ttc(C, p_hi=clip_hi_percentile)

    N = C.shape[0]
    i = int(start_time_idx)
    if not (0 <= i < N):
        raise ValueError(f"start_time_idx must be in [0, {N-1}]")

    # Anti-diagonal through (i,i): (i-k, i+k)
    kmax = min(i, N - 1 - i)
    ks = np.arange(0, kmax + 1, dtype=int)
    t1 = i - ks
    t2 = i + ks
    y = C[t2, t1]

    tau_idx = 2 * ks
    t = tau_idx.astype(np.float64) * float(dt_s)

    # Grids for omega/tau
    omega_grid = np.linspace(float(omega_min), float(omega_max), int(n_omega))

    if tau_min is None:
        tau_min = max(2.0 * dt_s, 0.02 * (t.max() if t.size else 1.0))
    if tau_max is None:
        tau_max = max(5.0 * tau_min, 2.0 * (t.max() if t.size else 1.0))
    tau_grid = np.geomspace(float(tau_min), float(tau_max), int(n_tau))

    # ----------------------------
    # FFT analysis (NEW – put it HERE)
    # ----------------------------
    dt_fft = 2.0 * dt_s
    freqs, power, f_peak, period_s, p_peak = fft_peak_from_lineout(
        y,
        dt_fft,
        detrend=True,
        window="hann",
        fmin=1 / 1000,  # periods ≤ 1000 s
        fmax=1 / 10,  # periods ≥ 10 s
    )

    print(
        f"FFT peak: f = {f_peak:.4g} Hz  "
        f"(period ≈ {period_s:.2f} s)"
    )

    # Optional: weight early times more (often helps)
    weights = None
    if use_weights and t.size:
        weights = np.exp(-t / (0.7 * t.max()))

    fit = fit_damped_cosine_with_linear_baseline(t, y, omega_grid, tau_grid, weights=weights)

    # Summary
    period = (2 * np.pi / fit["omega"]) if fit["omega"] != 0 else np.inf
    print("Fit parameters (baseline + damped cosine):")
    print(f"  C      = {fit['C']:.6g}")
    print(f"  b      = {fit['b']:.6g}")
    print(f"  tau    = {fit['tau']:.6g} s")
    print(f"  omega  = {fit['omega']:.6g} rad/s  (period ≈ {period:.2f} s)")
    print(f"  R      = {fit['R']:.6g}")
    print(f"  phi    = {fit['phi']:.6g} rad")
    print(f"  SSE    = {fit['sse']:.6g}")

    # Plot
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.35}
    )

    # Left: TTC + arrow
    im = ax0.imshow(C, origin="lower", cmap="plasma", interpolation="nearest")
    ax0.set_title(f"{sample_id}  M{mask_n}  TTC")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    end_t1 = i - kmax
    end_t2 = i + kmax
    ax0.annotate(
        "",
        xy=(end_t2, end_t1),
        xytext=(i, i),
        arrowprops=dict(arrowstyle="->", lw=2.8, color="tab:purple"),
    )
    ax0.plot(i, i, "o", color="white", ms=6)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # Right: lineout + fit
    ax1.plot(t, y, lw=2, color="tab:purple", label="anti-diagonal lineout")
    ax1.plot(t, fit["yhat"], lw=2, color="black", ls="--", label="fit")
    ax1.set_title("Anti-diagonal lineout + fit")
    ax1.set_xlabel("τ (s)")
    ax1.set_ylabel("TTC value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.show()

    return fit

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def symmetrize_ttc(ttc: np.ndarray) -> np.ndarray:
    """Mirror TTC along x=y."""
    ttc = np.asarray(ttc, dtype=np.float64)
    return ttc + ttc.T - np.diag(np.diag(ttc))


def clip_ttc(C: np.ndarray, p_hi: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(C, [0.0, p_hi])
    return np.clip(C, lo, hi)

def extract_antidiagonal_lineout(
    ttc: np.ndarray,
    *,
    start_idx: int,
    dt_s: float,
    clip_percentile: float | None = 99.9,
):
    C = symmetrize_ttc(ttc)

    if clip_percentile is not None:
        lo, hi = np.percentile(C, [0, clip_percentile])
        C = np.clip(C, lo, hi)

    N = C.shape[0]
    i = int(start_idx)
    if not (0 <= i < N):
        raise ValueError("start_idx outside TTC bounds")

    kmax = min(i, N - 1 - i)
    ks = np.arange(0, kmax + 1)

    t1 = i - ks
    t2 = i + ks
    y = C[t2, t1]

    tau_idx = 2 * ks
    t = tau_idx * dt_s

    return t, y


def estimate_fft(t: np.ndarray, y: np.ndarray, *, drop_first: int = 0, detrend: bool = True, window: bool = True):
    """
    Returns freqs (Hz), power, f_peak (Hz), period (s)
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if drop_first > 0:
        t = t[drop_first:]
        y = y[drop_first:]

    if len(t) < 4:
        raise ValueError("Need at least 4 points for FFT.")

    dt = float(np.median(np.diff(t)))
    yy = y.copy()

    if detrend:
        A = np.column_stack([np.ones_like(t), t])
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        yy = yy - (A @ beta)

    if window:
        yy = yy * np.hanning(len(yy))

    Y = np.fft.rfft(yy)
    freqs = np.fft.rfftfreq(len(yy), d=dt)
    power = (Y.real**2 + Y.imag**2)

    if len(power) > 0:
        power[0] = 0.0  # remove DC

    k = int(np.argmax(power)) if len(power) else 0
    f_peak = float(freqs[k]) if len(freqs) else 0.0
    period = (1.0 / f_peak) if f_peak > 0 else np.inf
    return freqs, power, f_peak, period


# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------

def plot_ttc_lineout_fft(
    data,
    *,
    start_idx: int,
    dt_s: float = 1.0,
    clip_hi_percentile: float = 99.9,
    cmap: str = "plasma",
    arrow_color: str = "C2",
    drop_first: int = 0,
    detrend: bool = True,
    window: bool = True,
    figsize=(13, 5.5),
):
    """
    Figure layout:
      - Left (spans full height): TTC plot with anti-diagonal arrow (C2)
      - Right-top: lineout vs tau
      - Right-bottom: FFT power spectrum

    Uses data = load_xpcs_arrays(sample_id, BASE_DIR, mask_n=mask_n)
    where data.ttc is the TTC array.
    """
    C = symmetrize_ttc(data.ttc)
    Cplot = clip_ttc(C, p_hi=clip_hi_percentile)

    t, y = extract_antidiagonal_lineout(
        data.ttc,
        start_idx=start_idx,
        dt_s=dt_s,
        clip_percentile=clip_hi_percentile,
    )
    # keep kmax separately for the arrow length:
    N = C.shape[0]
    i = int(start_idx)
    kmax = min(i, N - 1 - i)

    freqs, power, f_peak, period = estimate_fft(
        t, y, drop_first=drop_first, detrend=detrend, window=window
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], wspace=0.35, hspace=0.35)

    # -------------------------
    # LEFT: TTC + arrow
    # -------------------------
    ax0 = fig.add_subplot(gs[:, 0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title(f"TTC, start t1=t2={start_idx}")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    # anti-diagonal through (i,i): endpoints (i-kmax, i+kmax)
    i = int(start_idx)
    end_t1 = i - kmax
    end_t2 = i + kmax

    ax0.add_patch(
        FancyArrowPatch(
            (i, i),
            (end_t2, end_t1),
            arrowstyle="->",
            linewidth=3,
            mutation_scale=14,
            color=arrow_color,
        )
    )
    ax0.plot([i], [i], marker="o", markersize=5, color="white", zorder=5)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # -------------------------
    # RIGHT-TOP: lineout
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, y, lw=2, color=arrow_color, label="anti-diagonal lineout")
    ax1.set_title("Anti-diagonal lineout")
    ax1.set_xlabel("τ (s)")
    ax1.set_ylabel("TTC value")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # -------------------------
    # RIGHT-BOTTOM: FFT
    # -------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(freqs, power, lw=1.8, color="black")
    ax2.set_title(f"FFT power (peak period ≈ {period:.2f} s)" if np.isfinite(period) else "FFT power")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_yscale("log")
    ax2.set_xlim([-0.001, 0.01])
    ax2.grid(True, alpha=0.3)

    # mark peak
    if f_peak > 0:
        ax2.axvline(f_peak, lw=1.5, color=arrow_color, alpha=0.9)

    plt.show()
    return {
        "t": t,
        "y": y,
        "freqs": freqs,
        "power": power,
        "f_peak_hz": f_peak,
        "period_s": period,
    }

def fft_peak_from_lineout(
    y: np.ndarray,
    dt_s: float,
    *,
    detrend: bool = True,
    window: str = "hann",
    fmin: float | None = None,
    fmax: float | None = None,
):
    """
    Compute FFT power spectrum for a 1D signal and return the dominant peak.

    No zero-padding (n_fft = N), so no artificial interpolation.
    Uses mean removal + optional linear detrend + optional windowing.
    """
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if N < 8:
        raise ValueError("Need at least ~8 points for a meaningful FFT")

    # 1) remove DC
    x = y - np.mean(y)

    # 2) optional linear detrend
    if detrend:
        t = np.arange(N, dtype=np.float64)
        # Fit x ~ a*t + b, subtract
        a, b = np.polyfit(t, x, 1)
        x = x - (a * t + b)

    # 3) window (reduces leakage => typically makes the main peak taller/cleaner)
    if window is None or window.lower() == "none":
        w = np.ones(N, dtype=np.float64)
    elif window.lower() in ("hann", "hanning"):
        w = np.hanning(N)
    elif window.lower() == "hamming":
        w = np.hamming(N)
    elif window.lower() == "blackman":
        w = np.blackman(N)
    else:
        raise ValueError(f"Unknown window: {window}")

    xw = x * w

    # 4) rFFT (real FFT)
    F = np.fft.rfft(xw)  # length N//2+1
    freqs = np.fft.rfftfreq(N, d=dt_s)

    # Power spectrum (magnitude^2). Normalize by window power so comparisons are saner.
    # This does NOT change the peak frequency, just makes amplitudes comparable.
    W = np.sum(w**2)
    power = (np.abs(F) ** 2) / W

    # 5) Peak pick (ignore DC bin at 0 Hz)
    mask = np.ones_like(freqs, dtype=bool)
    mask[0] = False

    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    if not np.any(mask):
        raise ValueError("No frequencies left after applying fmin/fmax")

    k_peak = np.argmax(power[mask])
    idxs = np.flatnonzero(mask)
    k = idxs[k_peak]

    f_peak = freqs[k]
    p_peak = power[k]
    period_s = (1.0 / f_peak) if f_peak > 0 else np.inf

    return freqs, power, f_peak, period_s, p_peak

def detrend_linear(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - (A @ beta)

def window_fn(name: str, n: int) -> np.ndarray:
    name = (name or "").lower()
    if name in ("hann", "hanning"):
        return np.hanning(n)
    if name in ("hamming",):
        return np.hamming(n)
    return np.ones(n)

def segment_indices(n: int, seg_len: int, overlap: float) -> list[tuple[int,int]]:
    step = max(1, int(round(seg_len * (1 - overlap))))
    idx = []
    for s in range(0, n - seg_len + 1, step):
        idx.append((s, s + seg_len))
    return idx

def periodogram(y: np.ndarray, dt: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    n = len(y)
    w = window_fn(window, n)
    yw = (y - np.mean(y)) * w
    # rfft
    Y = np.fft.rfft(yw)
    # power (not absolute scaling-critical for peak picking)
    P = (np.abs(Y) ** 2)
    f = np.fft.rfftfreq(n, d=dt)
    return f, P

def peak_frequency_from_psd(f: np.ndarray, P: np.ndarray, fmin: float, fmax: float) -> float:
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        raise ValueError("No frequencies in band")
    i = np.argmax(P[m])
    return f[m][i]

def bootstrap_peak_frequency(
    y: np.ndarray,
    dt: float,
    *,
    fmin: float,
    fmax: float,
    seg_len: int | None = None,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: bool = True,
    n_boot: int = 2000,
    ci: float = 0.68,
    rng_seed: int = 0,
):
    """
    Returns:
      f_hat (median), f_lo, f_hi, f_samples
    """
    y = np.asarray(y, float)
    if detrend:
        y = detrend_linear(y)

    n = len(y)
    if seg_len is None:
        # sensible default: ~1/4 of record (at least 64 points)
        seg_len = max(64, n // 4)
    seg_len = min(seg_len, n)

    idx = segment_indices(n, seg_len, overlap)
    if len(idx) < 3:
        # fallback: use whole record as one segment (uncertainty will be meaningless)
        f, P = periodogram(y, dt, window=window)
        f0 = peak_frequency_from_psd(f, P, fmin, fmax)
        return f0, f0, f0, np.array([f0])

    # Compute PSD per segment
    f_ref = None
    Ps = []
    for (a, b) in idx:
        f, P = periodogram(y[a:b], dt, window=window)
        if f_ref is None:
            f_ref = f
        else:
            # same length => same freq grid
            pass
        Ps.append(P)
    Ps = np.stack(Ps, axis=0)  # (K, nf)
    f_ref = np.asarray(f_ref)

    # Bootstrap resample segments
    rng = np.random.default_rng(rng_seed)
    K = Ps.shape[0]
    f_samp = np.empty(n_boot, float)

    for i in range(n_boot):
        picks = rng.integers(0, K, size=K)  # resample K segments with replacement
        Pmean = Ps[picks].mean(axis=0)
        f_samp[i] = peak_frequency_from_psd(f_ref, Pmean, fmin, fmax)

    f_hat = float(np.median(f_samp))
    alpha = (1 - ci) / 2
    f_lo = float(np.quantile(f_samp, alpha))
    f_hi = float(np.quantile(f_samp, 1 - alpha))
    return f_hat, f_lo, f_hi, f_samp

# def fft_peak_with_bin_uncertainty(y: np.ndarray, dt_s: float, *, drop_first: int = 0):
#     """
#     Peak frequency from plain FFT + uncertainty from bin width (no segmentation).
#     Returns f_peak, sigma_f, period, sigma_period, delta_f.
#     """
#     y = np.asarray(y, dtype=np.float64)
#     if drop_first:
#         y = y[drop_first:]
#
#     N = y.size
#     if N < 8:
#         raise ValueError("Need at least ~8 points for FFT")
#
#     delta_f = 1.0 / (N * dt_s)          # FFT bin spacing
#     sigma_f = 0.5 * delta_f             # ~half-bin uncertainty
#
#     # Use your existing peak picker so it's consistent
#     freqs, power, f_peak, period_s, _ = fft_peak_from_lineout(
#         y, dt_s,
#         detrend=True,
#         window="hann",
#         fmin=1/1000,
#         fmax=1/10,
#     )
#
#     sigma_T = (sigma_f / (f_peak**2)) if f_peak > 0 else np.inf
#     return f_peak, sigma_f, period_s, sigma_T, delta_f

def fft_peak_with_bin_uncertainty(
    y: np.ndarray,
    dt_fft: float,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
    detrend: bool = True,
    window: str = "hann",
):
    """
    Peak frequency from FFT, plus bin-width uncertainty.

    Uncertainty model:
      df = 1 / (N * dt_fft)
      f ~ f_peak ± df/2

    Returns
    -------
    f_peak, f_lo, f_hi, period, period_lo, period_hi, df
    """
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if N < 8:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    freqs, power, f_peak, period_s, p_peak = fft_peak_from_lineout(
        y, dt_fft,
        detrend=detrend,
        window=window,
        fmin=fmin,
        fmax=fmax,
    )

    df = 1.0 / (N * dt_fft)  # FFT frequency bin spacing
    f_lo = f_peak - 0.5 * df
    f_hi = f_peak + 0.5 * df

    # Respect any user band limits (optional)
    if fmin is not None:
        f_lo = max(f_lo, fmin)
        f_hi = max(f_hi, fmin)
    if fmax is not None:
        f_lo = min(f_lo, fmax)
        f_hi = min(f_hi, fmax)

    # Convert to period band. (Note inversion flips bounds.)
    # Use conservative mapping: period_hi corresponds to the lower frequency bound.
    if f_lo <= 0:
        period_hi = np.nan
    else:
        period_hi = 1.0 / f_lo

    if f_hi <= 0:
        period_lo = np.nan
    else:
        period_lo = 1.0 / f_hi

    period = 1.0 / f_peak if f_peak > 0 else np.nan

    return f_peak, f_lo, f_hi, period, period_lo, period_hi, df


def extract_antidiagonal_lineout_y_only(Csym: np.ndarray, start_idx: int, *, drop_first: int = 0):
    """
    Extract y along anti-diagonal through (i,i): (t1=i-k, t2=i+k).
    Returns y (optionally dropping first points).
    """
    Csym = np.asarray(Csym, dtype=np.float64)
    n = Csym.shape[0]
    i = int(start_idx)
    if Csym.ndim != 2 or Csym.shape[0] != Csym.shape[1]:
        raise ValueError(f"TTC must be square, got {Csym.shape}")
    if not (0 <= i < n):
        raise ValueError(f"start_idx must be in [0, {n-1}]")

    kmax = min(i, n - 1 - i)
    ks = np.arange(0, kmax + 1, dtype=int)
    t1 = i - ks
    t2 = i + ks
    y = Csym[t2, t1]

    if drop_first > 0:
        y = y[drop_first:]

    return y


def plot_period_vs_diagonal_start(
    data,
    *,
    dt_s: float,
    start_idxs: np.ndarray | None = None,
    # preprocessing
    clip_hi_percentile: float = 99.9,
    drop_first_lineout: int = 0,
    drop_first_horizontal: int = 0,  # keep this 0 for your use-case
    # smoothing
    half_window: int = 5,            # <-- NEW: n lineouts either side
    band_ci: float = 0.68,           # <-- NEW: central CI for shaded band (0.68 ~ 1σ)
    # FFT options
    fmin: float | None = 1/1000,
    fmax: float | None = 1/10,
    detrend: bool = True,
    window: str = "hann",
    # plotting
    cmap: str = "plasma",
    figsize=(13, 5.5),
):
    """
    One figure:
      Left: TTC with diagonal start positions marked in C2
      Right: smoothed peak period vs start time with shaded uncertainty band

    Notes
    -----
    Anti-diagonal lag is tau = 2*k*dt_s, so FFT sampling interval is:
      dt_fft = 2*dt_s
    """
    # --- TTC prep (symmetrize + clip for display/lineouts) ---
    C = symmetrize_ttc(data.ttc)
    Cplot = clip_ttc(C, p_hi=clip_hi_percentile)

    n = C.shape[0]
    if start_idxs is None:
        lo = int(0.05 * (n - 1))
        hi = int(0.95 * (n - 1))
        start_idxs = np.linspace(lo, hi, 80).astype(int)
        start_idxs = np.unique(start_idxs)
    else:
        start_idxs = np.unique(np.asarray(start_idxs, dtype=int))

    # FFT sampling interval for anti-diagonal lineout
    dt_fft = 2.0 * float(dt_s)

    # ------------------------------------------------------------
    # Step 1: compute a RAW period estimate at each start index
    # ------------------------------------------------------------
    raw_period = np.full(start_idxs.shape, np.nan, dtype=float)

    for k, i in enumerate(start_idxs):
        y = extract_antidiagonal_lineout_y_only(C, int(i), drop_first=drop_first_lineout)

        if y.size < 16:
            continue

        f_peak, f_lo, f_hi, period, period_lo, period_hi, df = fft_peak_with_bin_uncertainty(
            y, dt_fft,
            fmin=fmin, fmax=fmax,
            detrend=detrend,
            window=window,
        )

        if np.isfinite(period) and period > 0:
            raw_period[k] = period

    # ------------------------------------------------------------
    # Step 2: smooth by pooling +/- half_window neighbors
    #         and compute a robust uncertainty band from quantiles
    # ------------------------------------------------------------
    half_window = int(max(0, half_window))
    smooth_period = np.full_like(raw_period, np.nan)
    band_lo = np.full_like(raw_period, np.nan)
    band_hi = np.full_like(raw_period, np.nan)

    # quantiles for central CI
    alpha = (1.0 - float(band_ci)) / 2.0
    q_lo = alpha
    q_hi = 1.0 - alpha

    for k in range(len(start_idxs)):
        a = max(0, k - half_window)
        b = min(len(start_idxs), k + half_window + 1)

        window_vals = raw_period[a:b]
        window_vals = window_vals[np.isfinite(window_vals)]

        if window_vals.size < 3:
            continue

        # center estimate
        smooth_period[k] = float(np.median(window_vals))

        # uncertainty band (robust)
        band_lo[k] = float(np.quantile(window_vals, q_lo))
        band_hi[k] = float(np.quantile(window_vals, q_hi))

    # Convert start idx -> seconds
    starts_s = start_idxs.astype(float) * float(dt_s)

    # Keep only finite points for plotting
    m = np.isfinite(smooth_period)
    starts_s_m = starts_s[m]
    smooth_m = smooth_period[m]
    blo_m = band_lo[m]
    bhi_m = band_hi[m]
    start_idxs_used = start_idxs[m]

    # --- plot ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    # Left: TTC with start points
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title(SAMPLE_ID + " mask " + str(MASK_N) + " TTC + diagonal start positions")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    ax0.plot(start_idxs_used, start_idxs_used, "o", ms=3.5, color="C2", alpha=0.9)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # Right: smoothed period vs start time with band
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(starts_s_m, smooth_m, lw=2.4, color="C2", label=f"median over ±{half_window}")

    mm = np.isfinite(blo_m) & np.isfinite(bhi_m)
    ax1.fill_between(
        starts_s_m[mm],
        blo_m[mm],
        bhi_m[mm],
        color="C2",
        alpha=0.25,
        linewidth=0,
        label=f"{int(band_ci*100)}% window band",
    )

    ax1.set_title(SAMPLE_ID + " M" + str(MASK_N) + " peak period vs diagonal start time")
    ax1.set_xlabel("Diagonal start time  (t = start_idx · dt_s)  [s]")
    ax1.set_ylabel("Peak period  [s]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return {
        "start_idx": start_idxs_used,
        "start_time_s": starts_s_m,
        "period_raw_s": raw_period,
        "period_smooth_s": smooth_m,
        "band_lo_s": blo_m,
        "band_hi_s": bhi_m,
        "dt_fft_used_s": dt_fft,
        "half_window": half_window,
        "band_ci": band_ci,
    }

def peak_period_from_start_idx(data, start_idx, dt_s, *, fmin, fmax):
    t, y = extract_antidiagonal_lineout(
        data.ttc, start_idx=int(start_idx), dt_s=float(dt_s), clip_percentile=None
    )
    dt_fft = 2.0 * float(dt_s)  # anti-diagonal sampling
    _, _, f_peak, period_s, _ = fft_peak_from_lineout(
        y, dt_fft,
        detrend=True, window="hann", fmin=fmin, fmax=fmax
    )
    return period_s

def smoothed_peak_period_vs_time(
    data,
    start_idxs: np.ndarray,
    *,
    dt_s: float,
    half_window: int = 5,
    fmin: float | None = 1 / 1000,
    fmax: float | None = 1 / 10,
    drop_first: int = 0,
    detrend: bool = True,
    window: str = "hann",
    clip_hi_percentile: float = 99.9,
):
    """
    For each diagonal start index i, average FFT peak periods over
    start indices [i-half_window, ..., i+half_window].

    Returns:
        period, period_lo, period_hi  (arrays, same length as start_idxs)
    """
    C = symmetrize_ttc(data.ttc)
    C = clip_ttc(C, p_hi=clip_hi_percentile)

    start_idxs = np.asarray(start_idxs, dtype=int)
    n = C.shape[0]

    dt_fft = 2.0 * float(dt_s)

    period = np.full(len(start_idxs), np.nan)
    period_lo = np.full(len(start_idxs), np.nan)
    period_hi = np.full(len(start_idxs), np.nan)

    for j, i0 in enumerate(start_idxs):
        # neighborhood of diagonal starts
        neigh = start_idxs[
            (start_idxs >= i0 - half_window) &
            (start_idxs <= i0 + half_window)
        ]

        periods_local = []

        for i in neigh:
            if i < 0 or i >= n:
                continue

            y = extract_antidiagonal_lineout_y_only(
                C, int(i), drop_first=drop_first
            )

            if y.size < 16:
                continue

            try:
                f_peak, _, _, p, _, _, _ = fft_peak_with_bin_uncertainty(
                    y,
                    dt_fft,
                    fmin=fmin,
                    fmax=fmax,
                    detrend=detrend,
                    window=window,
                )
            except Exception:
                continue

            if np.isfinite(p):
                periods_local.append(p)

        if len(periods_local) == 0:
            continue

        periods_local = np.asarray(periods_local)
        period[j] = np.median(periods_local)
        period_lo[j] = np.percentile(periods_local, 16)
        period_hi[j] = np.percentile(periods_local, 84)

    return period, period_lo, period_hi

def extract_horizontal_lineout_y_only(
    C: np.ndarray,
    start_idx: int,
    *,
    drop_first: int = 0,
) -> np.ndarray:
    """
    Horizontal lineout at fixed t2=i, from t1=0..i (ends at the diagonal).
    C indexed as C[t2, t1].
    """
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    i = int(start_idx)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"TTC must be square, got {C.shape}")
    if not (0 <= i < n):
        raise ValueError(f"start_idx must be in [0, {n-1}]")

    y = C[i, 0:i + 1]  # row=t2=i, col=t1=0..i

    if drop_first > 0:
        y = y[int(drop_first):]

    return y.astype(np.float64)


def _extract_antidiagonal_y_only(
    C: np.ndarray,
    start_idx: int,
    *,
    drop_first: int = 0,
) -> np.ndarray:
    """
    Anti-diagonal through (t1=i, t2=i): (t1=i-k, t2=i+k), k>=0.
    IMPORTANT: C is indexed as C[t2, t1] (row=t2, col=t1).
    """
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    i = int(start_idx)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"TTC must be square, got {C.shape}")
    if not (0 <= i < n):
        raise ValueError(f"start_idx must be in [0, {n-1}]")

    kmax = min(i, n - 1 - i)
    ks = np.arange(0, kmax + 1, dtype=int)

    t1 = i - ks
    t2 = i + ks

    y = C[t2, t1]  # <-- FIXED (row=t2, col=t1)

    if drop_first > 0:
        y = y[int(drop_first):]

    return y.astype(np.float64)


def _rolling_smooth_with_band(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple local smoothing across start index:
      - y_smooth: rolling mean of y
      - lo_smooth: rolling min of lo
      - hi_smooth: rolling max of hi
    (keeps a conservative uncertainty band)
    """
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)

    n = len(y)
    ys = np.full(n, np.nan, float)
    los = np.full(n, np.nan, float)
    his = np.full(n, np.nan, float)

    for i in range(n):
        a = max(0, i - int(half_window))
        b = min(n, i + int(half_window) + 1)
        m = np.isfinite(y[a:b]) & np.isfinite(lo[a:b]) & np.isfinite(hi[a:b])
        if not np.any(m):
            continue
        ys[i] = float(np.mean(y[a:b][m]))
        los[i] = float(np.min(lo[a:b][m]))
        his[i] = float(np.max(hi[a:b][m]))
    return ys, los, his

def bootstrap_peak_frequency_fixedbin(
    y: np.ndarray,
    dt: float,
    *,
    fmin: float,
    fmax: float,
    peak_halfwidth_bins: int = 2,   # search only near the global peak
    seg_len: int | None = None,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: bool = True,
    n_boot: int = 2000,
    ci: float = 0.68,
    rng_seed: int = 0,
):
    """
    Bootstrap CI for peak frequency, but prevents peak-hopping by
    restricting the peak search to a small neighborhood around the
    full-record peak.

    Returns: f_hat, f_lo, f_hi, f_samples
    """
    y = np.asarray(y, float)
    if detrend:
        y = detrend_linear(y)

    n = len(y)
    if seg_len is None:
        seg_len = max(64, n // 4)
    seg_len = min(seg_len, n)

    idx = segment_indices(n, seg_len, overlap)
    if len(idx) < 3:
        # fallback: use whole record
        f, P = periodogram(y, dt, window=window)
        m = (f >= fmin) & (f <= fmax)
        if not np.any(m):
            raise ValueError("No frequencies in band")
        k0 = np.argmax(P[m])
        f0 = float(f[m][k0])
        return f0, f0, f0, np.array([f0])

    # --- full-record PSD to define the reference peak bin ---
    f_full, P_full = periodogram(y, dt, window=window)
    band = (f_full >= fmin) & (f_full <= fmax)
    if not np.any(band):
        raise ValueError("No frequencies in band")
    band_idxs = np.flatnonzero(band)
    k_band_peak = band_idxs[np.argmax(P_full[band])]
    k0 = int(k_band_peak)

    # neighborhood search bins
    k_lo = max(1, k0 - int(peak_halfwidth_bins))
    k_hi = min(len(f_full) - 1, k0 + int(peak_halfwidth_bins))
    neigh = np.arange(k_lo, k_hi + 1)

    # --- PSD per segment on the same frequency grid ---
    Ps = []
    for (a, b) in idx:
        f_seg, P_seg = periodogram(y[a:b], dt, window=window)
        Ps.append(P_seg)
    Ps = np.stack(Ps, axis=0)  # (K, nf)

    rng = np.random.default_rng(rng_seed)
    K = Ps.shape[0]
    f_samp = np.empty(n_boot, float)

    for i in range(n_boot):
        picks = rng.integers(0, K, size=K)
        Pmean = Ps[picks].mean(axis=0)

        # pick peak ONLY within neighborhood
        kk = neigh[np.argmax(Pmean[neigh])]
        f_samp[i] = float(f_full[kk])

    f_hat = float(np.median(f_samp))
    alpha = (1.0 - float(ci)) / 2.0
    f_lo = float(np.quantile(f_samp, alpha))
    f_hi = float(np.quantile(f_samp, 1.0 - alpha))
    return f_hat, f_lo, f_hi, f_samp


def plot_period_vs_diagonal_start_both_lineouts(
    data,
    *,
    dt_s: float,
    start_idxs: np.ndarray | None = None,
    # preprocessing / extraction
    clip_hi_percentile: float = 99.9,
    drop_first_antidiag: int = 0,
    drop_first_horizontal: int = 0,
    # smoothing / band (SAME methodology as plot_period_vs_diagonal_start)
    half_window: int = 5,
    band_ci: float = 0.68,
    # FFT options
    fmin: float | None = 1 / 1000,
    fmax: float | None = 1 / 10,
    detrend: bool = True,
    window: str = "hann",
    # plotting
    cmap: str = "plasma",
    figsize=(13.5, 6.0),
):
    """
    Same methodology as plot_period_vs_diagonal_start(), but computes periods for BOTH:
      1) anti-diagonal lineout through (i,i): (t1=i-k, t2=i+k)  -> dt_fft = 2*dt_s
      2) horizontal lineout at t2=i: (t1=0..i)                 -> dt_fft = 1*dt_s

    For each start index:
      - compute RAW period estimate by FFT peak (no bootstrap)
    Then:
      - smooth by pooling ±half_window in *index space*
      - compute robust band via quantiles (central CI = band_ci)
    """
    # --- TTC prep ---
    C = symmetrize_ttc(data.ttc)
    Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))

    def block_average(C, out_n=300):
        """
        Downsample a square matrix C to out_n x out_n by block averaging.
        """
        n = C.shape[0]
        m = n // out_n
        C = C[:out_n * m, :out_n * m]  # trim
        return C.reshape(out_n, m, out_n, m).mean(axis=(1, 3))

    C_small = block_average(Cplot, out_n=300)
    np.savetxt(SAMPLE_ID + 'M' + str(MASK_N) + '.txt', C_small, fmt="%.3f")

    n = C.shape[0]
    if start_idxs is None:
        lo = int(0.05 * (n - 1))
        hi = int(0.95 * (n - 1))
        start_idxs = np.linspace(lo, hi, 90).astype(int)
        start_idxs = np.unique(start_idxs)
    else:
        start_idxs = np.unique(np.asarray(start_idxs, dtype=int))

    # FFT sampling intervals
    dt_fft_anti = 2.0 * float(dt_s)
    dt_fft_horz = 1.0 * float(dt_s)

    # ------------------------------------------------------------
    # Step 1: RAW period at each start index (anti + horizontal)
    # ------------------------------------------------------------
    raw_period_anti = np.full(start_idxs.shape, np.nan, dtype=float)
    raw_period_horz = np.full(start_idxs.shape, np.nan, dtype=float)

    for k, i in enumerate(start_idxs):
        # --- anti-diagonal y ---
        try:
            y_a = extract_antidiagonal_lineout_y_only(C, int(i), drop_first=drop_first_antidiag)
        except Exception:
            continue

        if y_a.size >= 16:
            f_peak, f_lo, f_hi, period, period_lo, period_hi, df = fft_peak_with_bin_uncertainty(
                y_a,
                dt_fft_anti,
                fmin=fmin,
                fmax=fmax,
                detrend=detrend,
                window=window,
            )
            if np.isfinite(period) and period > 0:
                raw_period_anti[k] = float(period)

        # --- horizontal y ---
        try:
            y_h = extract_horizontal_lineout_y_only(C, int(i), drop_first=drop_first_horizontal)
        except Exception:
            continue

        if y_h.size >= 16:
            f_peak, f_lo, f_hi, period, period_lo, period_hi, df = fft_peak_with_bin_uncertainty(
                y_h,
                dt_fft_horz,
                fmin=fmin,
                fmax=fmax,
                detrend=detrend,
                window=window,
            )
            if np.isfinite(period) and period > 0:
                raw_period_horz[k] = float(period)

    # ------------------------------------------------------------
    # Step 2: smooth by pooling ±half_window neighbors
    #         and compute robust band from quantiles
    #         (EXACTLY the same structure as your working function)
    # ------------------------------------------------------------
    half_window = int(max(0, half_window))

    def smooth_with_quantile_band(raw: np.ndarray):
        smooth = np.full_like(raw, np.nan, dtype=float)
        band_lo = np.full_like(raw, np.nan, dtype=float)
        band_hi = np.full_like(raw, np.nan, dtype=float)

        alpha = (1.0 - float(band_ci)) / 2.0
        q_lo = alpha
        q_hi = 1.0 - alpha

        for kk in range(len(start_idxs)):
            a = max(0, kk - half_window)
            b = min(len(start_idxs), kk + half_window + 1)

            window_vals = raw[a:b]
            window_vals = window_vals[np.isfinite(window_vals)]
            if window_vals.size < 3:
                continue

            smooth[kk] = float(np.median(window_vals))
            band_lo[kk] = float(np.quantile(window_vals, q_lo))
            band_hi[kk] = float(np.quantile(window_vals, q_hi))

        return smooth, band_lo, band_hi

    smooth_anti, blo_anti, bhi_anti = smooth_with_quantile_band(raw_period_anti)
    smooth_horz, blo_horz, bhi_horz = smooth_with_quantile_band(raw_period_horz)

    # Convert start idx -> seconds
    starts_s = start_idxs.astype(float) * float(dt_s)

    # For plotting, keep indices where at least one curve is finite
    m_any = np.isfinite(smooth_anti) | np.isfinite(smooth_horz)
    start_idxs_used = start_idxs[m_any]
    starts_s_used = starts_s[m_any]

    # --- plot ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    # Left: TTC with start points
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")
    ax0.set_title(f"{SAMPLE_ID} mask {MASK_N} TTC + diagonal start positions")
    ax0.plot(start_idxs_used, start_idxs_used, "o", ms=3.5, color="C2", alpha=0.9)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    # Right: both periods + bands
    ax1 = fig.add_subplot(gs[1])

    # anti-diagonal (C2)
    mA = np.isfinite(smooth_anti) & np.isfinite(blo_anti) & np.isfinite(bhi_anti)
    ax1.plot(starts_s[mA], smooth_anti[mA], lw=2.4, color="C2", label=f"anti-diag median over ±{half_window}")
    ax1.fill_between(
        starts_s[mA],
        blo_anti[mA],
        bhi_anti[mA],
        color="C2",
        alpha=0.22,
        linewidth=0,
        label=f"anti-diag {int(band_ci*100)}% window band",
    )

    # horizontal (C0)
    mH = np.isfinite(smooth_horz) & np.isfinite(blo_horz) & np.isfinite(bhi_horz)
    ax1.plot(starts_s[mH], smooth_horz[mH], lw=2.4, color="C0", label=f"horizontal median over ±{half_window}")
    ax1.fill_between(
        starts_s[mH],
        blo_horz[mH],
        bhi_horz[mH],
        color="C0",
        alpha=0.18,
        linewidth=0,
        label=f"horizontal {int(band_ci*100)}% window band",
    )

    ax1.set_title(f"{SAMPLE_ID} M{MASK_N} peak periods vs diagonal start time")
    ax1.set_xlabel("Diagonal start time  (t = start_idx · dt_s)  [s]")
    ax1.set_ylabel("Peak period  [s]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return {
        "start_idx": start_idxs_used,
        "start_time_s": starts_s_used,
        # raw
        "anti_period_raw_s": raw_period_anti,
        "horz_period_raw_s": raw_period_horz,
        # smoothed + bands
        "anti_period_smooth_s": smooth_anti[m_any],
        "anti_band_lo_s": blo_anti[m_any],
        "anti_band_hi_s": bhi_anti[m_any],
        "horz_period_smooth_s": smooth_horz[m_any],
        "horz_band_lo_s": blo_horz[m_any],
        "horz_band_hi_s": bhi_horz[m_any],
        # bookkeeping
        "dt_fft_anti_used_s": dt_fft_anti,
        "dt_fft_horz_used_s": dt_fft_horz,
        "half_window": half_window,
        "band_ci": band_ci,
    }

def plot_ttc_and_2dfft(
    ttc: np.ndarray,
    *,
    clip_hi_percentile: float = 99.9,
    dt_s: float = 1.0,
    cmap_ttc: str = "plasma",
    cmap_fft: str = "magma",
    window: bool = True,
    remove_mean: bool = True,
    figsize=(12.5, 5.5),
):
    """
    Side-by-side plot:
      Left : symmetrized TTC
      Right: log-power 2D FFT of TTC

    Parameters
    ----------
    ttc : (N,N) array
        Two-time correlation matrix.
    clip_hi_percentile : float
        Upper percentile clipping for TTC display.
    dt_s : float
        Time step per TTC index (seconds), used for frequency axes.
    window : bool
        Apply 2D Hann window before FFT to suppress edge leakage.
    remove_mean : bool
        Subtract mean of TTC before FFT (highly recommended).
    """

    # -----------------------------
    # TTC preprocessing
    # -----------------------------
    C = symmetrize_ttc(ttc)
    Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))

    N = C.shape[0]

    # -----------------------------
    # FFT preprocessing
    # -----------------------------
    X = C.astype(np.float64)

    if remove_mean:
        X = X - np.nanmean(X)

    X = np.nan_to_num(X, nan=0.0)

    if window:
        w = np.hanning(N)
        W = w[:, None] * w[None, :]
        X = X * W

    # # remove row/column means (kills separable terms)
    # X = (
    #         X
    #         - X.mean(axis=0, keepdims=True)
    #         - X.mean(axis=1, keepdims=True)
    #         + X.mean()
    # )

    # 2D FFT
    F = np.fft.fftshift(np.fft.fft2(X))
    P = np.abs(F) ** 2

    # frequency axes (Hz)
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt_s))

    # --------------------------------
    # restrict FFT display range
    # --------------------------------
    fmax = 0.01  # Hz
    m = (f >= -fmax) & (f <= fmax)

    f_zoom = f[m]
    P_zoom = P[np.ix_(m, m)]

    # masks in FFT space
    eps = 1e-12

    # axis masks
    axis_mask = (np.abs(f_zoom[:, None]) < eps) | (np.abs(f_zoom[None, :]) < eps)

    # diagonal (f1 + f2 = 0)
    F1, F2 = np.meshgrid(f_zoom, f_zoom, indexing="ij")
    diag_mask = np.abs(F1 + F2) < (2 * (f_zoom[1] - f_zoom[0]))

    axis_power = np.sum(P_zoom[axis_mask])
    diag_power = np.sum(P_zoom[diag_mask])

    print(f"Axis power     : {axis_power:.3e}")
    print(f"Diagonal power : {diag_power:.3e}")
    print(f"Ratio A_s/A_d ≈ {axis_power / diag_power:.3f}")

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)

    # ---- TTC ----
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(
        Cplot,
        origin="lower",
        cmap=cmap_ttc,
        interpolation="nearest",
        aspect="equal",
    )
    ax0.set_title("TTC (symmetrized)")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    # ---- 2D FFT ----
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(
        np.log10(P_zoom + 1e-12),
        origin="lower",
        extent=[f_zoom[0], f_zoom[-1], f_zoom[0], f_zoom[-1]],
        cmap=cmap_fft,
        aspect="equal",
    )
    ax1.set_title("2D FFT power  (log scale)")
    ax1.set_xlabel("f₁  [Hz]")
    ax1.set_ylabel("f₂  [Hz]")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    plt.tight_layout()
    plt.show()

    return {
        "ttc_sym": C,
        "fft_power": P,
        "freqs_hz": f,
    }

# -----------------------------
# TTC utilities
# -----------------------------
def symmetrize_ttc(ttc: np.ndarray) -> np.ndarray:
    ttc = np.asarray(ttc, dtype=np.float64)
    return ttc + ttc.T - np.diag(np.diag(ttc))


def clip_ttc(C: np.ndarray, p_hi: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(C[np.isfinite(C)], [0.0, p_hi])
    return np.clip(C, lo, hi)


def maybe_window_2d(X: np.ndarray, window: bool) -> np.ndarray:
    if not window:
        return X
    n = X.shape[0]
    w = np.hanning(n)
    W = w[:, None] * w[None, :]
    return X * W


# -----------------------------
# Model
# -----------------------------
def make_time_axes(n: int, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return t1,t2 grids in seconds, shape (n,n).
    We use t = idx * dt_s.
    """
    t = np.arange(n, dtype=np.float64) * float(dt_s)
    t1 = t[None, :]  # columns
    t2 = t[:, None]  # rows
    return t1, t2


def model_ttc(
    n: int,
    dt_s: float,
    *,
    C0: float,
    A_d: float,
    A_s: float,
    omega_d: float,
    omega_s: float,
) -> np.ndarray:
    t1, t2 = make_time_axes(n, dt_s)
    return (
        C0
        + A_d * np.cos(omega_d * (t2 - t1))
        + A_s * np.cos(omega_s * t1) * np.cos(omega_s * t2)
    )


def fit_amplitudes_linear(
    C: np.ndarray,
    dt_s: float,
    *,
    omega_d: float,
    omega_s: float,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Given omega_d, omega_s, solve for (C0, A_d, A_s) by (weighted) least squares.

    C ~ C0 + A_d * Bd + A_s * Bs
    where:
      Bd = cos(omega_d*(t2-t1))
      Bs = cos(omega_s*t1)*cos(omega_s*t2)
    """
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    t1, t2 = make_time_axes(n, dt_s)

    Bd = np.cos(omega_d * (t2 - t1))
    Bs = np.cos(omega_s * t1) * np.cos(omega_s * t2)

    y = C.reshape(-1)

    A = np.column_stack(
        [
            np.ones_like(y),
            Bd.reshape(-1),
            Bs.reshape(-1),
        ]
    )

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = np.clip(w, 0.0, np.inf)
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        yw = y * sw
        coeff, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    else:
        coeff, *_ = np.linalg.lstsq(A, y, rcond=None)

    C0, A_d, A_s = (float(coeff[0]), float(coeff[1]), float(coeff[2]))
    return {"C0": C0, "A_d": A_d, "A_s": A_s}


def sse_for_omegas(
    C: np.ndarray,
    dt_s: float,
    omega_d: float,
    omega_s: float,
    *,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute SSE after solving amplitudes at (omega_d, omega_s).
    """
    params = fit_amplitudes_linear(C, dt_s, omega_d=omega_d, omega_s=omega_s, weights=weights)
    C0, A_d, A_s = params["C0"], params["A_d"], params["A_s"]
    C_hat = model_ttc(C.shape[0], dt_s, C0=C0, A_d=A_d, A_s=A_s, omega_d=omega_d, omega_s=omega_s)

    R = C - C_hat
    if weights is None:
        sse = float(np.sum(R * R))
    else:
        W = np.asarray(weights, dtype=np.float64)
        sse = float(np.sum(W * R * R))
    return sse, params


# -----------------------------
# FFT-based initial guesses
# -----------------------------
def fft2d_power(
    C: np.ndarray,
    dt_s: float,
    *,
    remove_mean: bool = True,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (f_hz, P, F) where:
      f_hz: 1D frequency axis (Hz), fftshifted, length N
      P:    2D power spectrum |F|^2, shape (N,N), fftshifted
      F:    complex 2D FFT (fftshifted)
    """
    X = np.asarray(C, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)

    if remove_mean:
        X = X - float(np.mean(X))

    X = maybe_window_2d(X, window=window)

    F = np.fft.fftshift(np.fft.fft2(X))
    P = np.abs(F) ** 2
    f_hz = np.fft.fftshift(np.fft.fftfreq(X.shape[0], d=float(dt_s)))
    return f_hz, P, F


def guess_omega_d_from_diag_ridge(
    C: np.ndarray,
    dt_s: float,
    *,
    fmax_hz: float = 0.02,
    diag_band_hz: Optional[float] = None,
) -> float:
    """
    Estimate omega_d by collapsing power along the diagonal ridge f1 + f2 ~ const.
    For cos(omega_d*(t2-t1)), the FFT is concentrated near f1 = -f2 = ±f_d,
    i.e. the anti-diagonal (f1 + f2 = 0) with peaks away from 0.

    We approximate by sampling power along the anti-diagonal line (i, N-1-i).
    """
    f, P, _ = fft2d_power(C, dt_s, remove_mean=True, window=True)
    n = len(f)

    # zoom mask
    m = (f >= -fmax_hz) & (f <= fmax_hz)
    idx = np.flatnonzero(m)
    if idx.size < 8:
        raise ValueError("FFT zoom band too small for this N/dt_s")

    Pz = P[np.ix_(idx, idx)]
    fz = f[idx]
    nz = fz.size

    # anti-diagonal samples (f1 = -f2)
    anti = np.array([Pz[i, nz - 1 - i] for i in range(nz)], dtype=np.float64)

    # ignore the exact center (near DC)
    center = nz // 2
    anti[center] = 0.0

    k = int(np.argmax(anti))
    f_peak = float(abs(fz[k]))
    omega_d = float(2.0 * np.pi * f_peak)
    return omega_d


def guess_omega_s_from_axis_ridge(
    C: np.ndarray,
    dt_s: float,
    *,
    fmax_hz: float = 0.02,
) -> float:
    """
    Estimate omega_s by looking for peaks along the axes (f1=±f_s at f2~0 or vice versa).
    For cos(w t1)cos(w t2), FFT has components at (±f_s, ±f_s) and (±f_s, ∓f_s),
    but in practice you often see strong low-frequency axis structure.
    We do a simple 1D collapse near f2=0 and pick the strongest non-DC peak.
    """
    f, P, _ = fft2d_power(C, dt_s, remove_mean=True, window=True)
    n = len(f)

    m = (f >= -fmax_hz) & (f <= fmax_hz)
    idx = np.flatnonzero(m)
    if idx.size < 8:
        raise ValueError("FFT zoom band too small for this N/dt_s")

    Pz = P[np.ix_(idx, idx)]
    fz = f[idx]
    nz = fz.size

    # take a small band around f2=0 (central rows) and collapse over it
    c = nz // 2
    band = 2  # +/- 2 bins around 0
    r0 = max(0, c - band)
    r1 = min(nz, c + band + 1)

    axis_profile = np.mean(Pz[r0:r1, :], axis=0)
    axis_profile[c] = 0.0  # remove DC

    k = int(np.argmax(axis_profile))
    f_peak = float(abs(fz[k]))
    omega_s = float(2.0 * np.pi * f_peak)
    return omega_s


# -----------------------------
# Main fit routine
# -----------------------------
@dataclass
class FitResult:
    C0: float
    A_d: float
    A_s: float
    omega_d: float
    omega_s: float
    sse: float


def fit_ttc_four_params(
    ttc: np.ndarray,
    *,
    dt_s: float = 1.0,
    downsample: int = 1,
    # FFT guess params
    fmax_guess_hz: float = 0.02,
    # search params (Hz, converted to omega)
    fd_span_hz: float = 0.004,
    fs_span_hz: float = 0.004,
    n_fd: int = 45,
    n_fs: int = 45,
    refine_rounds: int = 2,
) -> FitResult:
    """
    Fit (A_d, A_s, omega_d, omega_s) + C0.

    Strategy:
      - symmetrize TTC (default)
      - optional downsample for speed (fit on smaller matrix)
      - FFT-based initial guesses for omega_d, omega_s
      - coarse-to-fine grid search around those frequencies
      - amplitudes solved exactly by linear LS for each (omega_d, omega_s)
    """
    C = symmetrize_ttc(ttc)

    if downsample > 1:
        C = C[::downsample, ::downsample]

    # remove any NaNs safely
    C = np.nan_to_num(C, nan=float(np.nanmean(C)))

    # initial guesses from FFT
    w_d0 = guess_omega_d_from_diag_ridge(C, dt_s * downsample, fmax_hz=fmax_guess_hz)
    w_s0 = guess_omega_s_from_axis_ridge(C, dt_s * downsample, fmax_hz=fmax_guess_hz)

    # work in Hz for search ranges, convert to omega each evaluation
    f_d0 = w_d0 / (2.0 * np.pi)
    f_s0 = w_s0 / (2.0 * np.pi)

    best = FitResult(C0=float(np.mean(C)), A_d=0.0, A_s=0.0, omega_d=w_d0, omega_s=w_s0, sse=np.inf)

    fd_span = float(fd_span_hz)
    fs_span = float(fs_span_hz)

    for _round in range(int(refine_rounds)):
        fd_grid = np.linspace(max(0.0, f_d0 - fd_span), f_d0 + fd_span, int(n_fd))
        fs_grid = np.linspace(max(0.0, f_s0 - fs_span), f_s0 + fs_span, int(n_fs))

        for fd in fd_grid:
            omega_d = float(2.0 * np.pi * fd)
            for fs in fs_grid:
                omega_s = float(2.0 * np.pi * fs)

                sse, amps = sse_for_omegas(C, dt_s * downsample, omega_d, omega_s, weights=None)

                if sse < best.sse:
                    best = FitResult(
                        C0=amps["C0"],
                        A_d=amps["A_d"],
                        A_s=amps["A_s"],
                        omega_d=omega_d,
                        omega_s=omega_s,
                        sse=sse,
                    )

        # re-center and shrink spans for refinement
        f_d0 = best.omega_d / (2.0 * np.pi)
        f_s0 = best.omega_s / (2.0 * np.pi)
        fd_span *= 0.35
        fs_span *= 0.35

    return best


# -----------------------------
# Plotting
# -----------------------------
def plot_measured_vs_model_ttc(
    ttc: np.ndarray,
    fit: FitResult,
    *,
    dt_s: float = 1.0,
    clip_hi_percentile: float = 99.9,
    cmap: str = "plasma",
    figsize: Tuple[float, float] = (12.5, 5.8),
):
    C_meas = symmetrize_ttc(ttc)
    C_mod = model_ttc(
        C_meas.shape[0],
        dt_s,
        C0=fit.C0,
        A_d=fit.A_d,
        A_s=fit.A_s,
        omega_d=fit.omega_d,
        omega_s=fit.omega_s,
    )
    # keep model symmetrized too (numerically should already be)
    C_mod = symmetrize_ttc(C_mod)

    C_meas_plot = clip_ttc(C_meas, p_hi=float(clip_hi_percentile))
    C_mod_plot = clip_ttc(C_mod, p_hi=float(clip_hi_percentile))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(C_meas_plot, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    ax0.set_title("Measured TTC (symmetrized)")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(C_mod_plot, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    ax1.set_title("Model TTC (fitted)")
    ax1.set_xlabel("t₁ index")
    ax1.set_ylabel("t₂ index")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    plt.tight_layout()
    plt.show()

    return {"C_meas_sym": C_meas, "C_model_sym": C_mod}


# -----------------------------
# Example usage
# -----------------------------
def demo_with_random():
    # synthetic demo (sanity check)
    n = 420
    dt_s = 1.0
    true = dict(C0=1.0, A_d=0.15, A_s=0.08, omega_d=2 * np.pi / 110.0, omega_s=2 * np.pi / 95.0)
    C = model_ttc(n, dt_s, **true)
    C = C + 0.02 * np.random.default_rng(0).standard_normal(C.shape)

    fit = fit_ttc_four_params(C, dt_s=dt_s, downsample=2, fmax_guess_hz=0.02)
    print("Fit:")
    print(f"  C0     = {fit.C0:.6g}")
    print(f"  A_d    = {fit.A_d:.6g}")
    print(f"  A_s    = {fit.A_s:.6g}")
    print(f"  omega_d= {fit.omega_d:.6g} rad/s  (T_d={2*np.pi/fit.omega_d:.2f} s)")
    print(f"  omega_s= {fit.omega_s:.6g} rad/s  (T_s={2*np.pi/fit.omega_s:.2f} s)")
    print(f"  SSE    = {fit.sse:.6g}")

    plot_measured_vs_model_ttc(C, fit, dt_s=dt_s, clip_hi_percentile=99.5)

def _safe_decode(x):
    try:
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return x


def _get_temperature_from_results_hdf(hdf_path):
    """
    Best-effort temperature fetcher. Returns float or None.
    Tries multiple common locations and attributes.
    """
    candidate_paths = [
        "experimental_parameters/temperature",
        "experimental_parameters/T",
        "experiment/temperature",
        "metadata/temperature",
        "entry/sample/temperature",
        "entry/instrument/sample/temperature",
        "entry/sample/temperature_setpoint",
        "entry/sample/temperature_actual",
    ]

    with h5py.File(hdf_path, "r") as f:
        # Try datasets
        for p in candidate_paths:
            if p in f:
                try:
                    v = f[p][()]
                    v = np.asarray(v).squeeze()
                    if v.size == 1:
                        return float(v)
                except Exception:
                    pass

        # Try attrs on some likely groups
        for grp_name in ("entry", "entry/sample", "experimental_parameters", "metadata"):
            if grp_name in f:
                g = f[grp_name]
                for k in ("temperature", "Temperature", "T", "temp", "Temp"):
                    if k in g.attrs:
                        try:
                            v = _safe_decode(g.attrs[k])
                            return float(np.asarray(v).squeeze())
                        except Exception:
                            pass

    return None


def find_brightest_mask_by_integrated_intensity(dynamic_roi_map, scattering_2d):
    """
    Pick the mask label with the largest integrated intensity in scattering_2d.

    dynamic_roi_map: 2D integer label image
    scattering_2d:    2D intensity image (same shape)

    Returns
    -------
    best_label : int
    best_sum   : float
    """
    lab = np.asarray(dynamic_roi_map)
    img = np.asarray(scattering_2d)

    if lab.ndim != 2 or img.ndim != 2 or lab.shape != img.shape:
        raise ValueError(f"Shape mismatch: roi_map {lab.shape}, scattering_2d {img.shape}")

    # labels present (exclude background 0 if it exists)
    labels = np.unique(lab)
    labels = labels[labels != 0]

    best_label = None
    best_sum = -np.inf

    # brute-force but fine for ~300 masks
    for k in labels:
        m = (lab == k)
        s = float(np.nansum(img[m]))
        if s > best_sum:
            best_sum = s
            best_label = int(k)

    if best_label is None:
        raise ValueError("No nonzero ROI labels found in dynamic_roi_map")

    return best_label, best_sum


def plot_A4_17scan_central_brightest_ttcs(
    *,
    base_dir,
    sample_ids,
    clip_hi_percentile=99.9,
    cmap="plasma",
    figsize_per_panel=(2.1, 2.3),
    title_fontsize=9,
    textbox_fontsize=8,
):
    """
    One very wide figure: 17 TTC panels in a row.

    For each scan:
      - load dynamic_roi_map + scattering_2d from results hdf
      - choose brightest mask by integrated intensity
      - load TTC for that mask
      - symmetrize
      - display clipped (optional)
      - annotate with temperature + textbox M<mask> min/max

    Assumes your existing helpers exist:
      - find_results_hdf(base_dir, sample_id)
      - symmetrize_ttc(C)
      - clip_ttc(C, p_hi=...)
    """
    sample_ids = list(sample_ids)
    n_panels = len(sample_ids)

    # Very wide figure
    fig_w = float(figsize_per_panel[0]) * n_panels
    fig_h = float(figsize_per_panel[1])
    fig, axs = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)
    axs = axs[0]

    # To keep comparable visual scaling across all panels, we can compute global clim
    # from the *clipped* versions, but textbox uses unclipped min/max.
    Cplots = []
    panel_meta = []

    for sid in sample_ids:
        hdf_path = find_results_hdf(Path(base_dir), sid)

        # load the bits needed to pick mask and load TTC
        with h5py.File(hdf_path, "r") as f:
            roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
            scat = f["xpcs/temporal_mean/scattering_2d"][...]
            if scat.ndim == 3:
                scat = scat[0, :, :]

            mask_n, _ = find_brightest_mask_by_integrated_intensity(roi_map, scat)

            ttc_path = f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}"
            if ttc_path not in f:
                raise KeyError(f"Missing TTC path {ttc_path} in {hdf_path}")

            C = f[ttc_path][...]

        # symmetrize by default (your preference)
        Csym = symmetrize_ttc(C)

        # min/max for textbox from *unclipped* symmetrized data
        cmin = float(np.nanmin(Csym))
        cmax = float(np.nanmax(Csym))

        # clip only for display (to avoid one panel dominating colormap)
        Cplot = clip_ttc(Csym, p_hi=float(clip_hi_percentile))
        Cplots.append(Cplot)

        T = _get_temperature_from_results_hdf(hdf_path)
        panel_meta.append((sid, mask_n, T, cmin, cmax))

    ims = []
    for ax, Cplot, meta in zip(axs, Cplots, panel_meta):
        sid, mask_n, T, cmin, cmax = meta

        im = ax.imshow(
            Cplot,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )
        ims.append(im)

        hdf_path = find_results_hdf(Path(base_dir), sid)
        temp_str = temperature_str_from_filename(hdf_path)

        if temp_str is None:
            ax.set_title(f"{sid}", fontsize=title_fontsize)
        else:
            ax.set_title(f"{sid} | {temp_str}", fontsize=title_fontsize)

        ax.set_xticks([])
        ax.set_yticks([])

        # Textbox: mask + min/max from UNCLIPPED symmetrized matrix
        txt = f"M{int(mask_n)}\nmin={cmin:.3g}\nmax={cmax:.3g}"
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=textbox_fontsize,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.5"),
        )

    # # One shared colorbar (since we forced shared vmin/vmax)
    # cbar = fig.colorbar(ims[-1], ax=axs, fraction=0.015, pad=0.01)
    # cbar.set_label("TTC (clipped for display)")
    #
    # fig.suptitle(f"A4: central brightest-mask TTCs (n={n_panels})", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()

    return panel_meta

def temperature_str_from_filename(hdf_path):
    """
    Extract e.g. '080K' from ..._080K_..._results.hdf
    """
    name = Path(hdf_path).name
    m = re.search(r"_([0-9]{2,4}K)_", name)
    return m.group(1) if m else None

def _load_ttc_for_mask(hdf_path: Path, mask_n: int) -> np.ndarray:
    ttc_path = f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}"
    with h5py.File(hdf_path, "r") as f:
        if ttc_path not in f:
            raise KeyError(f"Missing TTC path {ttc_path} in {hdf_path}")
        return f[ttc_path][...]


def plot_3x5_brightest_plus_offsets_ttcs(
    *,
    base_dir: Path | None = None,
    sample_ids: list[str] | tuple[str, ...] = ("A029", "A048", "A073", "A088", "A101"),
    clip_hi_percentile: float = 99.9,
    cmap: str = "plasma",
    figsize: tuple[float, float] = (12.0, 7.2),
    save: bool = True,
    out_path: Path | None = None,
    dpi: int = 250,
    label_fontsize: int = 12,
):
    """
    Make a 3x5 TTC figure:
      Row 0: brightest mask
      Row 1: brightest - 1
      Row 2: brightest - 2

    No titles, no axis labels.
    TEMP DEBUG: each panel shows a text label with sample_id and mask number.

    IMPORTANT:
    Each TTC is normalized to its OWN color scale (per-panel vmin/vmax),
    so visual contrast is local to each panel.
    """

    # ----------------------------
    # Resolve base_dir at runtime (prevents NameError)
    # ----------------------------
    if base_dir is None:
        if "BASE_DIR" not in globals():
            raise ValueError(
                "base_dir was None, and BASE_DIR is not defined in this module. "
                "Pass base_dir=Path(...) explicitly."
            )
        base_dir = globals()["BASE_DIR"]

    base_dir = Path(base_dir)

    sample_ids = list(sample_ids)
    if len(sample_ids) != 5:
        raise ValueError("sample_ids must have length 5 for a 3x5 grid")

    # ----------------------------
    # EXACT brightest-mask selection as A4 function
    # ----------------------------
    def _brightest_mask_like_A4(hdf_path: Path) -> int:
        with h5py.File(hdf_path, "r") as f:
            roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
            scat = f["xpcs/temporal_mean/scattering_2d"][...]
            if scat.ndim == 3:
                scat = scat[0, :, :]
        mask_n, _ = find_brightest_mask_by_integrated_intensity(roi_map, scat)
        return int(mask_n)

    # ----------------------------
    # TTC loader (local, explicit)
    # ----------------------------
    def _load_ttc_for_mask(hdf_path: Path, mask_n: int) -> np.ndarray:
        ttc_path = f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}"
        with h5py.File(hdf_path, "r") as f:
            if ttc_path not in f:
                raise KeyError(f"Missing TTC path {ttc_path} in {hdf_path}")
            return f[ttc_path][...]

    # ----------------------------
    # Precompute brightest masks for each sample (keeps row/col logic clean)
    # ----------------------------
    hdf_paths: dict[str, Path] = {}
    brightest_by_sid: dict[str, int] = {}

    for sid in sample_ids:
        hdf_path = find_results_hdf(base_dir, sid)
        hdf_paths[sid] = hdf_path
        brightest_by_sid[sid] = _brightest_mask_like_A4(hdf_path)

    # ----------------------------
    # Plot: 3 rows x 5 cols
    # Row r uses offset r (0,1,2); Col c uses sample_ids[c]
    # ----------------------------
    fig, axes = plt.subplots(3, 5, figsize=figsize)
    axes = np.asarray(axes)

    meta: list[tuple[str, int]] = []  # (sample_id, mask_n_used), row-major (r then c)

    for r, add in enumerate((0, -1, -2)):
        for c, sid in enumerate(sample_ids):
            ax = axes[r, c]

            mask0 = brightest_by_sid[sid]
            mask_n = int(mask0 + add)

            C = _load_ttc_for_mask(hdf_paths[sid], mask_n)

            # symmetrize + clip (your standard)
            Csym = symmetrize_ttc(C)
            Cplot = clip_ttc(Csym, p_hi=float(clip_hi_percentile))

            # per-panel normalization (after clipping)
            vmin = float(np.nanmin(Cplot))
            vmax = float(np.nanmax(Cplot))
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
                vmin, vmax = None, None

            ax.imshow(
                Cplot,
                origin="lower",
                cmap=cmap,
                interpolation="nearest",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            # TEMP label (remove later)
            ax.text(
                0.02, 0.98,
                f"{sid} | M{mask_n:03d}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=label_fontsize,
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="black",
                    alpha=0.6,
                    edgecolor="none",
                ),
            )

            meta.append((sid, mask_n))

    plt.tight_layout(pad=0.5)

    # ----------------------------
    # Save + show
    # ----------------------------
    if save:
        if out_path is None:
            out_path = Path.cwd() / "ttc_3x5_brightest_plus_0_1_2.png"
        else:
            out_path = Path(out_path)

        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()
    return meta


# ============================================================
# Execution functions
# ============================================================

def cosine_fitting_test():

    # If your TTC time step is known (seconds per TTC index step), set dt_s appropriately.
    DT_S = 1.0

    # "Start point" along t1=t2
    START_DIAG_IDX = 2500  # e.g. "t=t=600s" if dt_s=1s (otherwise it's an index)

    # Example: constrain possible periods
    # period ~100 s => omega ~ 2π/100 ~ 0.0628 rad/s
    fit = extract_fit_antidiagonal_with_ttc_plot(
        SAMPLE_ID,
        BASE_DIR,
        mask_n=MASK_N,
        start_time_idx=START_DIAG_IDX,
        dt_s=DT_S,
        omega_min=2 * np.pi / 500.0,  # periods up to 500 s
        omega_max=2 * np.pi / 20.0,  # periods down to 20 s
        n_omega=260,
        n_tau=120,
        despike_at_start=True,
        despike_halfwidth=1,
        use_weights=True,
    )

def plot_of_lineout_directions():

    START_DIAG_IDX = 2500

    # If you also want the multi-lineout plot:
    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)
    plot_ttc_with_lineouts(data, start=START_DIAG_IDX, add_antidiag_se=True, despike_at_start=True, despike_halfwidth=1)


def plot_of_period_vs_diagonal_start():
    DT_S = 1.0

    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)

    out = plot_period_vs_diagonal_start(
        data,
        dt_s=DT_S,
        drop_first_lineout=5,
        half_window=5,  # <- choose smoothing strength
        band_ci=0.68,  # <- 0.68 or 0.95 etc
        fmin=1 / 1000,
        fmax=1 / 10,
        detrend=True,
        window="hann",
    )

def plot_of_single_fft_antidiagonal_lineout():

    START_DIAG_IDX = 2500

    DT_S = 1.0

    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)

    out = plot_ttc_lineout_fft(
        data,
        start_idx=START_DIAG_IDX,
        dt_s=DT_S,
        drop_first=5,  # try 0, 5, 10
        detrend=True,
        window=True,
    )

def plot_of_period_vs_diagonal_start_both_lineouts():
    DT_S = 1.0

    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)

    out = plot_period_vs_diagonal_start_both_lineouts(
        data,
        dt_s=DT_S,
        drop_first_antidiag=5,
        drop_first_horizontal=0,
        half_window=5,  # set 0 for no smoothing
        fmin=1 / 1000,
        fmax=1 / 10,
        detrend=True,
        window="hann",
    )

def fft_2d_plot():

    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)

    plot_ttc_and_2dfft(
        data.ttc,
        dt_s=1.0,
        clip_hi_percentile=99.9,
    )

def fft_2d_fitting_and_parameter_extraction():

    # # Demo with random data:
    # demo_with_random()

    # Replace this with your TTC array:
    data = load_xpcs_arrays(SAMPLE_ID, BASE_DIR, mask_n=MASK_N)
    ttc = data.ttc

    # Then:
    fit = fit_ttc_four_params(ttc, dt_s=1.0, downsample=5)
    plot_measured_vs_model_ttc(ttc, fit, dt_s=1.0, clip_hi_percentile=99.5)

def read_sample_temperature(f_meta: h5py.File) -> float | None:
    """
    Try common temperature fields and return value in K if found.
    """
    candidates = [
        "entry/sample/qnw1_temperature",
        "entry/sample/qnw_lakeshore",
        "entry/sample/qnw2_temperature",
    ]

    for path in candidates:
        if path in f_meta:
            val = f_meta[path][()]
            try:
                return float(val)
            except Exception:
                pass

    return None


def plot_A4_17scan_central_brightest_ttcs_entrypoint():
    """
    Execution wrapper you can call from if __name__ == "__main__":.
    Uses your stated list.
    """
    scans = [
        "A010","A017","A023","A029","A036","A042","A048","A053","A063",
        "A073","A078","A083","A088","A093","A098","A101","A104",
    ]
    return plot_A4_17scan_central_brightest_ttcs(
        base_dir=BASE_DIR,
        sample_ids=scans,
        clip_hi_percentile=99.9,
        cmap="plasma",
        figsize_per_panel=(2.0, 2.25),
        title_fontsize=9,
        textbox_fontsize=8,
    )

def exec_plot_3x5_brightest_plus_offsets_ttcs():

    return plot_3x5_brightest_plus_offsets_ttcs(
        base_dir=BASE_DIR,
        sample_ids=["A029", "A048", "A073", "A088", "A101"],
        clip_hi_percentile=99.9,
        cmap="plasma",
        figsize=(12.0, 7.2),
        save=True,
        out_path=Path("A4_3x5_brightest_plus012.png"),
    )


BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
SAMPLE_ID = "A073"
MASK_N = 144

if __name__ == "__main__":

    # cosine_fitting_test()
    # plot_of_lineout_directions()
    # plot_of_period_vs_diagonal_start()
    # plot_of_single_fft_antidiagonal_lineout()
    # plot_of_period_vs_diagonal_start_both_lineouts()
    fft_2d_plot()
    # fft_2d_fitting_and_parameter_extraction()
    # plot_A4_17scan_central_brightest_ttcs_entrypoint()
    # exec_plot_3x5_brightest_plus_offsets_ttcs()


    pass