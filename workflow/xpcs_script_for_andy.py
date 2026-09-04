#!/usr/bin/env python3
"""
Standalone Nov 2024 APSU XPCS analysis for Andy.

Ship with this file only:
  - xpcs_script_for_andy.py
  - twotime.py          (same directory; import checked at startup)

Python packages: numpy, h5py, hdf5plugin, matplotlib, torch, tqdm

Edit RUN_DIR below to point at the run folder on your machine.
Uncomment one step in ``if __name__ == "__main__"`` to run it.

g2/TTC math matches twotime.py::TwotimeCorrelator (numpy implementation here;
twotime is imported so you can extend with TwotimeCorrelator if needed).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression filters
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

# twotime.py must sit next to this script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from twotime import TwotimeCorrelator  # noqa: F401

# ---------------------------------------------------------------------------
# User settings — edit paths / ROI for your machine
# ---------------------------------------------------------------------------
RUN_DIR = Path(
    "/Volumes/EmilioSD4TB/APSU_XPCS_Nov2024/A344_NaBH_att000020_329K_001"
)

# Square ROI on raw frames [y0:y1, x0:x1) in pixel indices
ROI_Y0, ROI_Y1 = 780, 840
ROI_X0, ROI_X1 = 1195, 1255


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------
def open_h5(path: Path) -> h5py.File:
    return h5py.File(Path(path), "r")


def print_h5_tree(
    f: h5py.File,
    *,
    max_depth: int = 6,
    max_children_per_group: int = 200,
) -> None:
    def _recurse(g: h5py.Group, prefix: str, depth: int) -> None:
        if depth > max_depth:
            print(prefix + "… (max_depth reached)")
            return
        try:
            items = list(g.items())
        except Exception as e:
            print(prefix + f"(cannot list items: {e})")
            return
        truncated = len(items) > max_children_per_group
        if truncated:
            items = items[:max_children_per_group]
        for _name, obj in items:
            path = obj.name
            if isinstance(obj, h5py.Dataset):
                chunks = obj.chunks
                comp = obj.compression
                extra = []
                if chunks is not None:
                    extra.append(f"chunks={chunks}")
                if comp is not None:
                    extra.append(f"compression={comp}")
                extra_s = ("  " + ", ".join(extra)) if extra else ""
                print(
                    prefix
                    + f"- {path}  [Dataset] shape={obj.shape} dtype={obj.dtype}{extra_s}"
                )
            elif isinstance(obj, h5py.Group):
                print(prefix + f"+ {path}  [Group]")
                _recurse(obj, prefix + "  ", depth + 1)
        if truncated:
            print(prefix + f"… ({max_children_per_group} children shown, truncated)")

    print(f"\nFILE: {getattr(f, 'filename', '<unknown>')}")
    print("+ /  [Group]")
    _recurse(f["/"], prefix="  ", depth=1)


def find_raw_h5(run_dir: Path) -> Path:
    h5_files = sorted(p for p in run_dir.glob("*.h5") if not p.name.startswith("._"))
    if len(h5_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .h5 in {run_dir}, found {len(h5_files)}"
        )
    return h5_files[0]


def preset_period_s(run_dir: Path) -> float:
    for p in sorted(run_dir.glob("*.batchinfo")):
        if p.name.startswith("._"):
            continue
        m = re.search(r"preset_period\s*=\s*([\d.]+)", p.read_text())
        if m:
            return float(m.group(1))
    return 1.0


# ---------------------------------------------------------------------------
# twotime-equivalent correlation (numpy; matches workflow / twotime.py math)
# ---------------------------------------------------------------------------
def symmetrize_upper_triangle(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c)
    return c + c.T - np.diag(np.diag(c))


def smooth_like_twotime(time_series: np.ndarray) -> np.ndarray:
    ts = np.asarray(time_series, dtype=np.float64)
    avg_pix = ts.mean(axis=0)
    avg_pix[avg_pix <= 0] = 1.0
    return ts / avg_pix[None, :]


def compute_ttc_like_twotime(time_series: np.ndarray) -> np.ndarray:
    ts = np.asarray(time_series, dtype=np.float64)
    norm_factor = ts.sum(axis=1)
    norm_factor[norm_factor <= 0] = 1.0
    norm_factor = 1.0 / norm_factor
    matmul_prod = ts @ ts.T
    npix = ts.shape[1]
    c2 = matmul_prod * norm_factor[:, None] * norm_factor[None, :] * float(npix)
    return np.triu(c2)


def g2_from_ttc_diagonal(c2_ut: np.ndarray) -> np.ndarray:
    c2 = np.asarray(c2_ut, dtype=np.float64)
    c2_sym = c2 + c2.T - np.diag(np.diag(c2))
    n = int(c2_sym.shape[0])
    lag_idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).ravel()
    weights = c2_sym.ravel()
    sums = np.bincount(lag_idx, weights=weights, minlength=n)
    counts = np.bincount(lag_idx, minlength=n)
    return sums / np.maximum(counts, 1)


def load_roi_time_series(
    dset: h5py.Dataset,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    frame_slice: slice | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    n_frames = int(dset.shape[0])
    sl = frame_slice if frame_slice is not None else slice(0, n_frames)
    frame_idxs = list(range(*sl.indices(n_frames)))
    n_use = len(frame_idxs)
    if n_use <= 0:
        raise ValueError("frame_slice selects no frames")

    fr0 = np.asarray(dset[int(frame_idxs[0])])
    if fr0.ndim == 3 and fr0.shape[0] == 1:
        fr0 = fr0[0]
    ny, nx = int(fr0.shape[0]), int(fr0.shape[1])
    y0c, y1c = int(np.clip(y0, 0, ny)), int(np.clip(y1, 0, ny))
    x0c, x1c = int(np.clip(x0, 0, nx)), int(np.clip(x1, 0, nx))
    if y1c <= y0c or x1c <= x0c:
        raise ValueError(
            f"Invalid ROI y=[{y0},{y1}) x=[{x0},{x1}) for detector shape ({ny}, {nx})"
        )

    npix = (y1c - y0c) * (x1c - x0c)
    ts = np.empty((n_use, npix), dtype=np.float64)
    for j, i in enumerate(frame_idxs):
        fr = np.asarray(dset[int(i)])
        if fr.ndim == 3 and fr.shape[0] == 1:
            fr = fr[0]
        ts[j, :] = fr[y0c:y1c, x0c:x1c].astype(np.float64, copy=False).ravel()
    return ts, (y0c, y1c, x0c, x1c)


def read_frame2d(dset: h5py.Dataset, frame_idx: int) -> np.ndarray:
    fr = np.asarray(dset[int(frame_idx)])
    if fr.ndim == 3 and fr.shape[0] == 1:
        fr = fr[0]
    if fr.ndim != 2:
        raise ValueError(f"Unexpected frame shape {fr.shape} at index {frame_idx}")
    return fr.astype(np.float64, copy=False)


def to_log_display(
    frame2d: np.ndarray,
    *,
    log_eps: float = 1.0,
    clip_percentile: float = 99.9,
) -> np.ndarray:
    f = np.clip(frame2d, 0.0, None)
    pos = f[f > 0]
    if pos.size:
        clip_hi = float(np.percentile(pos, float(clip_percentile)))
        f = np.minimum(f, clip_hi)
    return np.log10(f + float(log_eps))


# ---------------------------------------------------------------------------
# Nov 2024 analysis steps
# ---------------------------------------------------------------------------
def run_data_inspector(
    run_dir: Path | str | None = None,
    *,
    max_depth: int = 7,
    max_children_per_group: int = 300,
) -> None:
    """Print batchinfo text and HDF5 trees for .h5 and .hdf in the run folder."""
    run_dir = Path(run_dir) if run_dir is not None else RUN_DIR
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    batchinfo_files = sorted(
        p for p in run_dir.glob("*.batchinfo") if not p.name.startswith("._")
    )
    hdf_files = sorted(p for p in run_dir.glob("*.hdf") if not p.name.startswith("._"))
    h5_path = find_raw_h5(run_dir)

    print(f"\nNov 2024 run directory: {run_dir}")
    print(f"  batchinfo ({len(batchinfo_files)}): {[p.name for p in batchinfo_files]}")
    print(f"  .h5: {h5_path.name}")
    print(f"  .hdf ({len(hdf_files)}): {[p.name for p in hdf_files]}")

    if len(batchinfo_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .batchinfo in {run_dir}, found {len(batchinfo_files)}"
        )
    if len(hdf_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .hdf in {run_dir}, found {len(hdf_files)}"
        )

    batchinfo_path, hdf_path = batchinfo_files[0], hdf_files[0]

    print(f"\n--- batchinfo: {batchinfo_path.name} ---")
    print(batchinfo_path.read_text())

    print(f"\n--- raw images (.h5): {h5_path.name} ---")
    with open_h5(h5_path) as f_raw:
        print_h5_tree(
            f_raw, max_depth=max_depth, max_children_per_group=max_children_per_group
        )
        if "entry/data/data" in f_raw:
            dset = f_raw["entry/data/data"]
            print(
                f"  entry/data/data: shape={dset.shape} dtype={dset.dtype} "
                f"chunks={dset.chunks}"
            )

    print(f"\n--- metadata (.hdf): {hdf_path.name} ---")
    with open_h5(hdf_path) as f_meta:
        print_h5_tree(
            f_meta, max_depth=max_depth, max_children_per_group=max_children_per_group
        )


def raw_frame_viewer(
    run_dir: Path | str | None = None,
    *,
    start_frame: int = 0,
    cmap: str = "magma",
    log_eps: float = 1.0,
    clip_percentile: float = 99.9,
) -> None:
    """Log-scale frame viewer. Keys: ←/→ (or a/d), Home/End, slider."""
    run_dir = Path(run_dir) if run_dir is not None else RUN_DIR
    h5_path = find_raw_h5(run_dir)

    with open_h5(h5_path) as f_raw:
        if "entry/data/data" not in f_raw:
            raise KeyError(f"Missing entry/data/data in {h5_path}")
        dset = f_raw["entry/data/data"]
        n_frames = int(dset.shape[0])
        i0 = int(np.clip(int(start_frame), 0, n_frames - 1))

        disp0 = to_log_display(
            read_frame2d(dset, i0), log_eps=log_eps, clip_percentile=clip_percentile
        )
        vmin = float(np.nanpercentile(disp0, 1.0))
        vmax = float(np.nanpercentile(disp0, 99.9))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0

        print(f"Raw viewer: {h5_path}  frames={n_frames}  shape={dset.shape[1:]}")
        print("  keys: ←/→ (or a/d), Home/End, slider")

        fig, ax = plt.subplots(figsize=(8, 7))
        fig.subplots_adjust(bottom=0.12)
        state = {"idx": i0}
        im = ax.imshow(
            disp0,
            origin="upper",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        fig.colorbar(im, ax=ax, label="log10(ADU + eps)", shrink=0.85)
        title = ax.set_title(f"{h5_path.name}  |  frame {i0}/{n_frames - 1}", fontsize=10)

        ax_sl = fig.add_axes([0.20, 0.02, 0.60, 0.03])
        slider = Slider(ax_sl, "Frame", 0, n_frames - 1, valinit=i0, valstep=1, valfmt="%d")

        def _update(idx: float) -> None:
            state["idx"] = int(idx)
            im.set_data(
                to_log_display(
                    read_frame2d(dset, state["idx"]),
                    log_eps=log_eps,
                    clip_percentile=clip_percentile,
                )
            )
            title.set_text(
                f"{h5_path.name}  |  frame {state['idx']}/{n_frames - 1}"
            )
            fig.canvas.draw_idle()

        slider.on_changed(_update)

        def _on_key(event) -> None:
            if event.key in ("right", "d"):
                state["idx"] = min(n_frames - 1, state["idx"] + 1)
            elif event.key in ("left", "a"):
                state["idx"] = max(0, state["idx"] - 1)
            elif event.key == "home":
                state["idx"] = 0
            elif event.key == "end":
                state["idx"] = n_frames - 1
            else:
                return
            slider.set_val(state["idx"])

        fig.canvas.mpl_connect("key_press_event", _on_key)
        plt.show()


def roi_g2_analysis(
    run_dir: Path | str | None = None,
    *,
    y0: int = ROI_Y0,
    y1: int = ROI_Y1,
    x0: int = ROI_X0,
    x1: int = ROI_X1,
    start_frame: int = 0,
    frame_slice: slice | None = None,
    cmap: str = "magma",
    log_eps: float = 1.0,
    clip_percentile: float = 99.9,
    ttc_clip_percentile: float = 99.9,
    cmap_ttc: str = "plasma",
    dt_s: float | None = None,
) -> dict[str, Any]:
    """
    ROI g2 + TTC (twotime math) and interactive frame viewer with white ROI box.

    Figure 1: frames (←/→).  Figure 2: g2(tau).  Figure 3: TTC heatmap.
    """
    run_dir = Path(run_dir) if run_dir is not None else RUN_DIR
    if dt_s is None:
        dt_s = preset_period_s(run_dir)

    h5_path = find_raw_h5(run_dir)
    with open_h5(h5_path) as f_raw:
        if "entry/data/data" not in f_raw:
            raise KeyError(f"Missing entry/data/data in {h5_path}")
        dset = f_raw["entry/data/data"]
        n_frames = int(dset.shape[0])

        ts, (y0c, y1c, x0c, x1c) = load_roi_time_series(
            dset, y0=y0, y1=y1, x0=x0, x1=x1, frame_slice=frame_slice
        )
        n_g2 = int(ts.shape[0])
        print(f"ROI g2: {h5_path.name}")
        print(
            f"  ROI y=[{y0c},{y1c}) x=[{x0c},{x1c})  "
            f"({y1c - y0c}x{x1c - x0c} px, {ts.shape[1]} pixels)"
        )
        print(f"  frames={n_g2}  dt={dt_s} s")

        i_smooth = smooth_like_twotime(ts)
        c2_ut = compute_ttc_like_twotime(i_smooth)
        g2 = g2_from_ttc_diagonal(c2_ut)
        tau_idx = np.arange(g2.size, dtype=np.float64)
        tau_s = tau_idx * float(dt_s)
        c2_sym = symmetrize_upper_triangle(c2_ut)

        i0 = int(np.clip(int(start_frame), 0, n_frames - 1))
        disp0 = to_log_display(
            read_frame2d(dset, i0), log_eps=log_eps, clip_percentile=clip_percentile
        )
        vmin = float(np.nanpercentile(disp0, 1.0))
        vmax = float(np.nanpercentile(disp0, 99.9))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0

        # Figure 2: g2
        fig_g2, ax_g2 = plt.subplots(figsize=(6.5, 4.5))
        mask = (tau_idx > 0) & np.isfinite(g2)
        if np.any(mask):
            ax_g2.semilogx(tau_s[mask], g2[mask], "b.", ms=4)
        ax_g2.set_xlabel("Delay time tau (s)")
        ax_g2.set_ylabel("g2(tau)")
        ax_g2.set_title(
            f"g2 (twotime diagonal avg)  |  ROI [{y0c}:{y1c}, {x0c}:{x1c}]  |  N={n_g2}"
        )
        ax_g2.grid(True, alpha=0.3)
        fig_g2.tight_layout()

        # Figure 3: TTC
        lo = float(np.nanpercentile(c2_sym, 0.0))
        hi = float(np.nanpercentile(c2_sym, float(ttc_clip_percentile)))
        c2_plot = np.clip(c2_sym, lo, hi)
        fig_ttc, ax_ttc = plt.subplots(figsize=(7.5, 6.5))
        im_ttc = ax_ttc.imshow(
            c2_plot,
            origin="lower",
            cmap=cmap_ttc,
            aspect="equal",
            interpolation="nearest",
            extent=[tau_s[0], tau_s[-1], tau_s[0], tau_s[-1]],
        )
        ax_ttc.set_xlabel("t1 (s)")
        ax_ttc.set_ylabel("t2 (s)")
        ax_ttc.set_title(f"TTC (twotime)  |  ROI [{y0c}:{y1c}, {x0c}:{x1c}]  |  N={n_g2}")
        fig_ttc.colorbar(im_ttc, ax=ax_ttc, fraction=0.046, pad=0.04)
        fig_ttc.tight_layout()

        # Figure 1: frames + ROI
        fig_fr, ax_fr = plt.subplots(figsize=(8, 7))
        fig_fr.subplots_adjust(bottom=0.12)
        state = {"idx": i0}
        im = ax_fr.imshow(
            disp0,
            origin="upper",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax_fr.set_xlabel("x (pixel)")
        ax_fr.set_ylabel("y (pixel)")
        fig_fr.colorbar(im, ax=ax_fr, label="log10(ADU + eps)", shrink=0.85)
        ax_fr.add_patch(
            Rectangle(
                (x0c - 0.5, y0c - 0.5),
                x1c - x0c,
                y1c - y0c,
                linewidth=2.0,
                edgecolor="white",
                facecolor="none",
            )
        )

        def _frame_title() -> str:
            return (
                f"{h5_path.name}  |  frame {state['idx']}/{n_frames - 1}  |  "
                f"ROI x=[{x0c},{x1c}) y=[{y0c},{y1c})"
            )

        title_fr = ax_fr.set_title(_frame_title(), fontsize=10)
        print("  frame viewer: ←/→ (or a/d), Home/End, slider")

        ax_sl = fig_fr.add_axes([0.20, 0.02, 0.60, 0.03])
        slider = Slider(
            ax_sl, "Frame", 0, n_frames - 1, valinit=i0, valstep=1, valfmt="%d"
        )

        def _update_frame(idx: float) -> None:
            state["idx"] = int(idx)
            im.set_data(
                to_log_display(
                    read_frame2d(dset, state["idx"]),
                    log_eps=log_eps,
                    clip_percentile=clip_percentile,
                )
            )
            title_fr.set_text(_frame_title())
            fig_fr.canvas.draw_idle()

        slider.on_changed(_update_frame)

        def _on_key(event) -> None:
            if event.key in ("right", "d"):
                state["idx"] = min(n_frames - 1, state["idx"] + 1)
            elif event.key in ("left", "a"):
                state["idx"] = max(0, state["idx"] - 1)
            elif event.key == "home":
                state["idx"] = 0
            elif event.key == "end":
                state["idx"] = n_frames - 1
            else:
                return
            slider.set_val(state["idx"])

        fig_fr.canvas.mpl_connect("key_press_event", _on_key)
        plt.show()

    return {
        "g2": g2,
        "tau_idx": tau_idx,
        "tau_s": tau_s,
        "dt_s": float(dt_s),
        "roi": (y0c, y1c, x0c, x1c),
        "c2_ut": c2_ut,
        "c2_sym": c2_sym,
    }


if __name__ == "__main__":

    # ---- november 2024 double peak analysis ----
    # run_data_inspector()
    # raw_frame_viewer()
    roi_g2_analysis()
