# Merged library: raw_data_inspection + analysis_for_aps_08-ide-2025-1006 + google_sheet_upload_script
# Use workflow_run.py to run. Config in workflow_config.py.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Iterable
from io import BytesIO
import sys
import re
import time
import random
import json

import h5py
import hdf5plugin
import numpy as np

import matplotlib as mpl
mpl.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter

import cv2
from scipy.special import erfinv
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import curve_fit

import gspread
import httplib2
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError

from workflow_config import (
    BASE_DIR, SAMPLE_ID, MASK_N, CONTROL_MASK_N,
    FILE_ID, H5_FILE, H5_BASE_DIR, POSITION_NAME, OUT_DIR, FIGURES_DIR,
    SPREADSHEET_ID, TAB_NAME, TOKEN_PATH, CREDS_PATH, UPLOAD_FOLDER_ID,
    FIGTYPE_DIR, PLOT_COLS, ALL_PLOT_KEYS, DPI_BY_PLOT, SCOPES,
    GENERATE_KEYS, UPLOAD_TO_SHEETS, UPLOAD_KEYS,
    BASE_DIR_OVERRIDES, RESULTS_BASE_DIR,
)
h5_file = H5_FILE  # for analysis section


def _resolve_base_dir(scan_id: str, fallback: Path = BASE_DIR) -> Path:
    """Return the correct base directory for a given scan ID, checking overrides."""
    return BASE_DIR_OVERRIDES.get(scan_id, fallback)

# ---- raw_data_inspection ----
@dataclass
class RunData:
    # paths
    raw_path: Path
    meta_path: Path
    results_path: Path

    # open handles (raw + metadata stay open so you can read frames lazily)
    f_raw: h5py.File
    dset_raw: h5py.Dataset
    f_meta: h5py.File

    # processed arrays (loaded into memory)
    dynamic_roi_map: np.ndarray
    scattering_2d: np.ndarray
    ttc: np.ndarray
    g2: np.ndarray

    def close(self) -> None:
        """Close open HDF5 handles."""
        try:
            if self.f_raw:
                self.f_raw.close()
        finally:
            self.f_raw = None  # type: ignore
        try:
            if self.f_meta:
                self.f_meta.close()
        finally:
            self.f_meta = None  # type: ignore


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_processed_results(base_dir: Path, sample_id: str) -> Path:
    """
    Example target:
      <BASE_DIR>/Twotime_PostExpt_01/A073_*_results.hdf
    """
    proc_dir = base_dir / "Twotime_PostExpt_01"
    matches = sorted(proc_dir.glob(f"{sample_id}_*_results.hdf"))
    if not matches:
        raise FileNotFoundError(f"No processed results found in {proc_dir} matching {sample_id}_*_results.hdf")
    return matches[0]


def find_raw_run_dir(base_dir: Path, sample_id: str) -> Path:
    """
    Finds the raw run folder:
      <BASE_DIR>/data/<RUN_NAME>/
    where RUN_NAME starts with SAMPLE_ID, e.g. A073_IPA_NBH_...
    """
    data_dir = base_dir / "data"
    matches = sorted([p for p in data_dir.glob(f"{sample_id}_*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No raw run directory found in {data_dir} starting with {sample_id}")
    return matches[0]


def load_raw_data_only(
    base_dir: Path,
    sample_id: str,
) -> RunData:
    """
    Load only raw data (no processed results required).
    Creates a minimal RunData with dummy arrays for processed data.
    Useful when you only need raw frames (e.g., mask_n="peak" mode).
    """
    import hdf5plugin  # noqa: F401

    run_dir = find_raw_run_dir(base_dir, sample_id)
    raw_path, meta_path = find_raw_data_files(run_dir)

    f_raw = h5py.File(raw_path, "r")
    if "entry/data/data" not in f_raw:
        raise KeyError(f"Raw file missing dataset 'entry/data/data': {raw_path}")
    dset_raw = f_raw["entry/data/data"]

    f_meta = h5py.File(meta_path, "r")

    # Get frame shape to create dummy arrays
    frame0 = np.asarray(dset_raw[0])
    if frame0.ndim == 3 and frame0.shape[0] == 1:
        frame0 = frame0[0]
    ny, nx = frame0.shape[:2]

    # Create dummy arrays (not used when mask_n="peak")
    dynamic_roi_map = np.zeros((ny, nx), dtype=np.uint16)
    scattering_2d = np.zeros((ny, nx), dtype=np.float32)
    ttc = np.zeros((10, 10), dtype=np.float32)
    g2 = np.zeros((10,), dtype=np.float32)

    return RunData(
        raw_path=raw_path,
        meta_path=meta_path,
        results_path=Path(""),  # dummy
        f_raw=f_raw,
        dset_raw=dset_raw,
        f_meta=f_meta,
        dynamic_roi_map=dynamic_roi_map,
        scattering_2d=scattering_2d,
        ttc=ttc,
        g2=g2,
    )


def find_raw_data_files(run_dir: Path) -> tuple[Path, Path]:
    """
    Inside run_dir, find:
      - raw images file: <RUN_NAME>.h5  (or .hdf/.hdf5)
      - metadata file:   <RUN_NAME>_metadata.hdf (or .h5/.hdf5)

    Returns (raw_path, meta_path).
    """
    run_name = run_dir.name

    raw_candidates = [
        run_dir / f"{run_name}.h5",
        run_dir / f"{run_name}.hdf",
        run_dir / f"{run_name}.hdf5",
        run_dir / f"{run_name}.h5py",
    ]
    raw_path = _first_existing(raw_candidates)

    meta_candidates = [
        run_dir / f"{run_name}_metadata.hdf",
        run_dir / f"{run_name}_metadata.h5",
        run_dir / f"{run_name}_metadata.hdf5",
    ]
    meta_path = _first_existing(meta_candidates)

    if raw_path is None:
        raise FileNotFoundError(f"Could not find raw data file for run {run_name} in {run_dir}")
    if meta_path is None:
        raise FileNotFoundError(f"Could not find metadata file for run {run_name} in {run_dir}")

    return raw_path, meta_path


def load_run_data(
    base_dir: Path,
    sample_id: str,
    *,
    mask_n: int,
    scattering_first_frame_only: bool = True,
) -> RunData:
    """
    Loads:
      - raw image dataset handle: entry/data/data  (lazy; not read into RAM)
      - raw metadata file handle (lazy)
      - processed arrays from results.hdf into memory:
          dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
          scattering_2d   = f["xpcs/temporal_mean/scattering_2d"][...]
          ttc             = f["xpcs/twotime/correlation_map/c2_00{mask_n:03d}"][...]
          g2              = f["xpcs/twotime/normalized_g2"][...]
    """
    # Needed for compressed detector data (registers HDF5 filters)
    import hdf5plugin  # noqa: F401

    # --- locate files ---
    results_path = find_processed_results(base_dir, sample_id)

    run_dir = find_raw_run_dir(base_dir, sample_id)
    raw_path, meta_path = find_raw_data_files(run_dir)

    # --- open raw + metadata (keep open) ---
    f_raw = h5py.File(raw_path, "r")
    if "entry/data/data" not in f_raw:
        raise KeyError(f"Raw file missing dataset 'entry/data/data': {raw_path}")
    dset_raw = f_raw["entry/data/data"]

    f_meta = h5py.File(meta_path, "r")

    # --- load processed arrays (into memory) ---
    ttc_path = f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}"
    with h5py.File(results_path, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        if scattering_first_frame_only and scattering_2d.ndim == 3:
            scattering_2d = scattering_2d[0, :, :]
        ttc = f[ttc_path][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]

    return RunData(
        raw_path=raw_path,
        meta_path=meta_path,
        results_path=results_path,
        f_raw=f_raw,
        dset_raw=dset_raw,
        f_meta=f_meta,
        dynamic_roi_map=dynamic_roi_map,
        scattering_2d=scattering_2d,
        ttc=ttc,
        g2=g2,
    )

def _make_roi_boolean_mask(dynamic_roi_map, mask_n: int):
    """
    Tries to build a boolean mask from dynamic_roi_map for the given mask_n.

    Assumptions (common in XPCS pipelines):
      - dynamic_roi_map is an integer label image
      - pixels belonging to ROI k have value k (or sometimes k-1)

    This function tries:
      1) map == mask_n
      2) map == (mask_n - 1)
    """
    m = np.asarray(dynamic_roi_map)

    if m.ndim != 2:
        raise ValueError(f"dynamic_roi_map should be 2D, got {m.shape}")

    mask = (m == int(mask_n))
    if np.any(mask):
        return mask, int(mask_n)

    mask2 = (m == int(mask_n) - 1)
    if np.any(mask2):
        return mask2, int(mask_n) - 1

    # helpful debug info
    vals = np.unique(m[:: max(1, m.shape[0] // 64), :: max(1, m.shape[1] // 64)])
    raise ValueError(
        f"No pixels matched mask_n={mask_n} (or mask_n-1). "
        f"Sample of unique dynamic_roi_map values: {vals[:30]}{'...' if vals.size > 30 else ''}"
    )


def _apply_mask_and_clip(frame2d, roi_mask, clip_percentile: float):
    """
    Returns (masked_frame_float, vmin, vmax).

    - Outside ROI => NaN (so it won't drive percentiles)
    - Clip inside ROI at [0, clip_percentile] percentiles to suppress hot pixels
    """
    img = np.asarray(frame2d).astype(np.float32, copy=False)

    out = img.copy()
    out[~roi_mask] = np.nan

    # percentile computed on ROI pixels only
    roi_vals = out[roi_mask]
    roi_vals = roi_vals[np.isfinite(roi_vals)]

    if roi_vals.size == 0:
        return out, 0.0, 1.0

    p = float(clip_percentile)
    p = np.clip(p, 0.0, 100.0)

    lo = np.percentile(roi_vals, 0.0)
    hi = np.percentile(roi_vals, p)

    if not np.isfinite(lo):
        lo = float(np.nanmin(out))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    # clip only ROI pixels; keep outside as NaN
    out_roi = np.clip(out[roi_mask], lo, hi)
    out[roi_mask] = out_roi

    return out, float(lo), float(hi)


def launch_masked_raw_viewer(
    run: RunData,
    *,
    mask_n: int | str,
    start_frame: int = 0,
    clip_percentile_init: float = 99.9,
    cmap: str = "magma",
    use_log: bool = True,
    log_eps: float = 1e-3,
    fixed_log_vmax: float | None = 4.0,   # set None to auto (fast, from start frame)
    fixed_log_vmin: float | None = None,  # set None to auto (fast, from start frame)

    # --- MP4 export ---
    mp4_export: bool = False,
    export_path: str = "",  # set at runtime; default filled in by mask_roi_viewer_mp4_save()
    export_start_frame: int | None = None,
    export_n_frames: int | None = None,
    frame_skip: int = 1,
    fps: int = 30,

    # --- crop ---
    crop_size: int = 300,

    # If fixed_log_vmax is None and you REALLY want global scaling, set True.
    # This is slow because it scans many frames.
    scan_global_vmax: bool = False,
    scan_block: int = 64,
):
    """Interactive masked viewer + optional MP4 export.

    Viewer controls:
      - Left/right arrow keys: previous/next frame
      - Slider: clip percentile (suppresses hot pixels within ROI)

    Data requirements:
      - run.dset_raw        : HDF5 dataset (frames, ny, nx) or (frames, 1, ny, nx)
      - run.dynamic_roi_map : 2D label image

    mask_n:
      - int  : use ROI label mask_n
      - "peak": crop a square (crop_size x crop_size) around the brightest REGION in start_frame

    MP4 export:
      - If mp4_export=True, writes an mp4 using the SAME rendering as displayed.
      - Uses fixed color scaling (vmin/vmax constant across frames) so pulsing is visible.
      - frame_skip exports every Nth frame (1 = all frames).

    Notes
    -----
    - By default, fixed_log_vmax=4.0 keeps log scaling visually stable and fast.
    - If fixed_log_vmax=None and scan_global_vmax=True, we scan the export frame range
      to set vmax. This can be slow.
    """

    # ----------------------------
    # Small helpers
    # ----------------------------
    def _read_frame2d(frame_idx: int) -> np.ndarray:
        """Read one frame as 2D float64 without loading the whole stack."""
        fr = np.asarray(run.dset_raw[int(frame_idx)])
        if fr.ndim == 3 and fr.shape[0] == 1:
            fr = fr[0]
        if fr.ndim != 2:
            raise ValueError(f"Unexpected raw frame shape {fr.shape} at index {frame_idx}")
        return fr.astype(np.float64, copy=False)

    def _find_bright_region_center_in_frame(frame2d: np.ndarray) -> tuple[int, int]:
        """Return (cy,cx) for a bright region (robust vs single hot pixel)."""
        f = np.clip(frame2d, 0.0, None)
        pos = f[f > 0]
        if pos.size:
            clip_hi = float(np.percentile(pos, 99.9))
        else:
            clip_hi = float(np.nanmax(f)) if np.isfinite(np.nanmax(f)) else 0.0
        f = np.minimum(f, clip_hi)

        try:
            from scipy.ndimage import uniform_filter
            score = uniform_filter(f, size=21, mode="nearest")
            flat = int(np.nanargmax(score))
            cy, cx = np.unravel_index(flat, score.shape)
        except Exception:
            flat = int(np.nanargmax(f))
            cy, cx = np.unravel_index(flat, f.shape)

        return int(cy), int(cx)

    def _disp_from_masked(masked: np.ndarray) -> np.ndarray:
        """Convert masked (NaN outside ROI) to display array."""
        if use_log:
            return np.log10(np.clip(masked, 0.0, None) + float(log_eps))
        return np.asarray(masked, dtype=np.float64)

    def _finite_percentile(a: np.ndarray, p: float, default: float) -> float:
        vv = np.isfinite(a)
        if not np.any(vv):
            return float(default)
        return float(np.nanpercentile(a[vv], float(p)))

    # ----------------------------
    # Validate / normalize inputs
    # ----------------------------
    n_frames = int(run.dset_raw.shape[0])
    if n_frames <= 0:
        raise ValueError("Empty dataset: no frames")

    i0 = int(np.clip(int(start_frame), 0, n_frames - 1))
    frame_skip = int(max(1, frame_skip))
    crop_size = int(max(5, crop_size))
    fps = int(max(1, fps))

    # ----------------------------
    # Select mode: ROI label vs peak
    # ----------------------------
    if isinstance(mask_n, str) and mask_n.strip().lower() == "peak":
        used_label: int | str = "peak"
        f0 = _read_frame2d(i0)
        cy, cx = _find_bright_region_center_in_frame(f0)
        crop_h = crop_w = crop_size
        roi_mask_full = None  # keep everything inside crop
    else:
        used_label = int(mask_n)
        roi_mask_full, used_label = _make_roi_boolean_mask(run.dynamic_roi_map, used_label)
        cy, cx = roi_center_from_label_map(run.dynamic_roi_map, int(used_label))
        crop_h, crop_w = 100, 50

    def _get_cropped_and_mask(frame2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crop, _bbox = crop_around_center(frame2d, cy, cx, crop_h=crop_h, crop_w=crop_w)
        if roi_mask_full is None:
            m = np.ones(crop.shape, dtype=bool)
        else:
            m_f, _bbox2 = crop_around_center(
                roi_mask_full.astype(float), cy, cx, crop_h=crop_h, crop_w=crop_w
            )
            m = (m_f > 0)
        return crop, m

    def _render(frame_idx: int, clip_p: float) -> np.ndarray:
        fr = _read_frame2d(frame_idx)
        crop, m = _get_cropped_and_mask(fr)
        masked, _lo, _hi = _apply_mask_and_clip(crop, m, float(clip_p))
        return _disp_from_masked(masked)

    # ----------------------------
    # Fixed scaling for display/export
    # ----------------------------
    disp0 = _render(i0, clip_percentile_init)

    if use_log:
        # vmin
        if fixed_log_vmin is None:
            vmin_fixed = _finite_percentile(disp0, 1.0, np.log10(float(log_eps)))
        else:
            vmin_fixed = float(fixed_log_vmin)

        # vmax
        if fixed_log_vmax is None:
            # fast default: from start frame
            vmax_fixed = _finite_percentile(disp0, 99.9, vmin_fixed + 1.0)
        else:
            vmax_fixed = float(fixed_log_vmax)

        # optional slow global scan ONLY if requested
        if fixed_log_vmax is None and scan_global_vmax:
            # scan the export range (or whole stack) in display space
            es = 0 if export_start_frame is None else int(np.clip(export_start_frame, 0, n_frames - 1))
            if export_n_frames is None:
                ee = n_frames
            else:
                ee = int(np.clip(es + int(export_n_frames), 0, n_frames))
            if ee <= es:
                ee = min(n_frames, es + 1)

            vmax_scan = -np.inf
            step = frame_skip
            block = int(max(1, scan_block))

            for b0 in range(es, ee, step * block):
                idxs = list(range(b0, min(ee, b0 + step * block), step))
                for ii in idxs:
                    dd = _render(int(ii), clip_percentile_init)
                    m = float(np.nanmax(dd[np.isfinite(dd)])) if np.any(np.isfinite(dd)) else -np.inf
                    if m > vmax_scan:
                        vmax_scan = m

            if np.isfinite(vmax_scan):
                vmax_fixed = float(vmax_scan)

    else:
        # linear
        vmin_fixed = _finite_percentile(disp0, 1.0, 0.0)
        vmax_fixed = _finite_percentile(disp0, 99.9, vmin_fixed + 1.0)

    # Safety
    if not np.isfinite(vmin_fixed):
        vmin_fixed = 0.0
    if not np.isfinite(vmax_fixed) or vmax_fixed <= vmin_fixed:
        vmax_fixed = vmin_fixed + 1.0

    # ----------------------------
    # Build viewer figure
    # ----------------------------
    fig = plt.figure(figsize=(5.0, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax_slider = fig.add_subplot(gs[1, 0])
    ax_slider.axis("off")

    state = {"i": i0, "clip_p": float(clip_percentile_init)}

    im = ax.imshow(
        disp0,
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin_fixed,
        vmax=vmax_fixed,
    )
    ax.set_facecolor("black")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    def _title(frame_idx: int) -> str:
        return (
            f"Raw frame {frame_idx}/{n_frames-1}  |  ROI={used_label}  |  clip p={state['clip_p']:.2f}\n"
            f"center (cx,cy)=({cx},{cy})  crop={crop_w}x{crop_h}"
        )

    ax.set_title(_title(i0))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log10(ADU + eps)" if use_log else "ADU (clipped)")

    # Slider
    bbox = ax_slider.get_position()
    x0s, y0s, ws, hs = bbox.x0, bbox.y0, bbox.width, bbox.height
    axp = fig.add_axes([x0s + 0.10 * ws, y0s + 0.35 * hs, 0.82 * ws, 0.40 * hs])
    s_clip = Slider(axp, "clip percentile", 90.0, 100.0, valinit=state["clip_p"])

    def _redraw() -> None:
        disp = _render(state["i"], state["clip_p"])
        im.set_data(disp)
        im.set_clim(vmin=vmin_fixed, vmax=vmax_fixed)
        ax.set_title(_title(state["i"]))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["i"] = min(n_frames - 1, state["i"] + 1)
            _redraw()
        elif event.key in ("left", "a"):
            state["i"] = max(0, state["i"] - 1)
            _redraw()
        elif event.key == "home":
            state["i"] = 0
            _redraw()
        elif event.key == "end":
            state["i"] = n_frames - 1
            _redraw()

    def on_clip(_val):
        state["clip_p"] = float(s_clip.val)
        _redraw()

    s_clip.on_changed(on_clip)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # ----------------------------
    # MP4 EXPORT (no global scanning unless explicitly requested)
    # ----------------------------
    export_info: dict = {}
    if mp4_export:
        export_path_p = Path(export_path)
        export_path_p.parent.mkdir(parents=True, exist_ok=True)

        es = 0 if export_start_frame is None else int(np.clip(export_start_frame, 0, n_frames - 1))
        if export_n_frames is None:
            ee = n_frames
        else:
            ee = int(np.clip(es + int(export_n_frames), 0, n_frames))
        if ee <= es:
            ee = min(n_frames, es + 1)

        frame_indices = list(range(es, ee, frame_skip))
        print(f"Exporting MP4: frames {es} -> {ee-1} (step={frame_skip})  total={len(frame_indices)}")
        print(f"  -> {export_path_p}")

        # Use Matplotlib's ffmpeg writer (requires ffmpeg on PATH)
        from matplotlib.animation import FFMpegWriter

        try:
            writer = FFMpegWriter(
                fps=fps,
                bitrate=50000,  # High bitrate for quality (50 Mbps)
                codec="libx264",
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-crf", "10",  # Lower CRF = higher quality (0-51, 10 is very high quality)
                    "-preset", "veryslow",  # Slowest preset for best compression efficiency
                ],
            )
            with writer.saving(fig, str(export_path_p), dpi=150):
                for k, fi in enumerate(frame_indices):
                    state["i"] = int(fi)
                    disp = _render(int(fi), state["clip_p"])
                    im.set_data(disp)
                    im.set_clim(vmin=vmin_fixed, vmax=vmax_fixed)
                    ax.set_title(_title(int(fi)))
                    writer.grab_frame()
                    if (k + 1) % 200 == 0 or (k + 1) == len(frame_indices):
                        print(f"  wrote {k+1}/{len(frame_indices)} frames")
        except FileNotFoundError as e:
            if "ffmpeg" in str(e).lower():
                raise FileNotFoundError(
                    "ffmpeg not found. MP4 export requires ffmpeg on your PATH. "
                    "Install it with: brew install ffmpeg  (macOS) or apt install ffmpeg  (Linux)."
                ) from e
            raise

        print(f"Saved MP4 -> {export_path_p}")

        export_info.update(
            {
                "mp4_export": True,
                "export_path": str(export_path_p),
                "export_start_frame": int(es),
                "export_end_frame": int(ee - 1),
                "frame_skip": int(frame_skip),
                "fps": int(fps),
            }
        )

    plt.show()

    # ----------------------------
    # Return summary
    # ----------------------------
    roi_px = int(np.sum(roi_mask_full)) if roi_mask_full is not None else int(crop_h * crop_w)
    out = {
        "roi_label_used": used_label,
        "roi_pixel_count": roi_px,
        "center_cxcy": (float(cx), float(cy)),
        "crop_wh": (int(crop_w), int(crop_h)),
        "vmin_fixed": float(vmin_fixed),
        "vmax_fixed": float(vmax_fixed),
    }
    out.update(export_info)
    return out


def roi_center_from_label_map(dynamic_roi_map, mask_n: int):
    """
    dynamic_roi_map: 2D int array, same shape as detector image (rows, cols)
    mask_n: ROI label (e.g. 145)

    Returns (cy, cx) in pixel indices (row, col).
    """
    ys, xs = np.where(dynamic_roi_map == int(mask_n))
    if ys.size == 0:
        raise ValueError(f"mask_n={mask_n} not found in dynamic_roi_map")
    cy = int(np.round(np.mean(ys)))
    cx = int(np.round(np.mean(xs)))
    return cy, cx


def crop_around_center(img2d, cy: int, cx: int, *, crop_h: int = 100, crop_w: int = 50):
    """
    Returns cropped view of img2d centered at (cy, cx), clipped to image bounds.
    crop_h, crop_w are total output sizes.
    """
    H, W = img2d.shape
    hh = crop_h // 2
    hw = crop_w // 2

    y0 = max(0, cy - hh)
    y1 = min(H, cy + hh + (crop_h % 2))
    x0 = max(0, cx - hw)
    x1 = min(W, cx + hw + (crop_w % 2))

    return img2d[y0:y1, x0:x1], (y0, y1, x0, x1)

def inspect_raw_mask_oscillations(
    run: RunData,
    *,
    mask_signal: int,
    mask_control: int,
    dt_s: float = 1.0,
    fmin: float = 1 / 1000,
    fmax: float = 1 / 10,
    detrend: bool = True,
    window: bool = True,
    figsize=(12.5, 7.0),
):
    """
    Compare raw-intensity oscillations between a signal ROI and a control ROI.

    Uses raw detector frames (run.dset_raw) and dynamic_roi_map.
    """

    dset = run.dset_raw
    n_frames = int(dset.shape[0])
    t = np.arange(n_frames) * float(dt_s)

    # --- build masks ---
    mask_sig, used_sig = _make_roi_boolean_mask(run.dynamic_roi_map, mask_signal)
    mask_ctl, used_ctl = _make_roi_boolean_mask(run.dynamic_roi_map, mask_control)

    # --- extract summed intensity traces ---
    y_sig = np.zeros(n_frames, dtype=np.float64)
    y_ctl = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        frame = dset[i, :, :]
        y_sig[i] = np.sum(frame[mask_sig])
        y_ctl[i] = np.sum(frame[mask_ctl])

    # --- preprocessing helper ---
    def preprocess_and_fft(y):
        y = y.astype(np.float64)
        y = y - np.mean(y)

        if detrend:
            A = np.column_stack([t, np.ones_like(t)])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            y = y - (A @ beta)

        if window:
            y = y * np.hanning(len(y))

        F = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), d=dt_s)
        power = np.abs(F) ** 2

        m = (freqs >= fmin) & (freqs <= fmax)
        return y, freqs[m], power[m]

    y_sig_p, f_sig, P_sig = preprocess_and_fft(y_sig)
    y_ctl_p, f_ctl, P_ctl = preprocess_and_fft(y_ctl)

    # --- plotting ---
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    axs[0, 0].plot(t, y_sig_p, lw=1.4, color="C3")
    axs[0, 0].set_title(f"Signal mask {used_sig} (raw intensity)")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Intensity (a.u.)")

    axs[0, 1].plot(t, y_ctl_p, lw=1.4, color="0.3")
    axs[0, 1].set_title(f"Control mask {used_ctl} (raw intensity)")
    axs[0, 1].set_xlabel("Time [s]")

    axs[1, 0].plot(f_sig, P_sig, lw=1.8, color="C3")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("Power")

    axs[1, 1].plot(f_ctl, P_ctl, lw=1.8, color="0.3")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("Frequency [Hz]")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "signal": {"t": t, "y": y_sig, "freqs": f_sig, "power": P_sig},
        "control": {"t": t, "y": y_ctl, "freqs": f_ctl, "power": P_ctl},
    }


def extract_roi_intensity_matrix(
    dset_raw,
    *,
    dynamic_roi_map=None,
    mask_n: int | None = None,
    roi_mask=None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    dtype=np.float32,
):
    """
    Build the per-pixel intensity matrix for one ROI from the raw frames.

    Returns
    -------
    I : (T, P) array
        I[t, p] is the intensity of ROI pixel p at time/frame t.
        T is number of selected frames, P is number of ROI pixels.
    frame_idxs : (T,) array
        The raw frame indices used (accounts for start/stop/stride).

    Notes
    -----
    - You may pass either:
        (A) roi_mask= (bool 2D array), OR
        (B) dynamic_roi_map= (int 2D array) AND mask_n= (ROI label)
      If roi_mask is provided, it is used directly.
    - Reads frames lazily from the HDF5 dataset (no full load).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    n_frames = int(dset_raw.shape[0])
    if stop is None:
        stop = n_frames
    start = int(np.clip(start, 0, n_frames))
    stop = int(np.clip(stop, 0, n_frames))
    if stop <= start:
        raise ValueError(f"Invalid frame range: start={start}, stop={stop}")

    # --- resolve ROI mask ---
    if roi_mask is not None:
        m = np.asarray(roi_mask).astype(bool, copy=False)
        if m.ndim != 2:
            raise ValueError(f"roi_mask must be 2D, got shape {m.shape}")
    else:
        if dynamic_roi_map is None or mask_n is None:
            raise ValueError("Provide either roi_mask=..., or (dynamic_roi_map=... and mask_n=...)")
        m, _ = _make_roi_boolean_mask(dynamic_roi_map, int(mask_n))  # uses your existing helper

    P = int(np.sum(m))
    if P <= 0:
        raise ValueError("ROI mask has zero pixels")

    frame_idxs = np.arange(start, stop, stride, dtype=int)
    T = int(frame_idxs.size)

    # Preallocate output: (T, P)
    I = np.empty((T, P), dtype=dtype)

    # Read each frame lazily and vectorize ROI pixels
    for j, fi in enumerate(frame_idxs):
        frame = dset_raw[int(fi), :, :]
        I[j, :] = np.asarray(frame)[m].astype(dtype, copy=False)

    return I, frame_idxs


def ttc_corr_and_g_from_I(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Corr-TTC and G-TTC from intensity matrix I (T, P),
    where averaging is over pixels.

    Corr(t1,t2) = <I(t1)I(t2)> / (<I(t1)><I(t2)>)
    G(t1,t2)    = (<I(t1)I(t2)> - <I(t1)><I(t2)>) / (sigma(t1)sigma(t2))

    Returns (Corr, G), each shape (T, T).
    """
    I = np.asarray(I, dtype=np.float64)
    T, P = I.shape
    if T < 2 or P < 2:
        raise ValueError(f"Need at least 2 frames and 2 pixels, got I={I.shape}")

    # <I(t)> over pixels
    mu = I.mean(axis=1)  # (T,)

    # <I(t1)I(t2)> over pixels:
    # mean over pixels of product = (I @ I.T) / P
    cross = (I @ I.T) / float(P)  # (T,T)

    # Corr TTC
    denom_corr = mu[:, None] * mu[None, :]
    Corr = cross / np.where(denom_corr == 0, np.nan, denom_corr)

    # G TTC
    var = (I * I).mean(axis=1) - mu * mu
    var = np.clip(var, 0.0, np.inf)
    sigma = np.sqrt(var)  # (T,)
    denom_g = sigma[:, None] * sigma[None, :]
    G = (cross - denom_corr) / np.where(denom_g == 0, np.nan, denom_g)

    return Corr, G


def symmetrize_ttc(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    return C + C.T - np.diag(np.diag(C))


def clip_ttc(C: np.ndarray, p_hi: float = 99.9) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    lo, hi = np.nanpercentile(C, [0.0, float(p_hi)])
    return np.clip(C, lo, hi)


def plot_corr_vs_g_ttc(
    Corr: np.ndarray,
    G: np.ndarray,
    *,
    clip_hi_percentile: float = 99.9,
    cmap: str = "plasma",
    figsize=(13.0, 4.6),
):
    """
    Side-by-side:
      Left: Corr-TTC (symmetrized, clipped)
      Mid : G-TTC    (symmetrized, clipped)
      Right: (G - (Corr - 1)) as a diagnostic map

    The diagnostic uses the fully coherent / stable mean relation:
      G ≈ Corr - 1   (see their discussion around eqs. 3–4)  [oai_citation:2‡Ragulskaya et al. - 2024 - On the analysis of two-time correlation functions equilibrium versus non-equilibrium systems.pdf](sediment://file_00000000923c71f88df245b998ba4918)
    """
    Cc = symmetrize_ttc(Corr)
    Cg = symmetrize_ttc(G)

    Cc_plot = clip_ttc(Cc, p_hi=clip_hi_percentile)
    Cg_plot = clip_ttc(Cg, p_hi=clip_hi_percentile)

    # Diagnostic map: how far from G = Corr - 1
    D = Cg - (Cc - 1.0)
    D_plot = clip_ttc(D, p_hi=clip_hi_percentile)

    fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"wspace": 0.30})

    im0 = axs[0].imshow(Cc_plot, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[0].set_title("Corr-TTC")
    axs[0].set_xlabel("t₁ index")
    axs[0].set_ylabel("t₂ index")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(Cg_plot, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[1].set_title("G-TTC")
    axs[1].set_xlabel("t₁ index")
    axs[1].set_ylabel("t₂ index")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(D_plot, origin="lower", cmap="magma", interpolation="nearest", aspect="equal")
    axs[2].set_title("Diagnostic:  G - (Corr - 1)")
    axs[2].set_xlabel("t₁ index")
    axs[2].set_ylabel("t₂ index")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

def _slice_to_start_stop_stride(frame_slice: slice | None, n_frames: int) -> tuple[int, int, int]:
    """
    Convert a Python slice into (start, stop, stride) with bounds clipped to [0, n_frames].

    Examples
    --------
    None             -> (0, n_frames, 1)
    slice(0, 4800)   -> (0, 4800, 1)
    slice(100, 2000, 2) -> (100, 2000, 2)
    """
    if frame_slice is None:
        return 0, int(n_frames), 1

    start = 0 if frame_slice.start is None else int(frame_slice.start)
    stop = n_frames if frame_slice.stop is None else int(frame_slice.stop)
    stride = 1 if frame_slice.step is None else int(frame_slice.step)

    if stride == 0:
        raise ValueError("slice.step cannot be 0")

    # Clip to valid range
    start = max(0, min(int(n_frames), start))
    stop = max(0, min(int(n_frames), stop))

    # Ensure forward slicing (you can add reverse support later if you want)
    if stride < 0:
        raise ValueError("Negative slice.step not supported here. Use positive step.")

    if stop <= start:
        raise ValueError(f"Empty frame_slice after clipping: start={start}, stop={stop}, n_frames={n_frames}")

    return start, stop, stride

def compare_ttc_methods_from_raw(
    run: RunData,
    *,
    mask_n: int,
    frame_slice: slice | None = None,
    clip_hi_percentile: float = 99.9,
):
    """
    Build I(t,p) from raw frames for ROI mask_n, compute Corr and G TTCs, and plot.

    Uses your existing functions:
      - extract_roi_intensity_matrix(...)
      - ttc_corr_and_g_from_I(...)
      - plot_corr_vs_g_ttc(...)
    """
    n_frames = int(run.dset_raw.shape[0])
    start, stop, stride = _slice_to_start_stop_stride(frame_slice, n_frames)

    I, frame_idxs = extract_roi_intensity_matrix(
        run.dset_raw,
        dynamic_roi_map=run.dynamic_roi_map,
        mask_n=int(mask_n),
        start=start,
        stop=stop,
        stride=stride,
    )

    Corr, G = ttc_corr_and_g_from_I(I)

    plot_corr_vs_g_ttc(
        Corr,
        G,
        clip_hi_percentile=clip_hi_percentile,
    )

    return {
        "I": I,
        "frame_idxs": frame_idxs,
        "Corr": Corr,
        "G": G,
    }

def compare_existing_processed_ttc_with_corr_from_raw(
    run: RunData,
    *,
    mask_n: int,
    frame_slice: slice | None = None,
    clip_hi_percentile: float = 99.9,
    diff_symmetric: bool = True,
    cmap: str = "plasma",
    figsize=(14.2, 4.8),
):
    """
    Side-by-side comparison:

      [0] Existing processed TTC (run.ttc)
      [1] Corr TTC computed directly from raw frames for the same ROI
      [2] Difference map: Corr - Existing

    Notes
    -----
    - We compute Corr on a selected set of frames (frame_slice).
    - We compare to the corresponding submatrix of the processed TTC using those same frame indices.
    - By default, we symmetrize the maps for plotting (your preference).
    """
    # ---- compute Corr from raw for requested frames ----
    n_frames = int(run.dset_raw.shape[0])
    start, stop, stride = _slice_to_start_stop_stride(frame_slice, n_frames)

    I, frame_idxs = extract_roi_intensity_matrix(
        run.dset_raw,
        dynamic_roi_map=run.dynamic_roi_map,
        mask_n=int(mask_n),
        start=start,
        stop=stop,
        stride=stride,
    )

    Corr, _G = ttc_corr_and_g_from_I(I)  # we only need Corr here

    # ---- pull matching submatrix from existing processed TTC ----
    C_exist = np.asarray(run.ttc, dtype=np.float64)
    if C_exist.ndim != 2 or C_exist.shape[0] != C_exist.shape[1]:
        raise ValueError(f"run.ttc must be square, got {C_exist.shape}")

    # frame_idxs refer to raw frames. We assume processed TTC uses the same frame indexing.
    # So we take the corresponding rows/cols.
    if np.max(frame_idxs) >= C_exist.shape[0]:
        raise ValueError(
            f"Processed TTC size {C_exist.shape[0]} is smaller than max frame index {np.max(frame_idxs)}. "
            f"Check whether processed TTC was computed on fewer frames than raw."
        )

    C_exist_sub = C_exist[np.ix_(frame_idxs, frame_idxs)]

    # ---- symmetrize (optional) ----
    if diff_symmetric:
        C_exist_sub = symmetrize_ttc(C_exist_sub)
        Corr = symmetrize_ttc(Corr)

    D = Corr - C_exist_sub
    if diff_symmetric:
        D = symmetrize_ttc(D)

    # ---- clip for display ----
    C0 = clip_ttc(C_exist_sub, p_hi=clip_hi_percentile)
    C1 = clip_ttc(Corr, p_hi=clip_hi_percentile)
    Dp = clip_ttc(D, p_hi=clip_hi_percentile)

    # ---- plot ----
    fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"wspace": 0.28})

    im0 = axs[0].imshow(C0, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[0].set_title("Existing processed TTC (submatrix)")
    axs[0].set_xlabel("t₁ index")
    axs[0].set_ylabel("t₂ index")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(C1, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
    axs[1].set_title("Corr TTC from raw")
    axs[1].set_xlabel("t₁ index")
    axs[1].set_ylabel("t₂ index")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(Dp, origin="lower", cmap="magma", interpolation="nearest", aspect="equal")
    axs[2].set_title("Difference: Corr - Existing")
    axs[2].set_xlabel("t₁ index")
    axs[2].set_ylabel("t₂ index")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    for ax in axs:
        ax.grid(False)

    plt.tight_layout()
    plt.show()

    return {
        "frame_idxs": frame_idxs,
        "I": I,
        "Corr": Corr,
        "Existing_sub": C_exist_sub,
        "Diff": D,
    }


def _symmetrize_upper_triangle(C: np.ndarray) -> np.ndarray:
    """If C is stored as upper triangle, make a full symmetric map."""
    C = np.asarray(C)
    return C + C.T - np.diag(np.diag(C))


def _compute_ttc_like_twotime(time_series: np.ndarray) -> np.ndarray:
    """
    Match twotime.py::calc_normal_twotime()

    time_series: (nframes, npix_in_roi)

    Steps (same as twotime.py):
      norm_factor = 1/sum(time_series, axis=1) with <=0 guarded
      c2 = (time_series @ time_series.T) * norm_factor * norm_factor.T * npix
      return triu(c2)
    """
    ts = np.asarray(time_series, dtype=np.float64)

    # norm_factor = 1 / sum_over_pixels_per_frame
    norm_factor = ts.sum(axis=1)
    norm_factor[norm_factor <= 0] = 1.0
    norm_factor = 1.0 / norm_factor  # shape (nframes,)

    # matmul_prod = ts @ ts.T
    matmul_prod = ts @ ts.T  # (nframes, nframes)

    npix = ts.shape[1]
    c2 = matmul_prod * norm_factor[:, None] * norm_factor[None, :] * float(npix)

    # twotime yields torch.triu(c2)
    return np.triu(c2)


def _smooth_like_twotime(time_series: np.ndarray) -> np.ndarray:
    """
    Match twotime.py::compute_smooth_data() *restricted to one ROI*.

    In twotime.py they do, for each ROI:
      avg = cache[roi_pixels].mean(dim=0)    (mean over time per pixel)
      avg[avg <= 0] = 1
      cache[roi_pixels] /= avg

    For a single ROI time_series (nframes, npix):
      avg_pix = mean over time for each pixel -> shape (npix,)
      divide each pixel column by its avg.
    """
    ts = np.asarray(time_series, dtype=np.float64)
    avg_pix = ts.mean(axis=0)
    avg_pix[avg_pix <= 0] = 1.0
    return ts / avg_pix[None, :]


def _load_processed_ttc(hdf_path: Path, mask_n: int) -> np.ndarray:
    """
    Try common TTC dataset naming patterns you've used:
      - xpcs/twotime/correlation_map/c2_00###   (3 digits)
      - xpcs/twotime/correlation_map/c2_#####   (5 digits, twotime generator style)
    """
    key_candidates = [
        f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}",
        f"xpcs/twotime/correlation_map/c2_{int(mask_n):05d}",
    ]

    with h5py.File(hdf_path, "r") as f:
        for k in key_candidates:
            if k in f:
                return f[k][...]

        # If neither matched, give a helpful error listing what exists
        if "xpcs/twotime/correlation_map" in f:
            keys = list(f["xpcs/twotime/correlation_map"].keys())
            keys = sorted(keys)[:50]
            raise KeyError(
                f"Could not find processed TTC for mask {mask_n} in {hdf_path}.\n"
                f"Tried: {key_candidates}\n"
                f"First keys under xpcs/twotime/correlation_map: {keys}"
            )
        else:
            raise KeyError(
                f"Missing group xpcs/twotime/correlation_map in {hdf_path}"
            )


def _extract_roi_time_series_from_raw(
    base_dir: Path,
    sample_id: str,
    mask_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ROI pixel time series from RAW frames, using ROI map from PROCESSED results.

    Returns
    -------
    ts : (nframes, npix)
        Raw intensities for pixels in ROI across time.
    flat_idx : (npix,)
        Flattened pixel indices for the ROI (debugging).
    """
    base_dir = Path(base_dir)

    # ---- locate raw + processed ----
    results_path = find_results_hdf(base_dir, sample_id)   # processed *_results.hdf
    run_dir = find_raw_run_dir(base_dir, sample_id)        # base_dir/data/<run_name>/
    raw_path, _meta_path = find_raw_data_files(run_dir)    # raw <run_name>.h5 etc.

    # ---- load ROI map from processed ----
    with h5py.File(results_path, "r") as f:
        roi_map = f["xpcs/qmap/dynamic_roi_map"][...]

    roi_map = np.asarray(roi_map)
    if roi_map.ndim != 2:
        raise ValueError(f"Unexpected roi_map shape {roi_map.shape} in {results_path}")

    # ---- open raw frames and extract ROI pixels lazily ----
    with h5py.File(raw_path, "r") as f:
        if "entry/data/data" not in f:
            raise KeyError(f"Raw file missing 'entry/data/data': {raw_path}")

        dset = f["entry/data/data"]
        data_shape = dset.shape

        # normalize raw shape: (nframes, ny, nx) or (nframes,1,ny,nx)
        if len(data_shape) == 4 and data_shape[1] == 1:
            nframes, _one, ny, nx = data_shape
            if roi_map.shape != (ny, nx):
                raise ValueError(f"roi_map {roi_map.shape} != raw frame {(ny, nx)} in {raw_path}")

            roi_bool = (roi_map == int(mask_n))
            used = int(mask_n)
            if not np.any(roi_bool):
                roi_bool = (roi_map == int(mask_n) - 1)
                used = int(mask_n) - 1
            if not np.any(roi_bool):
                raise ValueError(f"Mask {mask_n} (or {mask_n-1}) selects 0 pixels in roi_map for {results_path}")

            ts = np.empty((int(nframes), int(np.sum(roi_bool))), dtype=np.float64)
            for i in range(int(nframes)):
                frame = dset[i, 0, :, :]
                ts[i, :] = np.asarray(frame)[roi_bool]

        elif len(data_shape) == 3:
            nframes, ny, nx = data_shape
            if roi_map.shape != (ny, nx):
                raise ValueError(f"roi_map {roi_map.shape} != raw frame {(ny, nx)} in {raw_path}")

            roi_bool = (roi_map == int(mask_n))
            used = int(mask_n)
            if not np.any(roi_bool):
                roi_bool = (roi_map == int(mask_n) - 1)
                used = int(mask_n) - 1
            if not np.any(roi_bool):
                raise ValueError(f"Mask {mask_n} (or {mask_n-1}) selects 0 pixels in roi_map for {results_path}")

            ts = np.empty((int(nframes), int(np.sum(roi_bool))), dtype=np.float64)
            for i in range(int(nframes)):
                frame = dset[i, :, :]
                ts[i, :] = np.asarray(frame)[roi_bool]

        else:
            raise ValueError(f"Unexpected raw dataset shape {data_shape} in {raw_path}")

    flat_idx = np.flatnonzero(roi_bool.reshape(-1))
    return ts, flat_idx

def find_results_hdf(base_dir: Path, sample_id: str) -> Path:
    """
    Processed results live under:
      <BASE_DIR>/Twotime_PostExpt_01/<SAMPLE_ID>_*_results.hdf
    """
    proc_dir = Path(base_dir) / "Twotime_PostExpt_01"
    pattern = f"{sample_id}_*_results.hdf"
    matches = sorted(proc_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No results HDF found in {proc_dir} matching {pattern}")
    return matches[0]


def exec_compare_raw_vs_processed_ttc(
    *,
    base_dir: Path,
    sample_id: str,
    mask_n: int,
    out_path: Path | None = None,
    clip_hi_percentile: float = 99.9,
    cmap_main: str = "plasma",
    cmap_diff: str = "seismic",
    show: bool = True,
    frame_slice: slice | None = None,
    stride: int = 1,
):
    """
    Execution function (uses your existing readers + helpers only):

      1) load_run_data(...) for raw + processed (same HDF readers you already use)
      2) extract ROI intensity matrix I(t,p) from raw frames
      3) compute TTC with the same math as twotime.py (smooth then calc_normal_twotime style)
      4) compare against the processed TTC from the results file
      5) plot Raw | Processed | Raw-Processed

    Notes
    -----
    - Raw TTC and processed TTC are symmetrized before display.
    - Raw TTC uses your _smooth_like_twotime() and _compute_ttc_like_twotime().
    - Display uses percentile clipping (0..clip_hi_percentile) independently for Raw and Processed.
    """
    run = load_run_data(Path(base_dir), str(sample_id), mask_n=int(mask_n))
    try:
        # ----------------------------
        # pick frames from raw (optional)
        # ----------------------------
        n_frames = int(run.dset_raw.shape[0])

        if frame_slice is None:
            start, stop = 0, n_frames
            step = int(stride)
        else:
            start = 0 if frame_slice.start is None else int(frame_slice.start)
            stop = n_frames if frame_slice.stop is None else int(frame_slice.stop)
            step = int(stride) if frame_slice.step is None else int(frame_slice.step)

        start = int(np.clip(start, 0, n_frames))
        stop = int(np.clip(stop, 0, n_frames))
        if step < 1:
            raise ValueError("stride (step) must be >= 1")
        if stop <= start:
            raise ValueError(f"Invalid frame range after clipping: start={start}, stop={stop}, n_frames={n_frames}")

        # ----------------------------
        # build I(t,p) from raw using your existing ROI map logic
        # ----------------------------
        I, frame_idxs = extract_roi_intensity_matrix(
            run.dset_raw,
            dynamic_roi_map=run.dynamic_roi_map,
            mask_n=int(mask_n),
            start=start,
            stop=stop,
            stride=step,
            dtype=np.float64,
        )

        # ----------------------------
        # raw TTC, matching twotime.py math (via your existing helpers)
        # ----------------------------
        I_smooth = _smooth_like_twotime(I)
        C_raw_ut = _compute_ttc_like_twotime(I_smooth)
        C_raw = _symmetrize_upper_triangle(C_raw_ut)

        # ----------------------------
        # processed TTC, already loaded by load_run_data for this mask
        # ----------------------------
        C_proc_ut = np.asarray(run.ttc, dtype=np.float64)

        # If processed TTC is larger than the raw selection, take the matching submatrix
        if C_proc_ut.ndim != 2 or C_proc_ut.shape[0] != C_proc_ut.shape[1]:
            raise ValueError(f"Processed TTC must be square, got {C_proc_ut.shape}")

        if int(np.max(frame_idxs)) >= C_proc_ut.shape[0]:
            raise ValueError(
                f"Processed TTC size {C_proc_ut.shape[0]} is smaller than max selected frame index {int(np.max(frame_idxs))}. "
                f"Either reduce frame_slice, or confirm processed TTC was computed on the same frame count."
            )

        C_proc_ut_sub = C_proc_ut[np.ix_(frame_idxs, frame_idxs)]
        C_proc = _symmetrize_upper_triangle(C_proc_ut_sub)

        # ----------------------------
        # clip for display (independent scaling)
        # ----------------------------
        def _clip(C: np.ndarray) -> np.ndarray:
            lo = np.nanpercentile(C, 0.0)
            hi = np.nanpercentile(C, float(clip_hi_percentile))
            return np.clip(C, lo, hi)

        C_raw_plot = _clip(C_raw)
        C_proc_plot = _clip(C_proc)

        # difference (unclipped, symmetrized)
        C_diff = C_raw - C_proc

        # ----------------------------
        # plot
        # ----------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
        ax0, ax1, ax2 = axes

        im0 = ax0.imshow(C_raw_plot, origin="lower", cmap=cmap_main, aspect="equal", interpolation="nearest")
        ax0.set_title(f"RAW TTC (twotime.py math)\n{sample_id} | M{int(mask_n):03d}")
        ax0.set_xticks([]); ax0.set_yticks([])

        im1 = ax1.imshow(C_proc_plot, origin="lower", cmap=cmap_main, aspect="equal", interpolation="nearest")
        ax1.set_title(f"PROCESSED TTC (from results)\n{sample_id} | M{int(mask_n):03d}")
        ax1.set_xticks([]); ax1.set_yticks([])

        max_abs = float(np.nanmax(np.abs(C_diff)))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0

        im2 = ax2.imshow(
            C_diff,
            origin="lower",
            cmap=cmap_diff,
            aspect="equal",
            interpolation="nearest",
            vmin=-max_abs,
            vmax=+max_abs,
        )
        ax2.set_title(f"DIFF (RAW − PROCESSED)\n{sample_id} | M{int(mask_n):03d}")
        ax2.set_xticks([]); ax2.set_yticks([])

        # colorbars (keep it minimal like you wanted earlier)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03, label="ΔC (a.u.)")
        fig.tight_layout()

        if out_path is not None:
            out_path = Path(out_path)
            fig.savefig(out_path, dpi=250, bbox_inches="tight")
            print(f"Saved: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return {
            "raw_path": str(run.raw_path),
            "meta_path": str(run.meta_path),
            "results_path": str(run.results_path),
            "sample_id": str(sample_id),
            "mask_n": int(mask_n),
            "frame_idxs": frame_idxs,
            "raw_shape_I": tuple(I.shape),
            "raw_ttc_shape": tuple(C_raw.shape),
            "processed_ttc_shape": tuple(C_proc.shape),
        }

    finally:
        run.close()

def compare_loaded_ttc_with_twotime_imported_ttc(
    run: RunData,
    *,
    mask_n: int,
    frame_slice: slice | None = None,
    stride: int = 1,
    clip_hi_percentile: float = 99.9,
    cmap_main: str = "plasma",
    cmap_diff: str = "seismic",
    figsize=(15.5, 5.2),
):
    """
    Compare:
      [0] Loaded processed TTC from results file (run.ttc, submatrix for chosen frames)
      [1] TTC computed using imported twotime.py logic (TwotimeCorrelator)
      [2] Difference map: (twotime_imported - loaded)

    Notes
    -----
    - Uses your existing readers and ROI extraction.
    - twotime.py path: must be importable as `import twotime` or `from twotime import TwotimeCorrelator`.
    - twotime's calc_normal_twotime() is a generator (yields one TTC per dq bin). For one ROI we take the first yield.
    """
    import numpy as np
    import torch
    from twotime import TwotimeCorrelator

    # ----------------------------
    # pick frames (optional)
    # ----------------------------
    n_frames = int(run.dset_raw.shape[0])

    if frame_slice is None:
        start, stop = 0, n_frames
        step = int(stride)
    else:
        start = 0 if frame_slice.start is None else int(frame_slice.start)
        stop = n_frames if frame_slice.stop is None else int(frame_slice.stop)
        step = int(stride) if frame_slice.step is None else int(frame_slice.step)

    start = int(np.clip(start, 0, n_frames))
    stop = int(np.clip(stop, 0, n_frames))
    if step < 1:
        raise ValueError("stride (step) must be >= 1")
    if stop <= start:
        raise ValueError(f"Invalid frame range after clipping: start={start}, stop={stop}, n_frames={n_frames}")

    # ----------------------------
    # build I(t,p) from raw frames (your existing method)
    # ----------------------------
    I, frame_idxs = extract_roi_intensity_matrix(
        run.dset_raw,
        dynamic_roi_map=run.dynamic_roi_map,
        mask_n=int(mask_n),
        start=start,
        stop=stop,
        stride=step,
        dtype=np.float32,
    )
    T, P = I.shape

    # ----------------------------
    # twotime.py TTC via imported functions/classes
    # ----------------------------
    # TwotimeCorrelator expects:
    #   cache shape = (frame_num, arr_size)
    #   dq_slc list of slices over columns; for a single ROI we use slice(0, P)
    qinfo = {
        "dq_idx": np.array([0], dtype=np.int32),
        "dq_slc": [slice(0, P)],
        "sq_idx": np.array([0], dtype=np.int32),
        "sq_slc": [slice(0, P)],
    }

    corr = TwotimeCorrelator(
        qinfo=qinfo,
        frame_num=T,
        det_size=run.dynamic_roi_map.shape,  # not actually used in normal mode TTC math, but required
        device="cpu",
        method="normal",
        dtype=torch.float32,
    )

    # load raw ROI matrix into correlator cache
    corr.process(torch.from_numpy(I))

    # apply the same smoothing step twotime.py uses
    corr.compute_smooth_data()

    # calc_normal_twotime() yields TTC per dq bin (generator)
    gen = corr.calc_normal_twotime()
    C_tw_ut_obj = next(gen)  # first (and only) ROI/dq
    if hasattr(C_tw_ut_obj, "detach"):
        # torch tensor case
        C_tw_ut = C_tw_ut_obj.detach().cpu().numpy()
    else:
        # numpy case (your current twotime.py)
        C_tw_ut = np.asarray(C_tw_ut_obj)
    C_tw = _symmetrize_upper_triangle(C_tw_ut)

    # ----------------------------
    # loaded processed TTC submatrix for the same frames
    # ----------------------------
    C_loaded_ut = np.asarray(run.ttc, dtype=np.float64)
    if C_loaded_ut.ndim != 2 or C_loaded_ut.shape[0] != C_loaded_ut.shape[1]:
        raise ValueError(f"Loaded processed TTC must be square, got {C_loaded_ut.shape}")

    if int(np.max(frame_idxs)) >= C_loaded_ut.shape[0]:
        raise ValueError(
            f"Loaded processed TTC size {C_loaded_ut.shape[0]} is smaller than max selected frame index {int(np.max(frame_idxs))}. "
            f"Either reduce frame_slice, or confirm the processed TTC was computed on the same frame count."
        )

    C_loaded_ut_sub = C_loaded_ut[np.ix_(frame_idxs, frame_idxs)]
    C_loaded = _symmetrize_upper_triangle(C_loaded_ut_sub)

    # ----------------------------
    # difference
    # ----------------------------
    C_diff = C_tw - C_loaded

    # ----------------------------
    # clip for display (independent for main maps)
    # ----------------------------
    C_loaded_plot = clip_ttc(C_loaded, p_hi=float(clip_hi_percentile))
    C_tw_plot = clip_ttc(C_tw, p_hi=float(clip_hi_percentile))

    # symmetric scaling for diff
    max_abs = float(np.nanmax(np.abs(C_diff)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    # ----------------------------
    # plot
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax0, ax1, ax2 = axes

    im0 = ax0.imshow(C_loaded_plot, origin="lower", cmap=cmap_main, aspect="equal", interpolation="nearest")
    ax0.set_title(f"LOADED processed TTC\n{Path(run.results_path).name} | M{int(mask_n):03d}")
    ax0.set_xticks([]); ax0.set_yticks([])

    im1 = ax1.imshow(C_tw_plot, origin="lower", cmap=cmap_main, aspect="equal", interpolation="nearest")
    ax1.set_title(f"twotime.py IMPORT TTC\nframes={T} pix={P} | M{int(mask_n):03d}")
    ax1.set_xticks([]); ax1.set_yticks([])

    im2 = ax2.imshow(
        C_diff,
        origin="lower",
        cmap=cmap_diff,
        aspect="equal",
        interpolation="nearest",
        vmin=-max_abs,
        vmax=+max_abs,
    )
    ax2.set_title("DIFF (twotime_imported − loaded)")
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03, label="ΔC (a.u.)")
    fig.tight_layout()
    plt.show()

    return {
        "frame_idxs": frame_idxs,
        "I_shape": (T, P),
        "loaded_shape": tuple(C_loaded.shape),
        "twotime_shape": tuple(C_tw.shape),
        "diff_max_abs": max_abs,
    }


def exec_compare_loaded_vs_twotime_imported_ttc(
    *,
    base_dir: Path,
    sample_id: str,
    mask_n: int,
    frame_slice: slice | None = None,
    stride: int = 1,
    clip_hi_percentile: float = 99.9,
):
    """
    Execution wrapper for if __name__ == "__main__":.

    Loads run via your existing load_run_data(), then calls
    compare_loaded_ttc_with_twotime_imported_ttc().
    """
    run = load_run_data(Path(base_dir), str(sample_id), mask_n=int(mask_n))
    try:
        return compare_loaded_ttc_with_twotime_imported_ttc(
            run,
            mask_n=int(mask_n),
            frame_slice=frame_slice,
            stride=int(stride),
            clip_hi_percentile=float(clip_hi_percentile),
        )
    finally:
        run.close()

def _gaussian_smooth(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Try scipy gaussian filter, fall back to no smoothing if scipy not available."""
    if sigma_px <= 0:
        return np.asarray(img, dtype=np.float64)
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        return gaussian_filter(np.asarray(img, dtype=np.float64), sigma=float(sigma_px))
    except Exception:
        return np.asarray(img, dtype=np.float64)


def _find_bright_region_center(avg_img: np.ndarray, *, smooth_sigma_px: float = 8.0) -> tuple[int, int]:
    """
    Find (cy, cx) using a smoothed version of avg_img so it reflects a region,
    not a single hot pixel.
    """
    sm = _gaussian_smooth(avg_img, sigma_px=float(smooth_sigma_px))
    cy, cx = np.unravel_index(np.nanargmax(sm), sm.shape)
    return int(cy), int(cx)


def make_bottom_half_ring_mask(
    avg_img: np.ndarray,
    *,
    r_in_px: float,
    r_out_px: float,
    smooth_sigma_px: float = 8.0,
    bottom_is_y_ge_center: bool = True,
) -> np.ndarray:
    """
    Returns a boolean mask for the bottom half of an annulus.

    bottom_is_y_ge_center=True means "bottom" is y >= cy (image origin='upper' convention).
    """
    if r_out_px <= r_in_px:
        raise ValueError(f"Need r_out_px > r_in_px, got {r_in_px=} {r_out_px=}")

    H, W = avg_img.shape
    cy, cx = _find_bright_region_center(avg_img, smooth_sigma_px=float(smooth_sigma_px))

    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    ring = (rr >= float(r_in_px)) & (rr <= float(r_out_px))
    if bottom_is_y_ge_center:
        half = (yy >= cy)
    else:
        half = (yy <= cy)

    return ring & half


# ------------------------------------------------------------
# Utilities: average image, extract I(t,p) for arbitrary mask
# ------------------------------------------------------------

def compute_average_image_from_raw(
    dset_raw,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> np.ndarray:
    """
    Streaming mean of raw frames so we do not load everything into RAM.
    Assumes frames are (T, H, W) or (T, 1, H, W).
    """
    n_frames = int(dset_raw.shape[0])
    if stop is None:
        stop = n_frames
    start = int(np.clip(start, 0, n_frames))
    stop = int(np.clip(stop, 0, n_frames))
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if stop <= start:
        raise ValueError("Empty frame range")

    # figure out frame shape
    f0 = np.asarray(dset_raw[start])
    if f0.ndim == 3 and f0.shape[0] == 1:
        f0 = f0[0]
    if f0.ndim != 2:
        raise ValueError(f"Unexpected frame shape {f0.shape}")

    acc = np.zeros_like(f0, dtype=np.float64)
    count = 0

    for i in range(start, stop, stride):
        fr = np.asarray(dset_raw[i])
        if fr.ndim == 3 and fr.shape[0] == 1:
            fr = fr[0]
        acc += fr.astype(np.float64, copy=False)
        count += 1

    if count == 0:
        raise ValueError("No frames accumulated")
    return acc / float(count)


def extract_intensity_matrix_from_mask(
    dset_raw,
    roi_mask: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build I(t,p) for an arbitrary boolean mask.

    Returns:
      I: (T, P) where P is number of pixels in mask
      frame_idxs: (T,)
    """
    m = np.asarray(roi_mask).astype(bool, copy=False)
    if m.ndim != 2:
        raise ValueError(f"roi_mask must be 2D, got {m.shape}")
    P = int(np.sum(m))
    if P <= 0:
        raise ValueError("roi_mask selects 0 pixels")

    n_frames = int(dset_raw.shape[0])
    if stop is None:
        stop = n_frames
    start = int(np.clip(start, 0, n_frames))
    stop = int(np.clip(stop, 0, n_frames))
    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if stop <= start:
        raise ValueError("Empty frame range")

    frame_idxs = np.arange(start, stop, stride, dtype=int)
    T = int(frame_idxs.size)

    I = np.empty((T, P), dtype=dtype)
    for j, fi in enumerate(frame_idxs):
        fr = np.asarray(dset_raw[int(fi)])
        if fr.ndim == 3 and fr.shape[0] == 1:
            fr = fr[0]
        I[j, :] = fr[m].astype(dtype, copy=False)

    return I, frame_idxs


# ------------------------------------------------------------
# twotime.py TTC computation wrapper (robust to torch/numpy/generator)
# ------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert torch tensor / numpy / generator to numpy array."""
    # generator -> take first (or assemble if it yields multiple)
    if hasattr(x, "__iter__") and not isinstance(x, (np.ndarray, bytes, str)):
        # If it's a generator from twotime, it probably yields once
        x = next(iter(x))

    # torch tensor
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()

    # numpy already
    return np.asarray(x)


def compute_ttc_with_twotime_py(
    I_tp: np.ndarray,
    *,
    do_pixel_smooth: bool = True,
) -> np.ndarray:
    """
    Compute TTC using twotime.py functions, with optional per-pixel smoothing.

    This assumes twotime.py exposes something equivalent to:
      - compute_smooth_data (or similar) for per-pixel normalization
      - calc_normal_twotime (or similar) for TTC

    If the function names in your twotime.py differ, adjust them in one place below.
    """
    # If import fails because twotime.py is not on path, uncomment the import-by-path version.
    import twotime as tw

    # --- OPTIONAL import-by-path fallback ---
    # import importlib.util
    # tw_path = Path(__file__).resolve().parent.parent / "source" / "twotime.py"
    # spec = importlib.util.spec_from_file_location("twotime", tw_path)
    # tw = importlib.util.module_from_spec(spec)
    # assert spec and spec.loader
    # spec.loader.exec_module(tw)

    data = np.asarray(I_tp, dtype=np.float64)

    if do_pixel_smooth:
        # twotime convention: divide each pixel column by its time-mean
        # If your twotime.py expects transposed shapes, adapt here.
        if hasattr(tw, "compute_smooth_data"):
            data = _to_numpy(tw.compute_smooth_data(data))
        elif hasattr(tw, "compute_smooth_data_single_roi"):
            data = _to_numpy(tw.compute_smooth_data_single_roi(data))
        else:
            raise AttributeError("twotime.py missing compute_smooth_data (or equivalent)")

    # TTC computation
    if hasattr(tw, "calc_normal_twotime"):
        C_ut = _to_numpy(tw.calc_normal_twotime(data))
    elif hasattr(tw, "calc_twotime"):
        C_ut = _to_numpy(tw.calc_twotime(data))
    else:
        raise AttributeError("twotime.py missing calc_normal_twotime (or equivalent)")

    return np.asarray(C_ut, dtype=np.float64)


def symmetrize_upper_triangle(C_ut: np.ndarray) -> np.ndarray:
    C = np.asarray(C_ut, dtype=np.float64)
    return C + C.T - np.diag(np.diag(C))


# ------------------------------------------------------------
# Main comparison plot: mask view (left) + TTC (right)
# ------------------------------------------------------------

def plot_mask_and_twotime_ttc(
    run,
    *,
    mask_func: Callable[[np.ndarray], np.ndarray],
    r_in_px: float,
    r_out_px: float,
    smooth_sigma_px: float = 8.0,
    bottom_is_y_ge_center: bool = True,
    frame_start: int = 0,
    frame_stop: int | None = None,
    frame_stride: int = 1,
    do_pixel_smooth: bool = True,
    clip_hi_percentile: float = 99.9,
    cmap_img: str = "magma",
    cmap_ttc: str = "plasma",
    figsize: tuple[float, float] = (12.6, 5.4),
):
    """
    Left: masked average image
    Right: TTC computed by twotime.py (optionally pixel-smoothed)
    """
    # 1) average image from raw
    avg = compute_average_image_from_raw(
        run.dset_raw,
        start=int(frame_start),
        stop=frame_stop,
        stride=int(frame_stride),
    )

    # 2) build custom mask from function
    roi_mask = mask_func(
        avg,
        r_in_px=float(r_in_px),
        r_out_px=float(r_out_px),
        smooth_sigma_px=float(smooth_sigma_px),
        bottom_is_y_ge_center=bool(bottom_is_y_ge_center),
    )

    # 3) left plot data: masked average
    avg_masked = avg.astype(np.float64, copy=True)
    avg_masked[~roi_mask] = np.nan

    # 4) build I(t,p) from raw using this mask
    I, frame_idxs = extract_intensity_matrix_from_mask(
        run.dset_raw,
        roi_mask,
        start=int(frame_start),
        stop=frame_stop,
        stride=int(frame_stride),
        dtype=np.float64,
    )

    # 5) TTC via twotime.py
    C_ut = compute_ttc_with_twotime_py(I, do_pixel_smooth=bool(do_pixel_smooth))
    C = symmetrize_upper_triangle(C_ut)

    # clip for display
    lo = np.nanpercentile(C, 0.0)
    hi = np.nanpercentile(C, float(clip_hi_percentile))
    C_plot = np.clip(C, lo, hi)

    # 6) plot
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.18})

    imL = axL.imshow(avg_masked, origin="upper", cmap=cmap_img, interpolation="nearest")
    axL.set_title(f"Masked average image\nring [{r_in_px:.0f}, {r_out_px:.0f}] px, bottom-half")
    axL.set_xticks([])
    axL.set_yticks([])
    fig.colorbar(imL, ax=axL, fraction=0.046, pad=0.03, label="ADU (masked)")

    imR = axR.imshow(C_plot, origin="lower", cmap=cmap_ttc, interpolation="nearest", aspect="equal")
    axR.set_title(f"twotime.py TTC\nframes {frame_idxs[0]}..{frame_idxs[-1]} step {frame_stride}")
    axR.set_xticks([])
    axR.set_yticks([])
    fig.colorbar(imR, ax=axR, fraction=0.046, pad=0.03, label="C(t₁,t₂) (clipped)")

    plt.tight_layout()
    plt.show()

    return {
        "roi_mask": roi_mask,
        "avg_image": avg,
        "I_shape": I.shape,
        "frame_idxs": frame_idxs,
        "C_ut_shape": C_ut.shape,
    }

def make_bottom_half_ring_mask_centered_on_brightest_region(
    scattering_2d: np.ndarray,
    *,
    r_inner_px: float,
    r_outer_px: float,
    bright_percentile: float = 99.7,
    center_px: tuple[float, float] | None = None,   # NEW
):
    """
    Returns:
      roi_mask (bool 2D),
      (cy, cx) center used (float)
    """
    img = np.asarray(scattering_2d, dtype=np.float64)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f"Expected scattering_2d to be 2D (or (1,H,W)), got {img.shape}")

    H, W = img.shape

    # -------------------------
    # Center selection
    # -------------------------
    if center_px is None:
        # EXISTING behaviour (keep your current logic here)
        # (Whatever you already do to compute (cy, cx) from bright_percentile.)
        thr = np.nanpercentile(img, float(bright_percentile))
        m = np.isfinite(img) & (img >= thr)
        if not np.any(m):
            raise ValueError("Bright-region mask is empty, lower bright_percentile")

        ys, xs = np.where(m)
        w = img[m]
        w = np.maximum(w, 0.0)
        if np.all(w == 0):
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
        else:
            cy = float(np.sum(ys * w) / np.sum(w))
            cx = float(np.sum(xs * w) / np.sum(w))
    else:
        cy, cx = float(center_px[0]), float(center_px[1])
        # optional sanity clip:
        cy = float(np.clip(cy, 0, H - 1))
        cx = float(np.clip(cx, 0, W - 1))

    # -------------------------
    # Build bottom-half ring mask about (cy,cx)
    # -------------------------
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    ring = (rr >= float(r_inner_px)) & (rr <= float(r_outer_px))

    # "bottom half": define bottom as yy > cy (image coordinates)
    bottom = (yy >= cy)

    roi_mask = ring & bottom
    return roi_mask, (cy, cx)


def plot_custom_mask_and_twotime_ttc(
    run: RunData,
    *,
    roi_mask: np.ndarray,
    mask_title: str = "Custom mask",
    frame_slice: slice | None = None,
    stride: int = 1,
    do_pixel_smooth: bool = True,
    clip_hi_percentile: float = 99.9,
    cmap_mask: str = "magma",
    cmap_ttc: str = "plasma",
    figsize: tuple[float, float] = (12.8, 5.4),
):
    """
    Left: masked average image (run.scattering_2d)
    Right: TTC computed via twotime.py TwotimeCorrelator (same logic as your working compare func)
    """
    import torch
    from twotime import TwotimeCorrelator

    # ----------------------------
    # choose frames
    # ----------------------------
    n_frames = int(run.dset_raw.shape[0])
    if frame_slice is None:
        start, stop = 0, n_frames
        step = int(stride)
    else:
        start = 0 if frame_slice.start is None else int(frame_slice.start)
        stop = n_frames if frame_slice.stop is None else int(frame_slice.stop)
        step = int(stride) if frame_slice.step is None else int(frame_slice.step)

    start = int(np.clip(start, 0, n_frames))
    stop = int(np.clip(stop, 0, n_frames))
    if step < 1:
        raise ValueError("stride must be >= 1")
    if stop <= start:
        raise ValueError(f"Invalid frame range after clipping: start={start}, stop={stop}, n_frames={n_frames}")

    # ----------------------------
    # build I(t,p) using your existing extractor
    # ----------------------------
    I, frame_idxs = extract_roi_intensity_matrix(
        run.dset_raw,
        roi_mask=np.asarray(roi_mask, dtype=bool),
        start=start,
        stop=stop,
        stride=step,
        dtype=np.float32,   # keep memory down
    )
    T, P = I.shape
    if P < 2:
        raise ValueError(f"Custom ROI has too few pixels: P={P}")

    # ----------------------------
    # twotime.py TTC (exact same pattern as your working compare function)
    # ----------------------------
    qinfo = {
        "dq_idx": np.array([0], dtype=np.int32),
        "dq_slc": [slice(0, P)],
        "sq_idx": np.array([0], dtype=np.int32),
        "sq_slc": [slice(0, P)],
    }

    corr = TwotimeCorrelator(
        qinfo=qinfo,
        frame_num=T,
        det_size=run.scattering_2d.shape,
        device="cpu",
        method="normal",
        dtype=torch.float32,
    )

    corr.process(torch.from_numpy(I))

    if bool(do_pixel_smooth):
        # divides each pixel column by its time-average (twotime's per-pixel flattening)
        corr.compute_smooth_data()

    gen = corr.calc_normal_twotime()
    C_ut_obj = next(gen)  # first/only dq bin
    C_ut = np.asarray(C_ut_obj)  # your current twotime.py yields numpy

    C = _symmetrize_upper_triangle(C_ut)

    # clip for display
    lo = np.nanpercentile(C, 0.0)
    hi = np.nanpercentile(C, float(clip_hi_percentile))
    Cplot = np.clip(C, lo, hi)

    # ----------------------------
    # masked average image for display
    # ----------------------------
    avg = np.asarray(run.scattering_2d, dtype=np.float64)
    avg_masked = avg.copy()
    avg_masked[~np.asarray(roi_mask, dtype=bool)] = np.nan

    a_lo = np.nanpercentile(avg, 1.0)
    a_hi = np.nanpercentile(avg, 99.9)

    # ----------------------------
    # plot
    # ----------------------------
    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.22})

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    pad = 20  # adjust crop margin in pixels

    # avg_img should be your average image (2D)
    # roi_mask should be your boolean ROI mask (2D, same shape)

    y0, y1, x0, x1 = _bbox_from_mask(roi_mask, pad=pad)

    avg_crop = avg_masked[y0:y1, x0:x1]
    mask_crop = roi_mask[y0:y1, x0:x1]

    # If you’re showing "masked average", keep outside ROI as NaN (or 0)
    avg_crop_masked = avg_crop.astype(float, copy=True)
    avg_crop_masked[~mask_crop] = np.nan

    # avoid zeros / negatives for LogNorm
    img = np.asarray(avg_crop_masked, dtype=np.float64)
    img = np.where(img > 0, img, np.nan)

    vmin = np.nanpercentile(img, 1.0)
    vmax = np.nanpercentile(img, clip_hi_percentile)

    im0 = axs[0].imshow(img,
                        origin="upper",
                        cmap="magma",
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        )
    axs[0].set_title(mask_title)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.03)

    im1 = axs[1].imshow(Cplot, origin="lower", cmap=cmap_ttc, aspect="equal", interpolation="nearest")
    axs[1].set_title(f"twotime.py TTC | frames={T} | pixels={P} | smooth={bool(do_pixel_smooth)}")
    axs[1].set_xticks([]); axs[1].set_yticks([])
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.03)

    plt.tight_layout()
    plt.show()

    return {
        "frame_idxs": frame_idxs,
        "I_shape": (int(T), int(P)),
        "C_shape": tuple(C.shape),
    }

def _bbox_from_mask(mask: np.ndarray, pad: int = 10):
    """
    Return (y0, y1, x0, x1) bounding box around True pixels, with padding.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("ROI mask has zero True pixels")
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1

def make_radial_mask(
    image_shape: tuple[int, int],
    *,
    center_rc: tuple[float, float],
    r_in: float = 0.0,
    r_out: float = 50.0,
    half: str = "bottom",   # "bottom", "top", or "full"
    filled: bool = False,   # False -> annulus (ring), True -> filled disk
) -> np.ndarray:
    """
    Returns a boolean ROI mask.

    Parameters
    ----------
    image_shape : (H, W)
    center_rc   : (cy, cx) in pixel coordinates (row, col)
    r_in, r_out : inner/outer radii in pixels
    half        : "bottom" keeps rows >= cy, "top" keeps rows <= cy, "full" keeps all
    filled      : if True, ignore r_in (treat as 0) to make a filled disk

    Notes
    -----
    - "bottom" vs "top" uses row index convention (increasing row goes downward).
    """
    H, W = image_shape
    cy, cx = map(float, center_rc)

    yy, xx = np.ogrid[:H, :W]
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dx * dx + dy * dy)

    r_out = float(r_out)
    r_in = 0.0 if filled else float(r_in)

    if r_out <= 0:
        raise ValueError("r_out must be > 0")
    if (not filled) and r_in < 0:
        raise ValueError("r_in must be >= 0")
    if (not filled) and r_in >= r_out:
        raise ValueError("Need r_in < r_out for an annulus")

    radial = (rr <= r_out) if filled else ((rr >= r_in) & (rr <= r_out))

    if half == "full":
        hemi = np.ones((H, W), dtype=bool)
    elif half == "bottom":
        hemi = (yy >= cy)
    elif half == "top":
        hemi = (yy <= cy)
    else:
        raise ValueError("half must be one of: 'bottom', 'top', 'full'")

    return (radial & hemi).astype(bool)

def exec_mask_and_twotime_ttc_custom_ring(
    *,
    base_dir: Path,
    sample_id: str,
    mask_n_for_loading: int,
    r_inner_px: float = 10.0,
    r_outer_px: float = 25.0,
    bright_percentile: float = 99.7,
    center_px: tuple[float, float] | None = None,
    shape: str = "semi",  # "semi" or "circle"
    fill: str = "ring",
    frame_slice: slice | None = None,
    stride: int = 1,
    do_pixel_smooth: bool = True,
    clip_hi_percentile: float = 99.9,
):
    """
    Execution wrapper:
      - uses existing load_run_data
      - builds custom ROI mask around brightest-region centroid
      - plots [masked avg] | [twotime TTC]
    """
    run = load_run_data(Path(base_dir), str(sample_id), mask_n=int(mask_n_for_loading))
    try:
        # --- choose centre: manual override or brightest-region centroid ---
        img = np.asarray(run.scattering_2d, dtype=np.float64)

        if center_px is None:
            thresh = np.percentile(img, float(bright_percentile))
            ys, xs = np.where(img >= thresh)
            if ys.size == 0:
                raise RuntimeError(f"No pixels above bright_percentile={bright_percentile}")
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
        else:
            cx = float(center_px[0])
            cy = float(center_px[1])

        half = "full" if shape.lower() in ("circle", "full") else "bottom"
        filled = True if fill.lower() in ("solid", "filled", "disk") else False

        # if it's solid, r_inner_px is irrelevant – but we can ignore it safely
        roi_mask = make_radial_mask(
            img.shape,
            center_rc=(cy, cx),
            r_in=0.0 if filled else float(r_inner_px),
            r_out=float(r_outer_px),
            half=half,  # "bottom" or "full"
            filled=filled,  # False=ring, True=solid disk/semidisk
        )

        return plot_custom_mask_and_twotime_ttc(
            run,
            roi_mask=roi_mask,
            mask_title=(
                f"{'Bottom-half' if half == 'bottom' else 'Full'} "
                f"{'solid' if filled else 'ring'} mask\n"
                f"center≈({cy:.1f},{cx:.1f}), r=[{(0.0 if filled else r_inner_px):.1f},{r_outer_px:.1f}] px"
            ),
            frame_slice=frame_slice,
            stride=int(stride),
            do_pixel_smooth=bool(do_pixel_smooth),
            clip_hi_percentile=float(clip_hi_percentile),
        )

    finally:
        run.close()

# ============================================================
# Execution functions
# ============================================================

def _print_h5_tree(
    f: h5py.File,
    *,
    max_depth: int = 6,
    max_children_per_group: int = 200,
    show_attrs: bool = False,
) -> None:
    """
    Print an HDF5 tree: groups + datasets with shape/dtype.
    Keeps output bounded via max_depth and max_children_per_group.
    """

    def _fmt_attrs(obj) -> str:
        if not show_attrs:
            return ""
        try:
            keys = list(obj.attrs.keys())
        except Exception:
            keys = []
        if not keys:
            return ""
        keys = keys[:12]
        return f"  attrs={keys}{'...' if len(keys) == 12 else ''}"

    def _recurse(g: h5py.Group, prefix: str, depth: int) -> None:
        if depth > max_depth:
            print(prefix + "… (max_depth reached)")
            return

        try:
            items = list(g.items())
        except Exception as e:
            print(prefix + f"(cannot list items: {e})")
            return

        if len(items) > max_children_per_group:
            items = items[:max_children_per_group]
            truncated = True
        else:
            truncated = False

        for name, obj in items:
            path = obj.name
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                # show chunking/compression if present
                chunks = obj.chunks
                comp = obj.compression
                extra = []
                if chunks is not None:
                    extra.append(f"chunks={chunks}")
                if comp is not None:
                    extra.append(f"compression={comp}")
                extra_s = ("  " + ", ".join(extra)) if extra else ""
                print(prefix + f"- {path}  [Dataset] shape={shape} dtype={dtype}{extra_s}{_fmt_attrs(obj)}")
            elif isinstance(obj, h5py.Group):
                print(prefix + f"+ {path}  [Group]{_fmt_attrs(obj)}")
                _recurse(obj, prefix + "  ", depth + 1)
            else:
                print(prefix + f"? {path}  [{type(obj)}]{_fmt_attrs(obj)}")

        if truncated:
            print(prefix + f"… ({max_children_per_group} children shown, truncated)")

    print(f"\nFILE: {getattr(f, 'filename', '<unknown>')}")
    print("+ /  [Group]")
    _recurse(f["/"], prefix="  ", depth=1)


def _open_h5_safely(path: Path) -> h5py.File:
    # hdf5plugin imported at top already, keep as-is
    return h5py.File(Path(path), "r")


def data_structure_viewer():
    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    print("\nLoaded:")
    print("  raw:", run.raw_path)
    print("  meta:", run.meta_path)
    print("  results:", run.results_path)

    # Raw file (keep handle open via RunData)
    _print_h5_tree(run.f_raw, max_depth=7, max_children_per_group=300, show_attrs=False)

    # Meta file (keep handle open via RunData)
    _print_h5_tree(run.f_meta, max_depth=7, max_children_per_group=300, show_attrs=False)

    # Results file (open separately because RunData only keeps arrays from it)
    f_res = _open_h5_safely(run.results_path)
    try:
        _print_h5_tree(f_res, max_depth=7, max_children_per_group=300, show_attrs=False)
    finally:
        f_res.close()

    run.close()


# ------------------------------------------------------------------ #
#  Waterfall ROI analysis: ravel a rectangular ROI into 1-D and       #
#  stack over time to produce kymograph-style waterfall plots.         #
# ------------------------------------------------------------------ #

def waterfall_roi_viewer(
    run: RunData,
    *,
    crop_size: int = 300,
    roi_offset_x: int = 30,
    roi_offset_y: int = 0,
    roi_w: int = 20,
    roi_h: int = 20,
    stride: int = 20,
    dt_s: float = 1.0,
    cmap_frame: str = "magma",
    cmap_waterfall: str = "inferno",
    log_eps: float = 1.0,
    start_frame: int = 0,
):
    """Interactive waterfall ROI analysis around the Bragg peak.

    Left panel : 300x300 crop around the Bragg peak with the rectangular
                 ROI boundary overlaid.  A slider scrubs through frames.
    Right panel: two stacked waterfall (kymograph) plots built by raveling
                 the ROI in row-major (top) and column-major (bottom) order.
                 X-axis = time, Y-axis = pixel index, intensity = log scale.
                 A vertical red line tracks the currently selected frame.

    Click on the left panel to reposition the ROI centre (waterfalls are
    recomputed automatically).

    Parameters
    ----------
    run            : RunData with an open dset_raw handle.
    crop_size      : side length of the square crop around the Bragg peak.
    roi_offset_x/y : initial offset of the ROI centre from the Bragg peak
                     (in pixels, relative to the crop coordinate system).
    roi_w, roi_h   : width and height of the rectangular ROI.
    stride         : frame step for the waterfall (every Nth frame).
    dt_s           : time per raw frame in seconds (used for the x-axis).
    cmap_frame     : colormap for the left frame panel.
    cmap_waterfall : colormap for the waterfall panels.
    log_eps        : epsilon added before log10 to avoid log(0).
    start_frame    : initial frame index shown.
    """

    # ---- helpers ----
    def _read_frame2d(idx: int) -> np.ndarray:
        fr = np.asarray(run.dset_raw[int(idx)])
        if fr.ndim == 3 and fr.shape[0] == 1:
            fr = fr[0]
        return fr.astype(np.float64, copy=False)

    def _find_peak_center(frame2d: np.ndarray) -> tuple[int, int]:
        """Return (cy, cx) of the bright region (robust to hot pixels)."""
        f = np.clip(frame2d, 0.0, None)
        pos = f[f > 0]
        if pos.size:
            clip_hi = float(np.percentile(pos, 99.9))
        else:
            clip_hi = float(np.nanmax(f)) if np.isfinite(np.nanmax(f)) else 0.0
        f = np.minimum(f, clip_hi)
        try:
            from scipy.ndimage import uniform_filter
            score = uniform_filter(f, size=21, mode="nearest")
            flat = int(np.nanargmax(score))
            cy, cx = np.unravel_index(flat, score.shape)
        except Exception:
            flat = int(np.nanargmax(f))
            cy, cx = np.unravel_index(flat, f.shape)
        return int(cy), int(cx)

    # ---- data geometry ----
    n_frames = int(run.dset_raw.shape[0])
    if n_frames <= 0:
        raise ValueError("Empty dataset: no frames")

    frame0 = _read_frame2d(start_frame)
    peak_cy, peak_cx = _find_peak_center(frame0)

    # Crop boundaries (full detector -> crop)
    H, W = frame0.shape
    half = crop_size // 2
    crop_y0 = max(0, peak_cy - half)
    crop_y1 = min(H, peak_cy + half + (crop_size % 2))
    crop_x0 = max(0, peak_cx - half)
    crop_x1 = min(W, peak_cx + half + (crop_size % 2))

    # Peak position in crop coordinates
    peak_cy_crop = peak_cy - crop_y0
    peak_cx_crop = peak_cx - crop_x0

    # Frame indices for the waterfall
    frame_indices = np.arange(0, n_frames, stride)
    n_wf = len(frame_indices)
    times = frame_indices * dt_s

    # ---- mutable state ----
    # ROI is stored as absolute crop-pixel corners (y0, y1, x0, x1)
    ch_crop, cw_crop = crop_y1 - crop_y0, crop_x1 - crop_x0
    _init_cx = peak_cx_crop + roi_offset_x
    _init_cy = peak_cy_crop + roi_offset_y
    _hw, _hh = roi_w // 2, roi_h // 2
    state = {
        "frame_idx": int(np.clip(start_frame, 0, n_frames - 1)),
        # ROI corners in crop coordinates
        "roi_x0": max(0, _init_cx - _hw),
        "roi_x1": min(cw_crop, _init_cx + _hw + (roi_w % 2)),
        "roi_y0": max(0, _init_cy - _hh),
        "roi_y1": min(ch_crop, _init_cy + _hh + (roi_h % 2)),
        # Drag state
        "drag_start": None,  # (x, y) in crop pixels or None
    }

    # ---- ROI extraction helpers ----
    def _roi_bounds():
        """Return (y0, y1, x0, x1) in crop coordinates, clipped."""
        return int(state["roi_y0"]), int(state["roi_y1"]), int(state["roi_x0"]), int(state["roi_x1"])

    def _extract_roi_from_crop(crop2d: np.ndarray):
        """Return the ROI sub-array from a crop."""
        y0, y1, x0, x1 = _roi_bounds()
        return crop2d[y0:y1, x0:x1]

    def _get_crop(frame_idx: int) -> np.ndarray:
        """Read a frame and return the crop around the peak."""
        fr = _read_frame2d(frame_idx)
        return fr[crop_y0:crop_y1, crop_x0:crop_x1]

    # ---- build waterfall data ----
    def _build_waterfalls():
        """Build both row-major and column-major waterfall arrays."""
        y0, y1, x0, x1 = _roi_bounds()
        roi_pixels = (y1 - y0) * (x1 - x0)
        if roi_pixels == 0:
            return np.zeros((1, n_wf)), np.zeros((1, n_wf))

        wf_row = np.zeros((roi_pixels, n_wf), dtype=np.float64)
        wf_col = np.zeros((roi_pixels, n_wf), dtype=np.float64)

        for k, fi in enumerate(frame_indices):
            crop = _get_crop(int(fi))
            roi = _extract_roi_from_crop(crop)
            wf_row[:, k] = roi.ravel(order="C")   # row-major (top-to-bottom)
            wf_col[:, k] = roi.ravel(order="F")   # column-major (left-to-right)
            if (k + 1) % 500 == 0:
                print(f"  waterfall: processed {k+1}/{n_wf} frames")

        return wf_row, wf_col

    print(f"Building waterfall from {n_wf} frames (stride={stride}) ...")
    wf_row, wf_col = _build_waterfalls()
    print("  done.")

    # ---- figure layout ----
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1, 1.6],
        height_ratios=[1, 0.06, 0.06],
        hspace=0.35, wspace=0.30,
        left=0.06, right=0.96, top=0.93, bottom=0.06,
    )

    # Left: frame view (spans 2 rows)
    ax_frame = fig.add_subplot(gs[0, 0])

    # Right: two waterfall panels stacked
    gs_right = gs[0, 1].subgridspec(2, 1, hspace=0.35)
    ax_wf_row = fig.add_subplot(gs_right[0])
    ax_wf_col = fig.add_subplot(gs_right[1])

    # Slider row
    ax_slider = fig.add_subplot(gs[1, :])

    # Status text row
    ax_status = fig.add_subplot(gs[2, :])
    ax_status.axis("off")

    # ---- draw the left panel (frame + ROI rect) ----
    crop0 = _get_crop(state["frame_idx"])
    disp0 = np.log10(np.clip(crop0, 0.0, None) + log_eps)

    im_frame = ax_frame.imshow(
        disp0, origin="upper", cmap=cmap_frame, interpolation="nearest",
    )
    ax_frame.set_xlabel("x (pixel)")
    ax_frame.set_ylabel("y (pixel)")

    divider = make_axes_locatable(ax_frame)
    cax_frame = divider.append_axes("right", size="4%", pad=0.06)
    fig.colorbar(im_frame, cax=cax_frame, label="log10(I + eps)")

    # ROI rectangle overlay
    y0, y1, x0, x1 = _roi_bounds()
    roi_rect = plt.Rectangle(
        (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
    )
    ax_frame.add_patch(roi_rect)

    # Mark peak center
    ax_frame.plot(peak_cx_crop, peak_cy_crop, "r+", markersize=10, markeredgewidth=1.5)

    def _frame_title():
        rw = state["roi_x1"] - state["roi_x0"]
        rh = state["roi_y1"] - state["roi_y0"]
        rcx = (state["roi_x0"] + state["roi_x1"]) // 2
        rcy = (state["roi_y0"] + state["roi_y1"]) // 2
        return (
            f"Frame {state['frame_idx']}/{n_frames-1}  |  "
            f"ROI {rw}x{rh}  |  "
            f"offset ({rcx - peak_cx_crop}, {rcy - peak_cy_crop})"
        )
    ax_frame.set_title(_frame_title(), fontsize=9)

    # ---- draw the waterfall panels ----
    vmin_wf = np.log10(log_eps)
    vmax_wf = float(np.nanpercentile(wf_row[wf_row > 0], 99.5)) if np.any(wf_row > 0) else 1.0
    vmax_wf = np.log10(max(vmax_wf, log_eps) + log_eps)

    wf_row_log = np.log10(np.clip(wf_row, 0.0, None) + log_eps)
    wf_col_log = np.log10(np.clip(wf_col, 0.0, None) + log_eps)

    extent_wf = [times[0], times[-1], wf_row.shape[0] - 0.5, -0.5]

    im_wf_row = ax_wf_row.imshow(
        wf_row_log, aspect="auto", origin="upper", cmap=cmap_waterfall,
        extent=extent_wf, interpolation="nearest",
    )
    ax_wf_row.set_ylabel("pixel idx (row-major)")
    ax_wf_row.set_title("Waterfall — ravel top→bottom, left→right", fontsize=9)

    im_wf_col = ax_wf_col.imshow(
        wf_col_log, aspect="auto", origin="upper", cmap=cmap_waterfall,
        extent=extent_wf, interpolation="nearest",
    )
    ax_wf_col.set_xlabel("Time (s)")
    ax_wf_col.set_ylabel("pixel idx (col-major)")
    ax_wf_col.set_title("Waterfall — ravel left→right, top→bottom", fontsize=9)

    # Colorbars for waterfalls
    div_r = make_axes_locatable(ax_wf_row)
    cax_r = div_r.append_axes("right", size="3%", pad=0.06)
    fig.colorbar(im_wf_row, cax=cax_r, label="log10(I + eps)")

    div_c = make_axes_locatable(ax_wf_col)
    cax_c = div_c.append_axes("right", size="3%", pad=0.06)
    fig.colorbar(im_wf_col, cax=cax_c, label="log10(I + eps)")

    # Vertical time markers
    t_current = state["frame_idx"] * dt_s
    vline_row = ax_wf_row.axvline(t_current, color="red", linewidth=2.5, linestyle="--", alpha=0.8)
    vline_col = ax_wf_col.axvline(t_current, color="red", linewidth=2.5, linestyle="--", alpha=0.8)

    # ---- slider ----
    s_frame = Slider(
        ax_slider, "Frame", 0, n_frames - 1,
        valinit=state["frame_idx"], valstep=1, valfmt="%d",
    )

    # ---- status text ----
    status_txt = ax_status.text(
        0.5, 0.5, "", transform=ax_status.transAxes,
        ha="center", va="center", fontsize=8,
    )

    def _update_status(msg: str = ""):
        status_txt.set_text(msg)

    # ---- redraw helpers ----
    def _redraw_frame():
        crop = _get_crop(state["frame_idx"])
        disp = np.log10(np.clip(crop, 0.0, None) + log_eps)
        im_frame.set_data(disp)
        ax_frame.set_title(_frame_title(), fontsize=9)

        # Update ROI rect
        y0, y1, x0, x1 = _roi_bounds()
        roi_rect.set_xy((x0 - 0.5, y0 - 0.5))
        roi_rect.set_width(x1 - x0)
        roi_rect.set_height(y1 - y0)

        # Update time marker
        t_now = state["frame_idx"] * dt_s
        vline_row.set_xdata([t_now, t_now])
        vline_col.set_xdata([t_now, t_now])

        fig.canvas.draw_idle()

    def _redraw_waterfalls():
        """Recompute and redraw both waterfall images."""
        _update_status("Recomputing waterfalls ...")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        wf_r, wf_c = _build_waterfalls()
        wf_r_log = np.log10(np.clip(wf_r, 0.0, None) + log_eps)
        wf_c_log = np.log10(np.clip(wf_c, 0.0, None) + log_eps)

        extent = [times[0], times[-1], wf_r.shape[0] - 0.5, -0.5]

        im_wf_row.set_data(wf_r_log)
        im_wf_row.set_extent(extent)
        im_wf_col.set_data(wf_c_log)
        im_wf_col.set_extent(extent)

        # Update color limits
        valid = wf_r_log[np.isfinite(wf_r_log)]
        if valid.size:
            im_wf_row.set_clim(float(np.nanmin(valid)), float(np.nanmax(valid)))
        valid_c = wf_c_log[np.isfinite(wf_c_log)]
        if valid_c.size:
            im_wf_col.set_clim(float(np.nanmin(valid_c)), float(np.nanmax(valid_c)))

        _update_status("")
        _redraw_frame()

    # ---- callbacks ----
    def on_slider(val):
        state["frame_idx"] = int(round(val))
        _redraw_frame()

    s_frame.on_changed(on_slider)

    def on_key(event):
        if event.key in ("right", "d"):
            state["frame_idx"] = min(n_frames - 1, state["frame_idx"] + 1)
        elif event.key in ("left", "a"):
            state["frame_idx"] = max(0, state["frame_idx"] - 1)
        elif event.key == "home":
            state["frame_idx"] = 0
        elif event.key == "end":
            state["frame_idx"] = n_frames - 1
        else:
            return
        s_frame.set_val(state["frame_idx"])
        _redraw_frame()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ---- drag-to-draw ROI ----
    # A temporary rectangle is shown during the drag; on release,
    # the ROI is updated and waterfalls are recomputed.
    drag_rect = plt.Rectangle(
        (0, 0), 0, 0,
        linewidth=1.2, edgecolor="lime", facecolor="lime", alpha=0.15,
        linestyle="-", visible=False,
    )
    ax_frame.add_patch(drag_rect)

    def _on_press(event):
        if event.inaxes is not ax_frame or event.button != 1:
            return
        sx = int(round(max(0, min(cw_crop - 1, event.xdata))))
        sy = int(round(max(0, min(ch_crop - 1, event.ydata))))
        state["drag_start"] = (sx, sy)
        drag_rect.set_xy((sx - 0.5, sy - 0.5))
        drag_rect.set_width(0)
        drag_rect.set_height(0)
        drag_rect.set_visible(True)
        fig.canvas.draw_idle()

    def _on_motion(event):
        if state["drag_start"] is None:
            return
        if event.inaxes is not ax_frame:
            return
        sx, sy = state["drag_start"]
        ex = int(round(max(0, min(cw_crop - 1, event.xdata))))
        ey = int(round(max(0, min(ch_crop - 1, event.ydata))))
        x0, x1 = min(sx, ex), max(sx, ex)
        y0, y1 = min(sy, ey), max(sy, ey)
        drag_rect.set_xy((x0 - 0.5, y0 - 0.5))
        drag_rect.set_width(x1 - x0 + 1)
        drag_rect.set_height(y1 - y0 + 1)
        fig.canvas.draw_idle()

    def _on_release(event):
        if state["drag_start"] is None:
            return
        if event.button != 1:
            return
        drag_rect.set_visible(False)

        sx, sy = state["drag_start"]
        state["drag_start"] = None

        # End position (clamp to crop)
        if event.inaxes is ax_frame and event.xdata is not None:
            ex = int(round(max(0, min(cw_crop - 1, event.xdata))))
            ey = int(round(max(0, min(ch_crop - 1, event.ydata))))
        else:
            ex, ey = sx, sy  # mouse left the axes — cancel

        x0, x1 = min(sx, ex), max(sx, ex) + 1  # +1 so a 1-pixel drag gives width 1
        y0, y1 = min(sy, ey), max(sy, ey) + 1

        # Require a minimum 2x2 ROI to avoid accidental clicks
        if (x1 - x0) < 2 or (y1 - y0) < 2:
            fig.canvas.draw_idle()
            return

        state["roi_x0"] = x0
        state["roi_x1"] = x1
        state["roi_y0"] = y0
        state["roi_y1"] = y1
        print(f"ROI drawn: x=[{x0},{x1}), y=[{y0},{y1})  "
              f"size={x1-x0}x{y1-y0}  "
              f"centre offset from peak = "
              f"({(x0+x1)//2 - peak_cx_crop}, {(y0+y1)//2 - peak_cy_crop})")
        _redraw_waterfalls()

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)

    plt.show()


def waterfall_roi_entrypoint(
    *,
    scan_id: str | None = None,
    base_dir: Path | None = None,
    crop_size: int = 300,
    roi_offset_x: int = 30,
    roi_offset_y: int = 0,
    roi_w: int = 20,
    roi_h: int = 20,
    stride: int = 20,
    dt_s: float = 1.0,
):
    """
    Convenience wrapper: loads raw data and launches waterfall_roi_viewer.

    Parameters mirror waterfall_roi_viewer; scan_id and base_dir default to
    workflow_config values if not supplied.
    """
    scan = scan_id if scan_id is not None else SAMPLE_ID
    if base_dir is None:
        base_dir = _resolve_base_dir(scan)
    else:
        base_dir = Path(base_dir)

    try:
        run = load_run_data(base_dir, scan, mask_n=MASK_N)
    except FileNotFoundError:
        run = load_raw_data_only(base_dir, scan)

    try:
        waterfall_roi_viewer(
            run,
            crop_size=crop_size,
            roi_offset_x=roi_offset_x,
            roi_offset_y=roi_offset_y,
            roi_w=roi_w,
            roi_h=roi_h,
            stride=stride,
            dt_s=dt_s,
        )
    finally:
        run.close()


def mask_roi_viewer_mp4_save(
    *,
    stride: int = 20,
    position_name: str | None = None,
    scan_id: str | None = None,
    mask_n: int | None = None,
    crop_size: int = 300,
    base_dir: Path | None = None,
    require_processed: bool = False,
):
    """
    Interactive masked raw viewer with optional MP4 export.

    stride: frames to skip when exporting MP4 (1 = every frame, 10 = every 10th frame).
    position_name, scan_id, mask_n: used in the MP4 filename when mp4_export=True.
    If None, they default to workflow_config POSITION_NAME, SAMPLE_ID, MASK_N.
    base_dir: base directory for finding raw data. If None, uses BASE_DIR from config.
    require_processed: if True, requires processed results file (for ROI map). If False,
                      uses raw-only loader (works with mask_n="peak").
    """
    scan = scan_id if scan_id is not None else SAMPLE_ID
    if base_dir is None:
        base_dir = _resolve_base_dir(scan)
    else:
        base_dir = Path(base_dir)

    m = mask_n if mask_n is not None else MASK_N
    
    # For "peak" mode, we don't need processed results - use raw-only loader
    if require_processed:
        run = load_run_data(base_dir, scan, mask_n=m)
    else:
        try:
            # Try to load with processed results first (for ROI map if needed)
            run = load_run_data(base_dir, scan, mask_n=m)
        except FileNotFoundError:
            # If processed results not found, use raw-only loader (works for mask_n="peak")
            run = load_raw_data_only(base_dir, scan)

    pos = position_name if position_name is not None else POSITION_NAME
    export_path = str(FIGURES_DIR / "movies" / pos / scan / f"{pos}_{scan}_crop{crop_size}_stride{stride}.mp4")

    launch_masked_raw_viewer(
        run,
        mask_n="peak",
        start_frame=0,
        clip_percentile_init=99.9,
        crop_size=crop_size,
        mp4_export=True,
        export_path=export_path,
        frame_skip=int(stride),
    )

    run.close()

def raw_mask_oscillation_inspector():

    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    inspect_raw_mask_oscillations(
        run,
        mask_signal=MASK_N,
        mask_control=CONTROL_MASK_N,
        dt_s=1.0,
        fmin=1 / 1000,
        fmax=1 / 10,
        detrend=True,
        window=True,
    )

    run.close()


def comparison_of_corr_and_g_ttc_plot_methods():
    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    out = compare_ttc_methods_from_raw(
        run,
        mask_n=MASK_N,
        frame_slice=slice(0, 4800),  # or None for all frames
        clip_hi_percentile=99.9,
    )

    run.close()

def compare_existing_vs_corr_entrypoint():

    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)
    try:
        return compare_existing_processed_ttc_with_corr_from_raw(
            run,
            mask_n=MASK_N,
            frame_slice=slice(0, 4800),  # or None
            clip_hi_percentile=99.9,
            diff_symmetric=True,
        )
    finally:
        run.close()

def compare_existing_ttc_and_cgpt_ttc_from_raw():

    return exec_compare_raw_vs_processed_ttc(
        base_dir=BASE_DIR,
        sample_id="A073",
        mask_n=MASK_N,
        out_path=FIGURES_DIR / "misc" / "A073_M146_raw_vs_processed_vs_diff.png",
        clip_hi_percentile=99.9,
    )

def compare_existing_ttc_and_ttc_from_raw():

    return exec_compare_loaded_vs_twotime_imported_ttc(
        base_dir=BASE_DIR,
        sample_id="A073",
        mask_n=MASK_N,
        frame_slice=slice(0, 4800),
        stride=1,
        clip_hi_percentile=99.9,
    )

def ttc_with_custom_mask():

    return exec_mask_and_twotime_ttc_custom_ring(
        base_dir=BASE_DIR,
        sample_id=SAMPLE_ID,
        mask_n_for_loading=MASK_N,      # only used to load run/scattering/results paths
        r_inner_px=160.0,
        r_outer_px=170.0,
        center_px=(1198, 216),  # (cx, cy) or None to auto-detect
        bright_percentile=99.9,
        shape='semi',  # "semi" or "circle"
        fill='ring',  # "ring" or "solid"
        frame_slice=slice(0, 2000),     # IMPORTANT: start small to avoid OOM
        stride=1,
        do_pixel_smooth=True,
        clip_hi_percentile=99.9,
    )




# ---- analysis (APS 08-IDE) ----
def print_h5_structure(name, obj):
    print(name)

def explore(name, obj):
    indent = "  " * (name.count("/") - 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}{name}/  (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}  (Dataset)  shape={obj.shape}, dtype={obj.dtype}")

def h5_file_inspector(filename):
    with h5py.File(filename, "r") as f:
        f.visititems(explore)

def g2_plotter(filename):
    with h5py.File(filename, "r") as f:
        g2 = f["xpcs/twotime/normalized_g2"][...]

    x = np.arange(len(g2[:, 100]))

    plt.figure()
    # plt.errorbar(delay[0, :], G2_result, yerr=G2_error, fmt='none', ecolor='b', capsize=2)
    plt.semilogx(x, g2[:, 194], 'b.')
    plt.title('g2 autocorrelation with pixelwise normalisation')
    plt.ylabel('g2(q,tau)')
    plt.xlabel('Delay Time, tau (s)')
    # plt.ylim([0, 1.5])

    plt.show()

def ttc_plotter(filename):
    with h5py.File(filename, "r") as f:
        C = f["xpcs/twotime/correlation_map/c2_00194"][...]
    C = C + C.T - np.diag(np.diag(C))
    # renomalize C
    C = C - np.min(C)
    C = C / np.max(C)
    # C_extent = [0, frameSpacing * det.shape[0], 0, frameSpacing * det.shape[0]]
    # C_extent = [0, det.shape[0], 0, det.shape[0]]

    plt.rcParams.update({'font.size': 24})

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_ylabel('t2 (s)')
    ax.set_xlabel('t1 (s)')
    ax.set_ylabel('Frame number')
    ax.set_xlabel('Frame number')
    # ax.set_title(f'Ring {0}')
    # ax.set_title('Whole Mask')
    im = ax.imshow(C, origin="lower", cmap='plasma')
    cbar = fig.colorbar(im, ax=ax)
    custom_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(custom_ticks)

    # diag_vals = np.diag(C)
    # x_min, x_max, y_min, y_max = C_extent
    # n = C.shape[0]  # assuming square matrix
    # t_axis = np.linspace(x_min, x_max, n)
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(np.arange(0, len(t_axis), 1), 1 - diag_vals, marker='.', linestyle='-', alpha=0.8, color='C1')
    # plt.xlabel('t (s)')
    # plt.ylabel('C(t,t)')
    # plt.title('Diagonal of C (t1 = t2)')
    # plt.grid(True)
    # plt.tight_layout()

    plt.show()

def intensity_vs_time(filename):
    with h5py.File(filename, "r") as f:
        data = f["xpcs/spatial_mean/intensity_vs_time"][...]

    print(data.shape)

    plt.figure()
    plt.plot(data[0, :], data[1, :])
    plt.show()

def static_vs_dynamic_bins(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]

    scattering_2d_reshape = scattering_2d[0, :, :]

    individual_mask_intensity = []


    print(dynamic_roi_map.shape)
    print(scattering_2d.shape)
    print(scattering_2d_reshape.shape)

    for i in range(300):
        individual_mask = dynamic_roi_map.copy()
        individual_mask[individual_mask != i+1] = 0
        individual_mask[individual_mask != 0] = 1
        scattering_2d_masked = scattering_2d_reshape * individual_mask
        individual_mask_intensity.append(np.sum(scattering_2d_masked))

    # individual_mask = dynamic_roi_map.copy()
    # individual_mask[individual_mask != 100] = 0
    # individual_mask[individual_mask != 0] = 1
    # scattering_2d_masked = scattering_2d_reshape * individual_mask
    # individual_mask_intensity.append(np.sum(scattering_2d_masked))
    #
    # individual_mask_intensity = np.array(individual_mask_intensity)

    plt.figure()
    # plt.plot(individual_mask_intensity)
    plt.semilogy(np.arange(1, 301, 1), individual_mask_intensity)
    plt.xlabel('mask number')
    plt.ylabel('integrated intensity')


    # plt.figure()
    # plt.imshow(individual_mask)

    plt.figure()
    plt.imshow(dynamic_roi_map)

    plt.figure()
    plt.imshow(individual_mask)

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    I = scattering_2d_reshape.astype(float)

    plt.imshow(I,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=np.max(I)),
        interpolation="nearest",
    )
    # ax0.set_facecolor("black")

    # np.savetxt('dynamic_roi_map.txt', dynamic_roi_map)
    # print(np.shape(dynamic_roi_map))


    plt.show()

def combined_plot(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    run_name = os.path.basename(h5_file).split("_")[0]


    scattering_2d_reshape = scattering_2d[0, :, :]
    individual_mask_intensity = []

    print('q:', q)

    print('phi:', phi)

    for index in np.arange(0, 300, 1):
        # print('index:', index, ', q:', q[10-int(np.floor(index/10))], ', phi:', phi[int(np.floor(index/30))])
        print('index:', index, ', x:', int((index // 30)), ', q:', q[int((index // 30))],
              ', y:', int(index % 30), ', phi:', phi[int(index % 30)])

    individual_mask = dynamic_roi_map.copy()
    # individual_mask[(individual_mask != 165) & (individual_mask != 225)] = 0
    individual_mask[individual_mask != 135] = 0
    individual_mask[individual_mask != 0] = 1

    for i in range(300):
        individual_mask = dynamic_roi_map.copy()
        individual_mask[individual_mask != i] = 0
        individual_mask[individual_mask != 0] = 1
        scattering_2d_masked = scattering_2d_reshape * individual_mask
        individual_mask_intensity.append(np.sum(scattering_2d_masked))

    i = np.argmax(individual_mask_intensity)
    # idxs = [0, 1, -1, -31, -30, -29, 29, 30, 31] + i
    idxs = [-29, 1, 31, -30, 0, 30, -31, -1, 29] + i

    masks = dynamic_roi_map.copy()
    combined_mask = np.isin(masks, idxs).astype(int)

    im = dynamic_roi_map.copy()
    im[~np.isin(im, idxs)] = 0

    plt.figure()
    plt.imshow(im)

    plt.figure()
    x = np.arange(len(g2[:, 100]))
    # plt.errorbar(delay[0, :], G2_result, yerr=G2_error, fmt='none', ecolor='b', capsize=2)
    for i in idxs:
        plt.semilogx(x, g2[:, i-1], label='M' + str(i) + ', q='+f"{q[int((i // 30))]:.3f}"
                                        + ',  phi='+f"{phi[int(i % 30)]:.3f}")
    plt.title('g2 autocorrelation for experiment ' + run_name)
    plt.ylabel('g2(q,tau)')
    plt.xlabel('Delay Time, tau')
    plt.legend()

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for i, ax in enumerate(axes.flat):
        path = f"xpcs/twotime/correlation_map/c2_00{idxs[i]:03d}"
        with h5py.File(filename, "r") as f:
            C = f[path][...]
        C = C + C.T - np.diag(np.diag(C))
        # renomalize C
        # C = C - np.min(C)
        # C = C / np.max(C)
        lo, hi = np.percentile(C, [0, 99.9])
        C = np.clip(C, lo, hi)
        # C = (C_clip - lo) / (hi - lo)
        # ax.set_title(f"M{idxs[i]}")
        ax.axis("off")
        # ax.set_ylabel('Frame number')
        # ax.set_xlabel('Frame number')
        im = ax.imshow(C, origin="lower", cmap='plasma')
        label = (
            f"M{idxs[i]}\n"
            f"min {np.min(C):.2f}\n"
            f"max {np.max(C):.2f}"
        )
        ax.text(
            0.05, 0.95, label,
            transform=ax.transAxes,  # axes-relative coordinates
            ha="left", va="top",
            fontsize=12,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="black",
                alpha=0.6,
                edgecolor="none"
            )
        )
        print(path)

    plt.tight_layout()

    plt.figure()

    I = scattering_2d_reshape.astype(float)
    I[combined_mask == 1] *= 10

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")  # or "navy", etc.
    cmap.set_bad("black")  # for NaN/inf

    ys, xs = np.where(combined_mask == 1)
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    half = 200
    ymin = max(cy - half, 0)
    ymax = min(cy + half, I.shape[0])
    xmin = max(cx - half, 0)
    xmax = min(cx + half, I.shape[1])
    img_crop = I[ymin:ymax, xmin:xmax]
    mask_crop = combined_mask[ymin:ymax, xmin:xmax]

    plt.imshow(img_crop, origin="lower", cmap=cmap, norm=LogNorm(vmin=0.1, vmax=I.max()))
    plt.colorbar()

    # plt.figure()
    # plt.imshow(combined_mask)

    print(run_name)
    print(filename)

    plt.show()


def oauth_test():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = str(CREDS_PATH)

    def get_creds():
        creds = None
        if Path("token.json").exists():
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials, SCOPES
                )
                creds = flow.run_local_server(port=0)
            Path("token.json").write_text(creds.to_json())

        return creds

    creds = get_creds()
    gc = gspread.authorize(creds)

    # Paste your spreadsheet ID here
    sh = gc.open_by_key("1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ")
    print("Opened spreadsheet:", sh.title)


def image_upload(fig, target_cell="AF142", upload_name="matplotlib_output.png",
                 tab_name="IPA NBH",
                 spreadsheet_id="1OAA7H4I3cgas32aSZkrLB8TOKHymMAv2uk_0eTywWcQ",
                 token_path="token.json",
                 creds_path="client_secret_180145739842-0ug37lsh4qltki62e8te8bqkde9u25jb.apps.googleusercontent.com.json"):

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    def get_creds():
        creds = None
        if Path(token_path).exists():
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            Path(token_path).write_text(creds.to_json())
        return creds

    creds = get_creds()

    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(tab_name)

    cols, rows = find_rows_with_position(ws, "A5")

    print(cols)

    for cell in rows_to_cells(rows, "AF"):
        print(cell)
        # image_upload(fig, target_cell=cell, upload_name=f"A5_{cell}.png")


    # drive = build("drive", "v3", credentials=creds)
    #
    # buf = BytesIO()
    # try:
    #     fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    #     buf.seek(0)
    #
    #     media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=False)
    #     created = drive.files().create(
    #         body={"name": upload_name},
    #         media_body=media,
    #         fields="id"
    #     ).execute()
    #
    #     file_id = created["id"]
    #
    #     drive.permissions().create(
    #         fileId=file_id,
    #         body={"type": "anyone", "role": "reader"},
    #     ).execute()
    #
    #     image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
    #     formula = f'=IMAGE("{image_url}", 4, 180, 320)'
    #
    #     ws.update(target_cell, [[formula]], value_input_option="USER_ENTERED")
    #     print(f"Inserted image into {ws.title} cell {target_cell}")
    #
    # finally:
    #     buf.close()

def figure_upload():

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 4, 2])
    image_upload(fig, target_cell="AF142", upload_name="run123_overview.png")
    plt.close(fig)

def q_spacing_inspector(filename):
    with h5py.File(filename, "r") as f:
        dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        g2 = f["xpcs/twotime/normalized_g2"][...]
        q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]

    run_name = os.path.basename(h5_file).split("_")[0]


    scattering_2d_reshape = scattering_2d[0, :, :]
    individual_mask_intensity = []

    print('q:', q)

    print('phi:', phi)

    for i in range(9):
        print(q[i + 1] - q[i])

    for j in range(29):
        print(phi[j + 1] - phi[j])

def integrated_intensities_inspector(filename):

    with h5py.File(filename, "r") as f:
        # dynamic_roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        # scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        # ttc = f["xpcs/twotime/correlation_map/c2_00194"][...]
        # g2 = f["xpcs/twotime/normalized_g2"][...]
        # q = f["xpcs/qmap/dynamic_v_list_dim0"][...]
        # phi = f["xpcs/qmap/dynamic_v_list_dim1"][...]
        integrated_intensities = f["xpcs/temporal_mean/scattering_1d"][...]
        integrated_intensities_segments = f["xpcs/temporal_mean/scattering_1d_segments"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]


        print(np.shape(integrated_intensities))
        print(np.shape(integrated_intensities_segments))
        print(integrated_intensities)

        np.savetxt("integrated_intensities.txt", integrated_intensities)
        np.savetxt("integrated_intensities_segments.txt", integrated_intensities_segments)

        print("q:", np.shape(q), q)
        print("phi:", np.shape(phi), phi)

        plt.figure()
        plt.plot(integrated_intensities[0])

        plt.figure()
        plt.plot(integrated_intensities_segments[0])
        plt.plot(integrated_intensities_segments[1])
        plt.plot(integrated_intensities_segments[2])


        # plt.ylim([0, 1])
        plt.show()

def bragg_peak_centroid_and_skew_qphi(
    I_mean_qphi: np.ndarray,
    q: np.ndarray,
    phi_deg: np.ndarray,
    *,
    eps: float = 1e-12,
    roi_mask_qphi: np.ndarray | None = None,  # optional (nq,nphi) boolean ROI
) -> dict:
    """
    Intensity-weighted centroid and skewness in q and phi.

    Notes
    -----
    - q: linear stats
    - phi: circular mean, then skewness of wrapped residuals dphi in [-180,180)
    """
    I = np.asarray(I_mean_qphi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    phi_deg = np.asarray(phi_deg, dtype=np.float64)

    nq, nphi = I.shape
    if q.size != nq or phi_deg.size != nphi:
        raise ValueError(f"Shape mismatch: I={I.shape}, q={q.shape}, phi={phi_deg.shape}")

    if roi_mask_qphi is None:
        vv = np.isfinite(I) & (I > 0)
    else:
        vv = np.asarray(roi_mask_qphi, dtype=bool) & np.isfinite(I) & (I > 0)

    if not np.any(vv):
        raise RuntimeError("No valid pixels for centroid/skewness.")

    w = I[vv]
    sw = float(np.sum(w))
    if sw <= eps:
        raise RuntimeError("Sum of weights is zero.")

    # Build coordinate grids for the valid points
    iq, ip = np.nonzero(vv)
    qv = q[iq]                 # (N,)
    ph_deg = phi_deg[ip]       # (N,)

    # ---------- q mean + skew ----------
    q_mean = float(np.sum(w * qv) / sw)
    dq = qv - q_mean
    mu2_q = float(np.sum(w * dq * dq) / sw)
    mu3_q = float(np.sum(w * dq * dq * dq) / sw)
    sigma_q = float(np.sqrt(max(mu2_q, 0.0)))
    skew_q = float(mu3_q / (mu2_q ** 1.5 + eps))

    # ---------- phi circular mean ----------
    ph_rad = np.deg2rad(ph_deg)
    c = float(np.sum(w * np.cos(ph_rad)) / sw)
    s = float(np.sum(w * np.sin(ph_rad)) / sw)
    phi_mean_rad = float(np.arctan2(s, c))
    phi_mean_deg = float(np.rad2deg(phi_mean_rad))

    # ---------- phi skewness on wrapped residuals ----------
    dphi_deg = ph_deg - phi_mean_deg
    dphi_deg = (dphi_deg + 180.0) % 360.0 - 180.0  # wrap to [-180,180)

    mu2_phi = float(np.sum(w * dphi_deg * dphi_deg) / sw)
    mu3_phi = float(np.sum(w * dphi_deg * dphi_deg * dphi_deg) / sw)
    sigma_phi_deg = float(np.sqrt(max(mu2_phi, 0.0)))
    skew_phi = float(mu3_phi / (mu2_phi ** 1.5 + eps))

    return {
        "q_mean": q_mean,
        "phi_mean_deg": phi_mean_deg,
        "sigma_q": sigma_q,
        "sigma_phi_deg": sigma_phi_deg,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "n_points": int(w.size),
    }

def integrated_intensities_plot(
    h5_file: str | Path,
    *,
    phi_fast_axis: bool = True,
    map_scale: str = "log",          # "linear" or "log"
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.8,
    relstd_vmax: float | None = None,
):
    """
    Plots scattering_1d (mean) and scattering_1d_segments (10 time segments).

    Flattening assumption:
      If phi_fast_axis=True (default):
          flat index = iq * nphi + iphi  -> reshape (nq, nphi)
      If phi_fast_axis=False:
          flat index = iphi * nq + iq    -> reshape (nphi, nq) then transpose to (nq, nphi)
    """
    h5_file = str(h5_file)

    import numpy as np
    import h5py
    import matplotlib.pyplot as plt

    with h5py.File(h5_file, "r") as f:
        I1d = f["xpcs/temporal_mean/scattering_1d"][...]
        Iseg = f["xpcs/temporal_mean/scattering_1d_segments"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]

    I1d = np.asarray(I1d)
    Iseg = np.asarray(Iseg)
    q = np.asarray(q)
    phi = np.asarray(phi)

    if I1d.ndim != 2 or I1d.shape[0] != 1:
        raise ValueError(f"Expected scattering_1d shape (1, 3600), got {I1d.shape}")
    if Iseg.ndim != 2 or Iseg.shape[1] != I1d.shape[1]:
        raise ValueError(f"Expected scattering_1d_segments shape (10, 3600), got {Iseg.shape}")

    nq = int(q.size)
    nphi = int(phi.size)
    if nq * nphi != int(I1d.shape[1]):
        raise ValueError(
            f"q.size * phi.size = {nq}*{nphi}={nq*nphi} does not match scattering_1d length {I1d.shape[1]}"
        )

    # ---- reshape to (q, phi) ----
    if phi_fast_axis:
        I_mean_qphi = I1d[0].reshape(nq, nphi)
        I_seg_qphi = Iseg.reshape(Iseg.shape[0], nq, nphi)  # (nseg, nq, nphi)
    else:
        # flat index = iphi*nq + iq
        I_mean_phiq = I1d[0].reshape(nphi, nq)
        I_seg_phiq = Iseg.reshape(Iseg.shape[0], nphi, nq)
        I_mean_qphi = np.transpose(I_mean_phiq, (1, 0))
        I_seg_qphi = np.transpose(I_seg_phiq, (0, 2, 1))

    # ----------------------------
    # Bragg peak center in (q, phi)
    # ----------------------------
    eps = 1e-12

    # weights, clip negatives just in case
    W = np.clip(I_mean_qphi, 0.0, None)
    sw = float(np.sum(W))

    if not np.isfinite(sw) or sw <= eps:
        raise RuntimeError("No positive intensity in I_mean_qphi to compute Bragg center.")

    # (1) argmax center
    iq_max, iphi_max = np.unravel_index(int(np.argmax(I_mean_qphi)), I_mean_qphi.shape)
    q_max = float(q[iq_max])
    phi_max = float(phi[iphi_max])  # degrees

    # (2) intensity-weighted centroid
    # weighted mean in q
    q_mean = float(np.sum(W * q[:, None]) / sw)

    # weighted circular mean in phi (degrees -> radians for trig)
    phi_rad = np.deg2rad(phi.astype(np.float64))
    c = float(np.sum(W * np.cos(phi_rad)[None, :]) / sw)
    s = float(np.sum(W * np.sin(phi_rad)[None, :]) / sw)
    phi_mean_rad = float(np.arctan2(s, c))
    phi_mean_deg = float((np.rad2deg(phi_mean_rad) + 180.0) % 360.0 - 180.0)  # wrap to [-180, 180)

    # nearest bin to centroid (useful for overlay and sanity)
    iq_c = int(np.argmin(np.abs(q - q_mean)))
    # circular distance in degrees for nearest phi
    dphi = (phi - phi_mean_deg + 180.0) % 360.0 - 180.0
    iphi_c = int(np.argmin(np.abs(dphi)))
    q_cent_bin = float(q[iq_c])
    phi_cent_bin = float(phi[iphi_c])

    print("Bragg peak center estimates from scattering_1d:")
    print(f"  argmax bin: iq={iq_max}, iphi={iphi_max}, q={q_max:.6f}, phi={phi_max:.3f} deg")
    print(f"  centroid:   q_mean={q_mean:.6f}, phi_mean={phi_mean_deg:.3f} deg")
    print(f"  nearest bin to centroid: iq={iq_c}, iphi={iphi_c}, q={q_cent_bin:.6f}, phi={phi_cent_bin:.3f} deg")

    metrics = bragg_peak_centroid_and_skew_qphi(I_mean_qphi, q, phi)
    print("Bragg peak centroid/skew from scattering_1d:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # representative phi index: closest to 0°
    iphi0 = int(np.argmin(np.abs(phi - 0.0)))

    # ---- summaries ----
    Iseg_q = I_seg_qphi.mean(axis=2)          # (nseg, nq)
    Imean_q = I_mean_qphi.mean(axis=1)        # (nq,)
    I_std_qphi = I_seg_qphi.std(axis=0)       # (nq, nphi)
    I_relstd_qphi = I_std_qphi / np.maximum(I_mean_qphi, 1e-12)

    # ---- display transforms + robust limits ----
    def _disp(arr: np.ndarray) -> np.ndarray:
        if map_scale.lower() == "log":
            return np.log10(np.clip(arr, 0.0, None) + 1e-12)
        return arr

    mean_disp = _disp(I_mean_qphi)
    rel_disp = I_relstd_qphi  # keep linear (already a ratio)

    # robust vmin/vmax for the mean map
    finite_mean = mean_disp[np.isfinite(mean_disp)]
    vmin = float(np.percentile(finite_mean, vmin_pct))
    vmax = float(np.percentile(finite_mean, vmax_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(finite_mean)), float(np.nanmax(finite_mean))

    # robust vmax for relstd
    finite_rel = rel_disp[np.isfinite(rel_disp)]
    if relstd_vmax is None:
        rel_vmax = float(np.percentile(finite_rel, 99.5))
    else:
        rel_vmax = float(relstd_vmax)

    # ---- plotting ----
    fig = plt.figure(figsize=(14.5, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.28)

    # (A) Mean map in (q, phi)
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        mean_disp,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
        vmin=vmin,
        vmax=vmax,
    )
    ax0.set_title(f"Mean scattering_1d → (q, φ)  [{map_scale}]")
    ax0.set_xlabel("φ (deg)")
    ax0.set_ylabel("q (Å$^{-1}$)")
    cblabel0 = "log10(Intensity + eps)" if map_scale.lower() == "log" else "Intensity (a.u.)"
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03, label=cblabel0)

    # (B) Segment evolution as (segment index, q) for φ≈0 slice
    ax1 = fig.add_subplot(gs[0, 1])
    seg_vs_q_phi0 = _disp(I_seg_qphi[:, :, iphi0])  # display same scaling
    finite_seg = seg_vs_q_phi0[np.isfinite(seg_vs_q_phi0)]
    svmin = float(np.percentile(finite_seg, vmin_pct))
    svmax = float(np.percentile(finite_seg, vmax_pct))
    im1 = ax1.imshow(
        seg_vs_q_phi0,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[q.min(), q.max(), 0, seg_vs_q_phi0.shape[0] - 1],
        vmin=svmin,
        vmax=svmax,
    )
    ax1.set_title(f"Segments vs q at φ≈{phi[iphi0]:.3f}° (closest to 0°)")
    ax1.set_xlabel("q (Å$^{-1}$)")
    ax1.set_ylabel("segment index (0..9)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, label=cblabel0)

    # (C) φ-averaged intensity vs q for each segment + mean
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(q, Imean_q, lw=2.5, label="mean (φ-avg)")
    for s in range(Iseg_q.shape[0]):
        ax2.plot(q, Iseg_q[s], lw=1.2, alpha=0.8, label=f"seg {s}" if s < 4 else None)
    ax2.set_title("φ-averaged intensity vs q (all segments)")
    ax2.set_xlabel("q (Å$^{-1}$)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=9)

    # (D) Relative variability map (std/mean) in (q, phi)
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(
        rel_disp,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[phi.min(), phi.max(), q.min(), q.max()],
        vmin=0.0,
        vmax=rel_vmax,
    )
    ax3.set_title("Temporal variability: std(seg)/mean  (q, φ)")
    ax3.set_xlabel("φ (deg)")
    ax3.set_ylabel("q (Å$^{-1}$)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03, label="Relative std")

    fig.suptitle(f"Integrated intensity diagnostics\n{h5_file}", y=0.98, fontsize=12)
    # plt.show()


def find_bragg_peak_center_from_scattering_2d_with_overlay_function(
    h5_file: str | Path,
    *,
    dataset_key: str = "xpcs/temporal_mean/scattering_2d",
    use_first_frame_if_3d: bool = True,
    smooth_sigma_px: float | None = 1.0,
    bright_percentile: float = 99.7,
    weight_mode: str = "log",  # "linear" | "sqrt" | "log"
    figsize: tuple[float, float] = (7.2, 6.4),
) -> tuple[tuple[float, float], dict]:
    """
    Option 1 (robust bright-region centroid) + ALWAYS makes an overlay plot.

    Steps
    -----
    1) Load scattering_2d (use first frame if it's (1,H,W))
    2) Light smoothing (optional)
    3) Threshold at bright_percentile to define a "bright region"
    4) Compute weighted centroid of that region
    5) Plot: scattering_2d with mask outline + centroid marker

    Returns
    -------
    (cy, cx) : (float, float)
        Estimated centre (row, col) in pixel coordinates.
    info : dict
        Debug info: threshold, n_pixels_used, etc.
    """
    h5_file = Path(h5_file)

    with h5py.File(h5_file, "r") as f:
        scat = f[dataset_key][...]

    scat = np.asarray(scat)
    if scat.ndim == 3 and use_first_frame_if_3d:
        scat = scat[0, :, :]
    if scat.ndim != 2:
        raise ValueError(f"{dataset_key} must be 2D (or 3D with first-frame), got {scat.shape}")

    img = scat.astype(np.float64, copy=False)

    # Optional smoothing (very light) to stabilize centroid under hot pixels
    if smooth_sigma_px is not None and smooth_sigma_px > 0:
        try:
            from scipy.ndimage import gaussian_filter
            img_s = gaussian_filter(img, sigma=float(smooth_sigma_px))
        except Exception:
            img_s = img
    else:
        img_s = img

    # Bright-region mask
    p = float(np.clip(bright_percentile, 0.0, 100.0))
    thr = float(np.nanpercentile(img_s, p))
    mask = np.isfinite(img_s) & (img_s >= thr)
    n = int(mask.sum())

    # If too few pixels, relax threshold slightly
    if n < 10:
        p2 = max(90.0, p - 5.0)
        thr = float(np.nanpercentile(img_s, p2))
        mask = np.isfinite(img_s) & (img_s >= thr)
        n = int(mask.sum())

    if n < 10:
        raise ValueError(
            f"Bright-region mask too small (n={n}). "
            f"Lower bright_percentile (currently {bright_percentile})."
        )

    yy, xx = np.nonzero(mask)
    vals = img_s[yy, xx]

    # Weights (to reduce dominance of extreme skew / hot pixels)
    if weight_mode == "linear":
        w = np.clip(vals, 0.0, np.inf)
    elif weight_mode == "sqrt":
        w = np.sqrt(np.clip(vals, 0.0, np.inf))
    elif weight_mode == "log":
        w = np.log1p(np.clip(vals, 0.0, np.inf))
    else:
        raise ValueError("weight_mode must be one of: 'linear', 'sqrt', 'log'")

    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError("Non-positive or non-finite weight sum, cannot compute centroid")

    cy = float(np.sum(yy * w) / wsum)
    cx = float(np.sum(xx * w) / wsum)

    info = {
        "h5_file": str(h5_file),
        "dataset_key": dataset_key,
        "img_shape": tuple(img.shape),
        "smooth_sigma_px": smooth_sigma_px,
        "bright_percentile": float(bright_percentile),
        "threshold_value": float(thr),
        "n_pixels_used": n,
        "weight_mode": weight_mode,
        "centroid_cy_cx": (cy, cx),
    }


    # Overlay plot (always)

    # Use log1p for display to handle heavy skew safely (works with zeros)
    disp = np.log1p(np.clip(img, 0.0, np.inf))

    # Robust display limits
    vmin = float(np.nanpercentile(disp, 1.0))
    vmax = float(np.nanpercentile(disp, 99.8))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = None, None

    img = np.asarray(disp, dtype=np.float64)
    img = np.where(img > 0, img, np.nan)

    vmin = np.nanpercentile(img, 1.0)
    vmax = np.nanpercentile(img, 99.999)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        img,
        origin="upper",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    # Mask outline
    ax.contour(mask.astype(np.float32), levels=[0.5], linewidths=1.2, colors="cyan")

    # Centroid marker
    ax.plot(cx, cy, marker="x", markersize=10, mew=2.2)

    ax.set_title(
        f"Bragg peak centre (bright-region centroid)\n"
        f"p={bright_percentile:.2f}, n={n}, weight={weight_mode}, σ={smooth_sigma_px}"
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("log1p(scattering_2d)")

    plt.tight_layout()
    plt.show()

    return (cy, cx), info

def execute_find_bragg_peak_center_from_scattering_2d_with_overlay():

    (cy, cx), info = find_bragg_peak_center_from_scattering_2d_with_overlay_function(
        filename,
        bright_percentile=99.7,
        smooth_sigma_px=1.0,
        weight_mode="log",
    )

    print(f"Bragg peak centre: cy={cy:.2f}, cx={cx:.2f}")
    print(info)

def _ensure_2d(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D (or (1,H,W)) scattering_2d, got shape {img.shape}")
    return img


def _match_axes(img2d, q_vals, phi_vals):
    """
    Returns (img, q, phi) such that:
      img.shape == (len(phi), len(q))  i.e. axis0=phi, axis1=q
    If user provided transposed vectors, we transpose img.
    """
    q_vals = np.asarray(q_vals).ravel()
    phi_vals = np.asarray(phi_vals).ravel()

    H, W = img2d.shape
    if (H, W) == (phi_vals.size, q_vals.size):
        return img2d, q_vals, phi_vals
    if (H, W) == (q_vals.size, phi_vals.size):
        return img2d.T, q_vals, phi_vals

    raise ValueError(
        f"Shape mismatch: img={img2d.shape}, len(phi)={phi_vals.size}, len(q)={q_vals.size}. "
        "Expected img=(len(phi),len(q)) or transposed."
    )


def _winsorize(x, p_hi=99.9):
    x = np.asarray(x, dtype=np.float64)
    hi = np.nanpercentile(x, float(p_hi))
    lo = np.nanpercentile(x, 0.0)
    return np.clip(x, lo, hi), float(lo), float(hi)


def _local_median_3x3(img):
    """
    Fast 3x3 local median using shifted stacks (pure numpy, no scipy).
    Edge-handled by padding with edge values.
    """
    a = np.asarray(img, dtype=np.float64)
    p = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    shifts = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifts.append(p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]])
    stack = np.stack(shifts, axis=0)  # (9,H,W)
    return np.median(stack, axis=0)


def _despike_hot_pixels(img, *, z_thresh=12.0, use_log=True):
    """
    Replace extreme outliers using a robust z-score on (optionally) log1p(img).
    Replacement value is local 3x3 median.
    """
    x = np.asarray(img, dtype=np.float64)
    x0 = np.clip(x, 0.0, np.inf)

    y = np.log1p(x0) if use_log else x0
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad == 0:
        return x0, np.zeros_like(x0, dtype=bool)

    # 1.4826 * MAD ~ sigma for normal, good robust scale
    z = (y - med) / (1.4826 * mad)
    hot = z > float(z_thresh)

    if np.any(hot):
        local_med = _local_median_3x3(x0)
        x0 = x0.copy()
        x0[hot] = local_med[hot]

    return x0, hot


def _weighted_quantile(x, w, qs):
    """
    Weighted quantile(s) of x with weights w.
    qs in [0,1] list/array.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    qs = np.asarray(qs, dtype=np.float64).ravel()

    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[m]
    w = w[m]
    if x.size == 0 or np.sum(w) <= 0:
        return np.full(qs.shape, np.nan, dtype=np.float64)

    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    return np.interp(qs, cdf, x)


def _weighted_moments_1d(x, w):
    """
    Returns (mu, sigma, skew) for weighted distribution.
    skew = E[(x-mu)^3] / sigma^3
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[m]
    w = w[m]
    W = np.sum(w)
    if x.size == 0 or W <= 0:
        return np.nan, np.nan, np.nan

    mu = np.sum(w * x) / W
    m2 = np.sum(w * (x - mu) ** 2) / W
    sigma = np.sqrt(max(m2, 0.0))
    if not np.isfinite(sigma) or sigma == 0:
        return float(mu), float(sigma), np.nan

    m3 = np.sum(w * (x - mu) ** 3) / W
    skew = m3 / (sigma ** 3)
    return float(mu), float(sigma), float(skew)


def bragg_peak_shape_metrics_fixed_q_phi(
    scattering_2d,
    *,
    q_vals,
    phi_vals,
    winsor_p_hi=99.9,
    use_log_weights=True,
    despike=True,
    hot_z_thresh=12.0,
):
    """
    Compute fixed-axis (q and phi) statistics for the Bragg peak shape.

    Returns a dict containing:
      - center_q, center_phi (weighted means)
      - sigma_q, sigma_phi  (weighted std dev)
      - skew_q,  skew_phi   (weighted moment skewness)
      - q_profile, phi_profile (marginals)
      - quantile skew (Bowley) in each axis
      - optional hot-pixel mask + thresholds

    Notes
    -----
    - Tails are included (no masking), but hot pixels are optionally despiked.
    - Robustness is controlled by winsor_p_hi and/or use_log_weights.
    - Axis convention enforced: img shape = (len(phi), len(q)).
    """
    img = _ensure_2d(scattering_2d)
    img, q, phi = _match_axes(img, q_vals, phi_vals)

    x = np.asarray(img, dtype=np.float64)
    x = np.clip(x, 0.0, np.inf)

    hot_mask = None
    if despike:
        x, hot_mask = _despike_hot_pixels(x, z_thresh=float(hot_z_thresh), use_log=True)

    # Winsor cap to prevent a few extreme pixels dominating moments
    xw, lo, hi = _winsorize(x, p_hi=float(winsor_p_hi))

    # Weights (optionally log-compressed)
    w = np.log1p(xw) if bool(use_log_weights) else xw
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, np.inf)

    # Marginals: axis0=phi, axis1=q
    Wq = np.sum(w, axis=0)   # (nq,)
    Wphi = np.sum(w, axis=1) # (nphi,)

    # Fixed-axis moments
    mu_q, sig_q, skew_q = _weighted_moments_1d(q, Wq)
    mu_phi, sig_phi, skew_phi = _weighted_moments_1d(phi, Wphi)

    # Quantile (Bowley) skewness for stability check
    q25, q50, q75 = _weighted_quantile(q, Wq, [0.25, 0.50, 0.75])
    p25, p50, p75 = _weighted_quantile(phi, Wphi, [0.25, 0.50, 0.75])

    def _bowley(a25, a50, a75):
        den = (a75 - a25)
        if not np.isfinite(den) or den == 0:
            return np.nan
        return float((a75 + a25 - 2.0 * a50) / den)

    bowley_q = _bowley(q25, q50, q75)
    bowley_phi = _bowley(p25, p50, p75)

    # Effective number of pixels (for “how concentrated are the weights”)
    wflat = w.ravel()
    sw = np.sum(wflat)
    sw2 = np.sum(wflat * wflat)
    neff = (sw * sw / sw2) if (sw2 > 0) else np.nan

    return {
        "center_q": mu_q,
        "center_phi": mu_phi,
        "sigma_q": sig_q,
        "sigma_phi": sig_phi,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "bowley_skew_q": bowley_q,
        "bowley_skew_phi": bowley_phi,
        "q_profile": Wq,
        "phi_profile": Wphi,
        "q_vals": q,
        "phi_vals": phi,
        "winsor_lo": lo,
        "winsor_hi": hi,
        "use_log_weights": bool(use_log_weights),
        "despike": bool(despike),
        "hot_pixel_mask": hot_mask,
        "neff": float(neff) if np.isfinite(neff) else np.nan,
    }


def make_phi_band_mask(
    phi_map: np.ndarray,
    *,
    phi0: float,
    dphi: float,
    valid: np.ndarray | None = None,
    q_map: np.ndarray | None = None,
    qmin: float | None = None,
    qmax: float | None = None,
) -> np.ndarray:
    """
    Boolean mask for pixels with phi within [phi0-dphi, phi0+dphi], with wraparound.
    Optionally also apply a q-range cut.
    """
    phi = np.asarray(phi_map, dtype=np.float64)
    if valid is None:
        vv = np.isfinite(phi)
    else:
        vv = np.asarray(valid, dtype=bool) & np.isfinite(phi)

    # wrap delta-phi to [-pi, pi]
    d = phi - float(phi0)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi

    m = vv & (np.abs(d) <= float(dphi))

    if q_map is not None and (qmin is not None or qmax is not None):
        q = np.asarray(q_map, dtype=np.float64)
        if qmin is not None:
            m &= (q >= float(qmin))
        if qmax is not None:
            m &= (q <= float(qmax))

    return m

def build_q_phi_maps_from_static_qmap(f, *, use_index_mapping: bool = True):
    """
    Build per-pixel Q_map and Phi_map (same shape as detector image)
    from the *static* qmap products in the results file.

    Assumes:
      - static_roi_map values: 0..(n_q*n_phi), where 0 means background
      - q_vals length = n_q, phi_vals length = n_phi
      - bins packed with phi fastest: flat = q_i*n_phi + phi_i
      - static_index_mapping is a length-(n_q*n_phi) permutation (optional)
    """
    roi_map = f["xpcs/qmap/static_roi_map"][...]
    idx_map = f["xpcs/qmap/static_index_mapping"][...]
    q_vals  = f["xpcs/qmap/static_v_list_dim0"][...]
    phi_vals = f["xpcs/qmap/static_v_list_dim1"][...]

    roi_map = np.asarray(roi_map)
    idx_map = np.asarray(idx_map, dtype=np.int64)
    q_vals = np.asarray(q_vals, dtype=np.float64)
    phi_vals = np.asarray(phi_vals, dtype=np.float64)

    n_q = int(q_vals.size)
    n_phi = int(phi_vals.size)
    n_bins = n_q * n_phi  # should be 3600

    # valid pixels are labeled 1..n_bins (0 = background)
    valid = (roi_map > 0) & (roi_map <= n_bins)

    Q_map = np.full(roi_map.shape, np.nan, dtype=np.float64)
    Phi_map = np.full(roi_map.shape, np.nan, dtype=np.float64)

    if not np.any(valid):
        return Q_map, Phi_map

    # 1..n_bins -> 0..n_bins-1
    roi0 = roi_map[valid].astype(np.int64) - 1

    if use_index_mapping:
        flat = idx_map[roi0]   # permutation into 0..n_bins-1
    else:
        flat = roi0            # assume roi labels already match flat packing

    # unpack (phi fastest)
    q_i = flat // n_phi
    p_i = flat %  n_phi

    Q_map[valid] = q_vals[q_i]
    Phi_map[valid] = phi_vals[p_i]
    return Q_map, Phi_map

def overlay_mask_contour(ax, mask: np.ndarray, *, color="lime", lw=2.0, alpha=0.9, zorder=10):
    """
    Draw contour boundary of a boolean mask on an existing imshow axes.
    Works like your ROI contours.
    """
    m = np.asarray(mask, dtype=bool)
    ax.contour(m.astype(np.float32), levels=[0.5], colors=[color], linewidths=float(lw),
               alpha=float(alpha), zorder=int(zorder))

def plot_bragg_peak_shape_metrics_overlay_from_maps(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    metrics: dict,
    valid_mask: np.ndarray | None = None,
    cmap: str = "magma",
    n_q_contours: int = 7,
    n_phi_contours: int = 7,
    crop_half_size: int = 250,
):
    """
    Overlay iso-q and iso-phi contours on detector-space scattering_2d,
    plus markers for (q_mean, phi_mean) projected back to pixels.

    This is the detector-space counterpart to the old binned overlay.
    """
    img = np.asarray(scattering_2d)
    if img.ndim == 3:
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}")

    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    if not np.any(valid):
        raise RuntimeError("No valid pixels for overlay.")

    # log-ish display without blowing out
    disp = np.log10(np.clip(img, 0.0, None) + 1e-12)
    img_pos = img[img > 0]

    vmin = np.percentile(img_pos, 1.0)
    vmax = np.percentile(img_pos, 99.9)

    vv = valid  # or your ROI mask if you have one
    w = np.clip(img[vv], 0.0, None)

    iy, ix = np.nonzero(vv)
    sw = float(np.sum(w))
    if sw <= 0:
        raise RuntimeError("No positive weight for centroid.")

    cx = float(np.sum(w * ix) / sw)
    cy = float(np.sum(w * iy) / sw)

    half = crop_half_size  # 500x500 box

    H, W = img.shape
    x0 = int(max(cx - half, 0))
    x1 = int(min(cx + half, W))
    y0 = int(max(cy - half, 0))
    y1 = int(min(cy + half, H))

    img_c = img[y0:y1, x0:x1]
    q_c = q_map[y0:y1, x0:x1]
    phi_c = phi_map[y0:y1, x0:x1]
    valid_c = valid[y0:y1, x0:x1]

    extent = [x0, x1, y1, y0]

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.2))
    im = ax.imshow(
        img_c,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
        extent=extent,
    )

    ax.set_title("Detector-space scattering_2d with q/phi contour overlay")
    ax.set_facecolor("black")
    ax.set_xlabel("Detector x (pixels)")
    ax.set_ylabel("Detector y (pixels)")

    # Contour levels chosen from valid pixels
    qv = q_map[valid]
    phv = phi_map[valid]

    q_lo, q_hi = np.nanpercentile(qv, [2, 98])
    ph_lo, ph_hi = np.nanpercentile(phv, [2, 98])

    q_levels = np.linspace(q_lo, q_hi, int(n_q_contours))
    ph_levels = np.linspace(ph_lo, ph_hi, int(n_phi_contours))

    # Mask invalid pixels for contouring
    q_for_contour = np.where(valid, q_map, np.nan)
    ph_for_contour = np.where(valid, phi_map, np.nan)

    # --- contour on CROPPED maps, using the same extent as imshow ---
    # flip vertically to match imshow(origin="upper")
    q_for_contour_c = np.where(valid_c, q_c, np.nan)
    ph_for_contour_c = np.where(valid_c, phi_c, np.nan)

    q_for_contour_c = np.flipud(q_for_contour_c)
    ph_for_contour_c = np.flipud(ph_for_contour_c)

    ax.contour(
        q_for_contour_c,
        levels=q_levels,
        linewidths=0.7,
        alpha=0.8,
        extent=extent,
    )

    ax.contour(
        ph_for_contour_c,
        levels=ph_levels,
        linewidths=0.7,
        alpha=0.5,
        extent=extent,
    )

    # --- lock the view to the crop (contours can otherwise autoscale) ---
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # because origin="upper"

    # Mark the mean (q_mean, phi_mean) by finding nearest pixel in (q,phi) space
    q_mean = float(metrics.get("q_mean"))
    phi_mean = float(metrics.get("phi_mean_rad"))

    ax.plot([cx], [cy], marker="x", markersize=10, mew=2)
    ax.text(
        cx + 50,
        cy + 50,
        f"Pixel centroid (x̄,ȳ)\nq̄={q_mean:.4f}\nφ̄={phi_mean:.4f} rad",
        fontsize=12,
        color="yellow",
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="log10(Intensity + eps)")
    fig.tight_layout()
    plt.show()

    return {"center_px_rc": (float(cy), float(cx))}

def regrid_detector_to_qphi(
    img: np.ndarray,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    q_edges: np.ndarray | None = None,
    phi_edges: np.ndarray | None = None,
    n_q: int = 400,
    n_phi: int = 360,
    phi_wrap: str = "pi",   # "pi" -> [-pi,pi), "2pi" -> [0,2pi)
    statistic: str = "mean" # "mean" or "median"
) -> dict:
    """
    Regrid detector-space intensity img(y,x) onto a regular (q, phi) grid using binning.

    Returns a dict with q_centers, phi_centers, I_qphi, N_qphi, edges.
    Shapes: I_qphi and N_qphi are (Nphi, Nq).
    """
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}")

    if valid_mask is None:
        vv = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        vv = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    if not np.any(vv):
        raise RuntimeError("No valid pixels to regrid.")

    qv = q_map[vv].ravel()
    phv = phi_map[vv].ravel()
    Iv = img[vv].ravel()

    # Wrap phi
    if phi_wrap == "pi":
        phv = (phv + np.pi) % (2.0 * np.pi) - np.pi
        default_phi_lo, default_phi_hi = -np.pi, np.pi
    elif phi_wrap == "2pi":
        phv = phv % (2.0 * np.pi)
        default_phi_lo, default_phi_hi = 0.0, 2.0 * np.pi
    else:
        raise ValueError("phi_wrap must be 'pi' or '2pi'")

    # Build edges if not provided
    if q_edges is None:
        q_lo, q_hi = np.nanpercentile(qv, [0.5, 99.5])
        q_edges = np.linspace(float(q_lo), float(q_hi), int(n_q) + 1)

    if phi_edges is None:
        # You can also restrict phi range to the detector-covered region if you want
        phi_edges = np.linspace(float(default_phi_lo), float(default_phi_hi), int(n_phi) + 1)

    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # Bin indices
    iq = np.searchsorted(q_edges, qv, side="right") - 1
    ip = np.searchsorted(phi_edges, phv, side="right") - 1

    Nq = len(q_centers)
    Np = len(phi_centers)

    in_range = (iq >= 0) & (iq < Nq) & (ip >= 0) & (ip < Np)
    iq = iq[in_range]
    ip = ip[in_range]
    Iv = Iv[in_range]

    # Accumulate
    N_qphi = np.zeros((Np, Nq), dtype=np.int64)

    if statistic == "mean":
        S_qphi = np.zeros((Np, Nq), dtype=np.float64)
        np.add.at(S_qphi, (ip, iq), Iv)
        np.add.at(N_qphi, (ip, iq), 1)
        I_qphi = np.full((Np, Nq), np.nan, dtype=np.float64)
        m = N_qphi > 0
        I_qphi[m] = S_qphi[m] / N_qphi[m]

    elif statistic == "median":
        # Median needs storing lists per bin (slower but robust to hot pixels)
        bins = [[[] for _ in range(Nq)] for __ in range(Np)]
        for p, q, val in zip(ip, iq, Iv):
            bins[p][q].append(float(val))
        I_qphi = np.full((Np, Nq), np.nan, dtype=np.float64)
        for p in range(Np):
            for q in range(Nq):
                if bins[p][q]:
                    arr = np.asarray(bins[p][q], dtype=np.float64)
                    I_qphi[p, q] = float(np.median(arr))
                    N_qphi[p, q] = int(arr.size)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    return {
        "q_edges": q_edges,
        "phi_edges": phi_edges,
        "q_centers": q_centers,
        "phi_centers": phi_centers,
        "I_qphi": I_qphi,     # shape (Nphi, Nq)
        "N_qphi": N_qphi,     # shape (Nphi, Nq)
    }

def plot_bragg_peak_shape_metrics_overlay(
    scattering_2d,
    *,
    q_vals,
    phi_vals,
    metrics: dict,
    title: str = "Bragg peak shape metrics (fixed q/phi axes)",
    cmap: str = "magma",
):
    """
    Standard overlay plot:
      - 2D image (log display) with (center_q, center_phi) marker
      - marginal profiles Wq(q) and Wphi(phi)
    """
    img = _ensure_2d(scattering_2d)
    img, q, phi = _match_axes(img, q_vals, phi_vals)

    # Display image as log1p for visibility without thresholding
    disp = np.log1p(np.clip(img, 0.0, np.inf))

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        disp,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[q[0], q[-1], phi[0], phi[-1]],
    )
    ax0.plot([metrics["center_q"]], [metrics["center_phi"]], marker="x", markersize=10, mew=2)
    ax0.set_xlabel("q")
    ax0.set_ylabel("phi")
    ax0.set_title("log1p(scattering_2d) + center")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(q, metrics["q_profile"], lw=1.8)
    ax1.set_xlabel("q")
    ax1.set_title(
        f"Wq(q)\n"
        f"μ={metrics['center_q']:.6g}, σ={metrics['sigma_q']:.4g}, "
        f"skew={metrics['skew_q']:.3g}"
    )
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(phi, metrics["phi_profile"], lw=1.8)
    ax2.set_xlabel("phi")
    ax2.set_title(
        f"Wφ(φ)\n"
        f"μ={metrics['center_phi']:.6g}, σ={metrics['sigma_phi']:.4g}, "
        f"skew={metrics['skew_phi']:.3g}"
    )
    ax2.grid(True, alpha=0.25)

    fig.suptitle(
        f"{title}\n"
        f"Bowley skew: q={metrics['bowley_skew_q']:.3g}, phi={metrics['bowley_skew_phi']:.3g} | "
        f"Neff={metrics['neff']:.1f}",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()
    return fig

def bragg_peak_shape_metrics_fixed_q_phi_from_maps(
    scattering_2d: np.ndarray,
    *,
    q_map: np.ndarray,
    phi_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    hot_z_thresh: float = 12.0,
    eps: float = 1e-12,
) -> dict:
    """
    Compute Bragg-peak shape metrics with q and phi defined per DETECTOR PIXEL.

    Inputs
    ------
    scattering_2d : (H,W) or (1,H,W)
        Detector-space average image (NOT 120x30 binned).
    q_map, phi_map : (H,W)
        Per-pixel q and phi maps (e.g. from inferred_qphi_maps.npz).
    valid_mask : (H,W) bool or None
        Optional mask of valid pixels. If None, inferred from finiteness.
    hot_z_thresh : float
        Suppress extreme hot pixels using robust z-score on intensity.
        This only rejects extreme outliers, it does NOT mask long physical tails.

    Returns
    -------
    dict of scalar metrics (means, sigmas, skewness in q and phi).
    """
    img = np.asarray(scattering_2d)
    if img.ndim == 3:
        if img.shape[0] != 1:
            raise ValueError(f"Expected scattering_2d with leading dim 1, got {img.shape}")
        img = img[0]
    img = np.asarray(img, dtype=np.float64)

    H, W = img.shape
    iy_max, ix_max = np.unravel_index(np.nanargmax(img), img.shape)
    print("argmax (ix, iy):", ix_max, iy_max)
    print("argmax display-y if origin='lower':", (H - 1 - iy_max))
    print("img max:", img[iy_max, ix_max])

    q_map = np.asarray(q_map, dtype=np.float64)
    phi_map = np.asarray(phi_map, dtype=np.float64)

    if img.shape != q_map.shape or img.shape != phi_map.shape:
        raise ValueError(
            f"Shape mismatch: img={img.shape}, q_map={q_map.shape}, phi_map={phi_map.shape}"
        )

    if valid_mask is None:
        valid = np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(img) & np.isfinite(q_map) & np.isfinite(phi_map)

    # ---- hot pixel suppression (robust z-score on intensity)
    # This is a minimal, conservative rejection of extreme spikes.
    vv = valid
    if np.any(vv):
        med = np.nanmedian(img[vv])
        mad = np.nanmedian(np.abs(img[vv] - med))
        sigma_rob = 1.4826 * mad + eps
        z = (img - med) / sigma_rob
        vv = vv & (z < float(hot_z_thresh))

    if not np.any(vv):
        raise RuntimeError("No valid pixels after masking / hot-pixel suppression.")

    # freeze the mask used for pixel-space moments
    vv_used = vv.copy()

    iy0, ix0 = np.unravel_index(np.nanargmax(img), img.shape)

    half_w = 250  # tune
    half_h = 250  # tune
    roi = np.zeros_like(vv_used, dtype=bool)
    y0, x0 = iy0, ix0
    roi[max(0, y0 - half_h):min(img.shape[0], y0 + half_h + 1),
    max(0, x0 - half_w):min(img.shape[1], x0 + half_w + 1)] = True

    vv_used = vv_used & roi

    w = img[vv_used].copy()

    # Weights must be non-negative for moment interpretation
    # (if you have negative values from processing, clip them softly)
    w = np.clip(w, 0.0, None)

    iy, ix = np.nonzero(vv_used)

    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= eps:
        raise RuntimeError("Sum of weights is zero or non-finite, cannot compute moments.")

    qv = q_map[vv_used]
    ph = phi_map[vv_used]

    # ---- weighted mean in q
    q_mean = float(np.sum(w * qv) / sw)

    # ---- weighted circular mean in phi
    c = float(np.sum(w * np.cos(ph)) / sw)
    s = float(np.sum(w * np.sin(ph)) / sw)
    phi_mean = float(np.arctan2(s, c))

    # unwrap phi about mean to compute *directional* moments
    dphi = ph - phi_mean
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]

    dq = qv - q_mean

    # ---- weighted central moments and skewness
    mu2_q = float(np.sum(w * dq * dq) / sw)
    mu3_q = float(np.sum(w * dq * dq * dq) / sw)

    mu2_phi = float(np.sum(w * dphi * dphi) / sw)
    mu3_phi = float(np.sum(w * dphi * dphi * dphi) / sw)

    sigma_q = float(np.sqrt(max(mu2_q, 0.0)))
    sigma_phi = float(np.sqrt(max(mu2_phi, 0.0)))

    skew_q = float(mu3_q / (mu2_q ** 1.5 + eps))
    skew_phi = float(mu3_phi / (mu2_phi ** 1.5 + eps))

    # Optional extra diagnostics that are often useful
    peak_adu = float(np.nanmax(img[vv_used]))
    n_pix = int(np.sum(vv_used))
    frac_kept = float(n_pix / np.sum(valid)) if np.sum(valid) > 0 else float("nan")

    # ----------------------------
    # Pixel-space (x, y) metrics
    # ----------------------------
    # iy, ix = np.nonzero(valid_mask)

    w_sum = np.sum(w)

    # centroids in pixel coordinates
    x_mean = np.sum(w * ix) / w_sum
    y_mean = np.sum(w * iy) / w_sum

    # second moments
    sigma_x = np.sqrt(np.sum(w * (ix - x_mean) ** 2) / w_sum)
    sigma_y = np.sqrt(np.sum(w * (iy - y_mean) ** 2) / w_sum)

    # skewness
    skew_x = np.sum(w * (ix - x_mean) ** 3) / (w_sum * sigma_x ** 3)
    skew_y = np.sum(w * (iy - y_mean) ** 3) / (w_sum * sigma_y ** 3)

    H, W = img.shape
    print("argmax array (x,y):", ix0, iy0, " display-y:", (H - 1) - iy0)
    print("centroid array (x,y):", x_mean, y_mean, " display-y:", (H - 1) - y_mean)

    return {
        "x_mean_px": float(x_mean),
        "y_mean_px": float(y_mean),
        "sigma_x_px": float(sigma_x),
        "sigma_y_px": float(sigma_y),
        "skew_x": float(skew_x),
        "skew_y": float(skew_y),
        "q_mean": q_mean,
        "phi_mean_rad": phi_mean,
        "sigma_q": sigma_q,
        "sigma_phi_rad": sigma_phi,
        "skew_q": skew_q,
        "skew_phi": skew_phi,
        "n_pix_used": n_pix,
        "frac_valid_used": frac_kept,
        "peak_intensity": peak_adu,
        "hot_z_thresh": float(hot_z_thresh),
    }



def exec_bragg_peak_shape_metrics_fixed_q_phi():

    with h5py.File(filename, "r") as f:
        scattering_2d = f["xpcs/temporal_mean/scattering_2d"][...]
        q = f["xpcs/qmap/static_v_list_dim0"][...]
        phi = f["xpcs/qmap/static_v_list_dim1"][...]

    # Example: point this to the file you just saved
    npz_path = H5_FILE.parent / "A073_inferred_qphi_maps.npz"
    d = np.load(npz_path, allow_pickle=False)

    metrics = bragg_peak_shape_metrics_fixed_q_phi_from_maps(
        scattering_2d,
        q_map=d["q_map"],
        phi_map=d["phi_map"],
        valid_mask=d.get("valid_mask", None),
        hot_z_thresh=12.0,
    )

    print("Bragg peak shape metrics (fixed q/phi via per-pixel maps):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_bragg_peak_shape_metrics_overlay_from_maps(
        scattering_2d,
        q_map=d["q_map"],
        phi_map=d["phi_map"],
        valid_mask=d.get("valid_mask", None),
        metrics=metrics,
        cmap="magma",
        crop_half_size=500,
    )

    return metrics


def infer_q_phi_maps_from_static_qmap(
    f,
    *,
    q_key: str = "xpcs/qmap/static_v_list_dim0",
    phi_key: str = "xpcs/qmap/static_v_list_dim1",
    roi_map_key: str = "xpcs/qmap/static_roi_map",
    index_map_key: str = "xpcs/qmap/static_index_mapping",
    num_pts_key: str = "xpcs/qmap/static_num_pts",
    invalid_roi_value: int = 0,
):
    """
    Build per-pixel q_map / phi_map (and pseudo-Qx/Qy) from the static qmap products.

    Requires:
      - static_roi_map: per-pixel ROI id
      - static_index_mapping: length Nq*Nphi, maps (iq,iphi) bin -> ROI id
      - static_v_list_dim0: q bin centers (Nq)
      - static_v_list_dim1: phi bin centers (Nphi)

    Returns
    -------
    out : dict with keys
      q_map, phi_map, Qx_map, Qy_map, valid_mask, roi_map, q_vals, phi_vals

    Notes
    -----
    - phi may be in degrees or radians; we infer by range and convert to radians for cos/sin.
    - invalid_roi_value (often 0) is treated as background/invalid.
    """
    q_vals = np.asarray(f[q_key][...], dtype=np.float64)          # (Nq,)
    phi_vals = np.asarray(f[phi_key][...], dtype=np.float64)      # (Nphi,)
    roi_map = np.asarray(f[roi_map_key][...], dtype=np.int64)     # (ny,nx)

    # index_mapping: length Nq*Nphi, values are ROI ids (uint16)
    idx_to_roi = np.asarray(f[index_map_key][...], dtype=np.int64)

    # sanity check Nq,Nphi from file (optional but helpful)
    if num_pts_key in f:
        npts = np.asarray(f[num_pts_key][...], dtype=np.int64).ravel()
        if npts.size == 2:
            Nq_file, Nphi_file = int(npts[0]), int(npts[1])
            if Nq_file != q_vals.size or Nphi_file != phi_vals.size:
                raise ValueError(
                    f"static_num_pts says (Nq,Nphi)=({Nq_file},{Nphi_file}) "
                    f"but v_list sizes are (Nq,Nphi)=({q_vals.size},{phi_vals.size})"
                )

    Nq = int(q_vals.size)
    Nphi = int(phi_vals.size)
    if idx_to_roi.size != Nq * Nphi:
        raise ValueError(
            f"static_index_mapping has length {idx_to_roi.size}, expected Nq*Nphi={Nq*Nphi}"
        )

    # infer phi units -> radians
    # (if values look like degrees, convert)
    phi_max = float(np.nanmax(phi_vals))
    phi_min = float(np.nanmin(phi_vals))
    # crude but reliable for your typical bin-centers
    if (phi_max - phi_min) > (2.0 * np.pi + 0.5):
        phi_rad = np.deg2rad(phi_vals)
    else:
        phi_rad = phi_vals.copy()

    # Build ROI->(q,phi) lookup.
    # idx = iq*Nphi + iphi  (iphi changes fastest, matches your "repeats every 30")
    # idx_to_roi[idx] gives ROI id
    roi_max = int(np.max(roi_map))
    roi_to_q = np.full((roi_max + 1,), np.nan, dtype=np.float64)
    roi_to_phi = np.full((roi_max + 1,), np.nan, dtype=np.float64)

    for iq in range(Nq):
        for iphi in range(Nphi):
            lin = iq * Nphi + iphi
            roi_id = int(idx_to_roi[lin])
            if roi_id <= 0 or roi_id > roi_max:
                continue
            # assign (q,phi) for that ROI id
            # if duplicates exist, they should match; if not, last wins
            roi_to_q[roi_id] = float(q_vals[iq])
            roi_to_phi[roi_id] = float(phi_rad[iphi])

    # Now map per-pixel
    valid_mask = (roi_map != int(invalid_roi_value)) & (roi_map >= 0) & (roi_map <= roi_max)

    q_map = np.full_like(roi_map, np.nan, dtype=np.float64)
    phi_map = np.full_like(roi_map, np.nan, dtype=np.float64)

    q_map[valid_mask] = roi_to_q[roi_map[valid_mask]]
    phi_map[valid_mask] = roi_to_phi[roi_map[valid_mask]]

    # pseudo-components
    Qx_map = np.full_like(q_map, np.nan, dtype=np.float64)
    Qy_map = np.full_like(q_map, np.nan, dtype=np.float64)

    vv = valid_mask & np.isfinite(q_map) & np.isfinite(phi_map)
    Qx_map[vv] = q_map[vv] * np.cos(phi_map[vv])
    Qy_map[vv] = q_map[vv] * np.sin(phi_map[vv])

    return {
        "q_map": q_map,
        "phi_map": phi_map,
        "Qx_map": Qx_map,
        "Qy_map": Qy_map,
        "valid_mask": vv,
        "roi_map": roi_map,
        "q_vals": q_vals,
        "phi_vals": phi_vals,
    }


def save_inferred_qphi_maps_npz(
    results_hdf_path: str | Path,
    out_npz_path: str | Path,
):
    """
    Convenience wrapper: open results .hdf, infer maps, save to .npz.
    """
    results_hdf_path = Path(results_hdf_path)
    out_npz_path = Path(out_npz_path)

    with h5py.File(results_hdf_path, "r") as f:
        # your results file stores under /xpcs/...
        maps = infer_q_phi_maps_from_static_qmap(f, q_key="xpcs/qmap/static_v_list_dim0",
                                                 phi_key="xpcs/qmap/static_v_list_dim1",
                                                 roi_map_key="xpcs/qmap/static_roi_map",
                                                 index_map_key="xpcs/qmap/static_index_mapping",
                                                 num_pts_key="xpcs/qmap/static_num_pts")

    np.savez_compressed(
        out_npz_path,
        q_map=maps["q_map"],
        phi_map=maps["phi_map"],
        Qx_map=maps["Qx_map"],
        Qy_map=maps["Qy_map"],
        valid_mask=maps["valid_mask"],
        roi_map=maps["roi_map"],
        q_vals=maps["q_vals"],
        phi_vals=maps["phi_vals"],
    )
    print(f"Saved inferred maps: {out_npz_path}")
    return out_npz_path

def exec_make_and_save_inferred_qphi_maps():
    results = H5_FILE
    out = H5_FILE.parent / "A073_inferred_qphi_maps.npz"
    save_inferred_qphi_maps_npz(results, out)

def exec_quick_check_inferred_qphi_npz():

    npz_path = H5_FILE.parent / "A073_inferred_qphi_maps.npz"

    d = np.load(Path(npz_path), allow_pickle=False)

    q_map = d["q_map"]
    phi_map = d["phi_map"]
    Qx_map = d["Qx_map"]
    Qy_map = d["Qy_map"]
    valid = d["valid_mask"]

    print("Shapes:")
    print("  q_map:", q_map.shape, "phi_map:", phi_map.shape, "valid:", valid.shape)
    print("Valid fraction:", float(np.mean(valid)))

    print("Ranges on valid pixels:")
    vv = valid & np.isfinite(q_map) & np.isfinite(phi_map)
    print("  q:", float(np.nanmin(q_map[vv])), "to", float(np.nanmax(q_map[vv])))
    print("  phi(rad):", float(np.nanmin(phi_map[vv])), "to", float(np.nanmax(phi_map[vv])))
    print("  Qx:", float(np.nanmin(Qx_map[vv])), "to", float(np.nanmax(Qx_map[vv])))
    print("  Qy:", float(np.nanmin(Qy_map[vv])), "to", float(np.nanmax(Qy_map[vv])))

    # quick visual
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax0, ax1, ax2 = axes
    im0 = ax0.imshow(q_map, origin="upper", interpolation="nearest")
    ax0.set_title("q_map")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)

    im1 = ax1.imshow(phi_map, origin="upper", interpolation="nearest")
    ax1.set_title("phi_map (rad)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    im2 = ax2.imshow(valid, origin="upper", interpolation="nearest")
    ax2.set_title("valid_mask")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    plt.show()

def exec_build_q_phi_map():

    with h5py.File(h5_file, "r") as f:
        with h5py.File(h5_file, "r") as f:
            Q_full, Phi_full = build_q_phi_maps_from_geometry(f)

    print(Q_full)
    print(Phi_full)
    print(np.shape(Q_full))
    print(np.shape(Phi_full))

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(
        Q_full,
        origin="upper",
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title("Q map (per pixel)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, label="q")

    im1 = axes[1].imshow(
        Phi_full,
        origin="upper",
        cmap="twilight",
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Phi map (per pixel)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03, label="phi (deg)")

    fig.suptitle("Full detector q / φ maps (nearest-valid extrapolated)", fontsize=14)
    fig.tight_layout()
    plt.show()

def _wrap_deg_to_180(phi_deg: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180)."""
    return (phi_deg + 180.0) % 360.0 - 180.0


def _circular_mean_deg(phi_deg: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> float:
    """Weighted circular mean in degrees, returns value in [-180, 180)."""
    phi_rad = np.deg2rad(phi_deg)
    sw = float(np.sum(w)) + eps
    c = float(np.sum(w * np.cos(phi_rad)) / sw)
    s = float(np.sum(w * np.sin(phi_rad)) / sw)
    mean_rad = float(np.arctan2(s, c))
    return float(_wrap_deg_to_180(np.rad2deg(mean_rad)))


def _circular_centered_deg(phi_deg: np.ndarray, phi0_deg: float) -> np.ndarray:
    """
    Return signed angular difference (deg) from phi0, wrapped to [-180, 180).
    """
    return _wrap_deg_to_180(phi_deg - float(phi0_deg))


def _weighted_moments_1d(x: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    """
    Weighted mean, sigma, skewness for 1D variable x.
    Skewness is central mu3 / mu2^(3/2).
    """
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= eps:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.sum(w * x) / sw)
    dx = x - mean
    mu2 = float(np.sum(w * dx * dx) / sw)
    mu3 = float(np.sum(w * dx * dx * dx) / sw)

    sigma = float(np.sqrt(max(mu2, 0.0)))
    skew = float(mu3 / (mu2 ** 1.5 + eps))
    return mean, sigma, skew


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    sa = float(np.linalg.norm(a))
    sb = float(np.linalg.norm(b))
    if sa <= eps or sb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (sa * sb))

def _peak_anchored_tail_asymmetry_1d(x: np.ndarray, w: np.ndarray, i0: int, p: float = 1.0, eps: float = 1e-12):
    """
    Peak-anchored left/right tail asymmetry about the ARGMAX index i0.

    A(p) = (L - R) / (L + R), where
      L = sum_{i<i0} w_i * |x_i - x0|^p
      R = sum_{i>i0} w_i * |x_i - x0|^p

    Negative => heavier/brighter tail on the LEFT (smaller x).
    Positive => heavier/brighter tail on the RIGHT (larger x).
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)

    if not (0 <= int(i0) < x.size):
        raise ValueError("i0 out of range")

    x0 = float(x[int(i0)])

    left = slice(0, int(i0))
    right = slice(int(i0) + 1, x.size)

    dxL = np.abs(x[left] - x0)
    dxR = np.abs(x[right] - x0)

    L = float(np.sum(w[left] * (dxL ** float(p))))
    R = float(np.sum(w[right] * (dxR ** float(p))))

    A = (R - L) / (L + R + float(eps))

    print("tail debug:",
          "x0=", x0,
          "L=", L, "R=", R,
          "A(L-R)=", (L - R) / (L + R + eps),
          "A(R-L)=", (R - L) / (L + R + eps))

    return A, L, R, x0


def _argmax_in_roi(I_qphi: np.ndarray, roi_qphi: np.ndarray) -> tuple[int, int]:
    """
    Return (iq0, iphi0) of the maximum intensity inside ROI.
    """
    M = np.where(roi_qphi, I_qphi, -np.inf)
    flat = int(np.argmax(M))
    iq0, iphi0 = np.unravel_index(flat, I_qphi.shape)
    return int(iq0), int(iphi0)

def integrated_intensities_peak_stability(
    h5_file: str | Path,
    *,
    q_range: tuple[float, float] = (1.09, 1.13),
    phi_range_deg: tuple[float, float] = (-10.0, 10.0),
    use_log_for_similarity: bool = True,
    log_eps: float = 1e-12,
    weight_power: float = 1.0,
    show_plots: bool = True,
) -> dict:
    """
    Peak + stability metrics using scattering_1d and scattering_1d_segments (10 segments).

    Physics ROI:
      - q within q_range
      - phi within phi_range_deg  (phi is in degrees)

    Metrics per segment (in ROI):
      - q_mean, sigma_q, skew_q
      - phi_mean_deg (circular), sigma_phi_deg, skew_phi
      - I_tot, I_peak

    Stability vs mean pattern (in ROI):
      - cosine_similarity[s]
      - pearson_r[s]  (on log(I+eps) if use_log_for_similarity=True)

    Returns a dict with arrays and a few summary scalars.
    """
    h5_file = str(h5_file)

    with h5py.File(h5_file, "r") as f:
        I1d = np.asarray(f["xpcs/temporal_mean/scattering_1d"][...], dtype=np.float64)
        Iseg = np.asarray(f["xpcs/temporal_mean/scattering_1d_segments"][...], dtype=np.float64)
        q = np.asarray(f["xpcs/qmap/static_v_list_dim0"][...], dtype=np.float64)
        phi = np.asarray(f["xpcs/qmap/static_v_list_dim1"][...], dtype=np.float64)

    # ---- sanity ----
    if I1d.ndim != 2 or I1d.shape[0] != 1:
        raise ValueError(f"Expected scattering_1d shape (1, 3600), got {I1d.shape}")
    if Iseg.ndim != 2 or Iseg.shape[1] != I1d.shape[1]:
        raise ValueError(f"Expected scattering_1d_segments shape (10, 3600), got {Iseg.shape}")

    nq = int(q.size)
    nphi = int(phi.size)
    if nq * nphi != int(I1d.shape[1]):
        raise ValueError(f"q.size * phi.size = {nq*nphi} does not match scattering_1d length {I1d.shape[1]}")

    # ---- reshape (phi fast axis) ----
    I_mean_qphi = I1d[0].reshape(nq, nphi)
    I_mean_phiq = I_mean_qphi.T
    I_seg_qphi = Iseg.reshape(Iseg.shape[0], nq, nphi)  # (nseg, nq, nphi)
    nseg = int(I_seg_qphi.shape[0])

    I_q = I_mean_qphi.mean(axis=1)  # or sum(axis=1), depending on what you plot
    # print("argmax iq:", np.argmax(I_q), "q at argmax:", q[np.argmax(I_q)])
    # print("left edge q:", q[0], "right edge q:", q[-1])

    I_q = I_mean_qphi.mean(axis=1)  # or sum(axis=1)
    iq0 = int(np.argmax(I_q))

    # print("peak:", iq0, q[iq0], I_q[iq0])
    # print("left  (iq0-5..iq0-1):", list(zip(range(iq0 - 5, iq0), q[iq0 - 5:iq0], I_q[iq0 - 5:iq0])))
    # print("right (iq0+1..iq0+5):", list(zip(range(iq0 + 1, iq0 + 6), q[iq0 + 1:iq0 + 6], I_q[iq0 + 1:iq0 + 6])))

    # compare_q_skew_two_methods(
    #     I_mean_qphi,
    #     I_seg_qphi,
    #     q,
    #     phi,
    #     choose_phi="argmax",  # or "closest0"
    #     # iphi=24,             # optional explicit override
    # )

    # ---- build ROI mask in (q,phi) ----
    q_lo, q_hi = float(min(q_range)), float(max(q_range))
    ph_lo, ph_hi = float(min(phi_range_deg)), float(max(phi_range_deg))

    qq, pp = np.meshgrid(q, phi, indexing="ij")  # (nq,nphi)
    roi = (qq >= q_lo) & (qq <= q_hi) & (pp >= ph_lo) & (pp <= ph_hi)

    if not np.any(roi):
        raise RuntimeError("ROI is empty. Check q_range and phi_range_deg against q,phi arrays.")

    # ----------------------------
    # Peak-anchored tail "skew" (Method A analog) for q and phi
    # ----------------------------
    # Anchor: argmax bin in the MEAN map, restricted to ROI
    M = np.where(roi, I_mean_qphi, -np.inf)
    iq0, iphi0 = np.unravel_index(int(np.argmax(M)), M.shape)

    # arrays to plot in ax4
    q_skew_peak = np.full((nseg,), np.nan, dtype=np.float64)
    phi_skew_peak = np.full((nseg,), np.nan, dtype=np.float64)

    # choose p=1.0 (your earlier output used p=1 as the intuitive tail metric)
    p_tail = 1.0

    for s in range(nseg):
        Is = I_seg_qphi[s]

        # q tail: single-phi lineout at iphi0
        wq = np.clip(Is[:, iphi0].astype(np.float64), 0.0, None)
        Aq, _, _, _ = _peak_anchored_tail_asymmetry_1d(q, wq, int(iq0), p=p_tail, eps=log_eps)
        q_skew_peak[s] = Aq

        # phi tail: single-q lineout at iq0
        wphi = np.clip(Is[int(iq0), :].astype(np.float64), 0.0, None)
        Aphi, _, _, _ = _peak_anchored_tail_asymmetry_1d(phi, wphi, int(iphi0), p=p_tail, eps=log_eps)
        phi_skew_peak[s] = Aphi

    # ---- per-segment metrics ----
    q_mean = np.full((nseg,), np.nan, dtype=np.float64)
    q_sigma = np.full((nseg,), np.nan, dtype=np.float64)
    q_skew = np.full((nseg,), np.nan, dtype=np.float64)

    phi_mean = np.full((nseg,), np.nan, dtype=np.float64)       # degrees
    phi_sigma = np.full((nseg,), np.nan, dtype=np.float64)      # degrees
    phi_skew = np.full((nseg,), np.nan, dtype=np.float64)

    I_tot = np.full((nseg,), np.nan, dtype=np.float64)
    I_peak = np.full((nseg,), np.nan, dtype=np.float64)

    # Flattened coords inside ROI
    q_roi = qq[roi].astype(np.float64)
    phi_roi = pp[roi].astype(np.float64)

    # Reference pattern for similarity
    I_roi_stack = np.empty((nseg, q_roi.size), dtype=np.float64)

    for s in range(nseg):
        Is = I_seg_qphi[s]
        w = Is[roi].astype(np.float64)

        # weights: non-negative, optionally emphasize peak
        w = np.clip(w, 0.0, None)
        if weight_power != 1.0:
            w = w ** float(weight_power)

        sw = float(np.sum(w))
        I_tot[s] = sw
        I_peak[s] = float(np.max(Is[roi]))

        # q moments
        qm, qs, qk = _weighted_moments_1d(q_roi, w)
        q_mean[s], q_sigma[s], q_skew[s] = qm, qs, qk

        # phi circular mean, then centered moments on wrapped differences
        phm = _circular_mean_deg(phi_roi, w)
        phi_mean[s] = phm
        dphi = _circular_centered_deg(phi_roi, phm)
        ph_mu, ph_sig, ph_sk = _weighted_moments_1d(dphi, w)
        phi_sigma[s] = ph_sig
        phi_skew[s] = ph_sk

        # store pattern for similarity (use raw intensity, similarity step decides log or not)
        I_roi_stack[s, :] = np.clip(Is[roi].astype(np.float64), 0.0, None)

    # ---- stability vs mean pattern ----
    I_ref = np.mean(I_roi_stack, axis=0)

    cos_sim = np.full((nseg,), np.nan, dtype=np.float64)
    pearson_r = np.full((nseg,), np.nan, dtype=np.float64)

    if use_log_for_similarity:
        Aref = np.log10(I_ref + float(log_eps))
    else:
        Aref = I_ref

    for s in range(nseg):
        A = np.log10(I_roi_stack[s] + float(log_eps)) if use_log_for_similarity else I_roi_stack[s]
        cos_sim[s] = _cosine_similarity(A, Aref)
        pearson_r[s] = _pearson_corr(A, Aref)

    # ---- summary scalars ----
    drift_q = float(np.nanmax(q_mean) - np.nanmin(q_mean))
    drift_phi = float(_wrap_deg_to_180(float(np.nanmax(phi_mean) - np.nanmin(phi_mean))))  # rough wrap-aware

    out = {
        "q": q,
        "phi_deg": phi,
        "q_range": (q_lo, q_hi),
        "phi_range_deg": (ph_lo, ph_hi),
        "roi_mask_qphi": roi,
        "I_mean_qphi": I_mean_qphi,
        "I_seg_qphi": I_seg_qphi,

        "q_mean": q_mean,
        "sigma_q": q_sigma,
        "skew_q": q_skew,

        "phi_mean_deg": phi_mean,
        "sigma_phi_deg": phi_sigma,
        "skew_phi": phi_skew,

        "I_tot": I_tot,
        "I_peak": I_peak,

        "cosine_similarity": cos_sim,
        "pearson_r": pearson_r,

        "drift_q": drift_q,
        "drift_phi_deg": drift_phi,

        "q_skew_peak": q_skew_peak,
        "phi_skew_peak": phi_skew_peak,
    }

    if show_plots:
        # ROI mean map
        mean_roi = np.where(roi, I_mean_qphi, np.nan)

        # Mean centroid (from mean map, in ROI)
        w0 = np.clip(I_mean_qphi[roi].astype(np.float64), 0.0, None)
        if weight_power != 1.0:
            w0 = w0 ** float(weight_power)
        q0m, _, _ = _weighted_moments_1d(q_roi, w0)
        ph0m = _circular_mean_deg(phi_roi, w0)

        fig = plt.figure(figsize=(14.5, 8))
        gs = fig.add_gridspec(2, 3, wspace=0.5, hspace=0.5)

        # (A) Mean map, with ROI boundary idea and centroid marker
        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(
            np.log10(np.clip(I_mean_phiq, 0.0, None) + float(log_eps)),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[q.min(), q.max(), phi.min(), phi.max()],
        )
        # ax0.axvline(ph_lo, lw=1.0, alpha=0.8)
        # ax0.axvline(ph_hi, lw=1.0, alpha=0.8)
        # ax0.axhline(q_lo, lw=1.0, alpha=0.8)
        # ax0.axhline(q_hi, lw=1.0, alpha=0.8)
        # ax0.plot([ph0m], [q0m], marker="x", ms=10, mew=2)
        ax0.set_title("Mean log10 intensity map (q, φ)\nROI bounds + mean centroid")
        ax0.set_ylabel("φ (deg)")
        ax0.set_xlabel("q (Å$^{-1}$)")
        fig.colorbar(im0, ax=ax0, fraction=0.04, pad=0.03, label="log10(I + eps)")

        # print(q[:5], q[-5:])

        # (B) Segment centroids drift in (q,phi)
        ax1 = fig.add_subplot(gs[0, 1])
        sc = ax1.scatter(phi_mean, q_mean, c=np.arange(nseg), s=60)
        ax1.plot(phi_mean, q_mean, lw=1.2, alpha=0.7)
        ax1.set_title("Segment centroid drift (q_mean vs φ_mean)")
        ax1.set_xlabel("φ_mean (deg)")
        ax1.set_ylabel("q_mean (Å$^{-1}$)")
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax1.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03, label="segment index")

        # (C) q_mean and phi_mean vs segment index
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(np.arange(nseg), q_mean, marker="o", lw=1.8, label="q_mean")
        ax2b = ax2.twinx()
        ax2b.plot(np.arange(nseg), phi_mean, marker="s", lw=1.6, alpha=0.85, label="phi_mean", linestyle="--")
        ax2.set_title("Centroid vs segment index")
        ax2.set_xlabel("segment index")
        ax2.set_ylabel("q_mean (Å$^{-1}$)")
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax2b.set_ylabel("φ_mean (deg)")
        ax2.grid(True, alpha=0.25)

        # (D) widths and skewness
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(np.arange(nseg), q_sigma, marker="o", lw=1.8, label="sigma_q")
        ax3.plot(np.arange(nseg), phi_sigma, marker="s", lw=1.6, alpha=0.85, label="sigma_phi (deg)")
        ax3.set_title("Widths vs segment")
        ax3.set_xlabel("segment index")
        ax3.set_ylabel("width")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best", fontsize=9)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(np.arange(nseg), q_skew_peak, marker="o", lw=1.8, label="q tail asym (peak-anchored)")
        ax4.plot(np.arange(nseg), phi_skew_peak, marker="s", lw=1.6, alpha=0.85, label="phi tail asym (peak-anchored)")
        ax4.set_title("Peak-anchored tail asymmetry vs segment")
        ax4.set_xlabel("segment index")
        ax4.set_ylabel("A = (L - R) / (L + R)")
        ax4.axhline(0.0, lw=1.0, alpha=0.4)
        ax4.grid(True, alpha=0.25)
        ax4.legend(loc="best", fontsize=9)

        # (E) pattern similarity stability
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(np.arange(nseg), cos_sim, marker="o", lw=1.8, label="cosine similarity")
        ax5.plot(np.arange(nseg), pearson_r, marker="s", lw=1.6, alpha=0.85, label="pearson r")
        ax5.set_title("Pattern stability vs mean (ROI)")
        ax5.set_xlabel("segment index")
        ax5.set_ylabel("similarity")
        ax5.set_ylim(-1.05, 1.05)
        ax5.grid(True, alpha=0.25)
        ax5.legend(loc="best", fontsize=9)

        fig.suptitle(
            "Integrated intensity peak stability (10 segments)\n"
            f"ROI: q=[{q_lo:.3f},{q_hi:.3f}] Å^-1, φ=[{ph_lo:.1f},{ph_hi:.1f}] deg",
            y=0.98,
            fontsize=12,
        )
        plt.show()

    return out

def fill_map_no_nans_nearest_valid(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Fill invalid pixels in m with the value from the nearest valid pixel (in x,y).
    Returns a dense map with no NaNs.

    m     : (H,W) float map, values only meaningful where valid==True
    valid : (H,W) bool mask of where m is valid

    Note: This extrapolates into detector regions where q/phi are undefined by the qmap.
    """
    m = np.asarray(m, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    if m.shape != valid.shape:
        raise ValueError(f"Shape mismatch: m={m.shape}, valid={valid.shape}")

    if not np.any(valid):
        raise RuntimeError("No valid pixels to extrapolate from (valid mask is empty).")

    # We fill invalid pixels using nearest-neighbor in pixel space.
    # SciPy is the simplest reliable way:
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        raise RuntimeError(
            "SciPy is required for nearest-valid filling (scipy.ndimage.distance_transform_edt not found)."
        ) from e

    # distance_transform_edt expects True for "background" to compute distances to False,
    # so we invert: invalid pixels are True background, valid pixels are False features.
    invalid = ~valid
    _, (iy_near, ix_near) = distance_transform_edt(invalid, return_indices=True)

    filled = m.copy()
    filled[invalid] = m[iy_near[invalid], ix_near[invalid]]

    # guarantee no NaNs remain
    if np.isnan(filled).any():
        raise RuntimeError("Filling failed: NaNs remain after nearest-valid extrapolation.")

    return filled

def build_q_phi_maps_from_geometry(
    f,
    *,
    shape: tuple[int, int] | None = None,
    phi_offset_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build full-detector per-pixel q and phi maps from geometry.

    Returns
    -------
    Q_map : (H,W) float64
        q magnitude in Angstrom^-1
    Phi_map_deg : (H,W) float64
        azimuth angle in degrees, from atan2(dy, dx), with dy positive DOWN
        (i.e. array row index increases downward).
        Range (-180, 180], then shifted by phi_offset_deg.
    """
    # --------- infer shape (H,W)
    if shape is None:
        if "xpcs/temporal_mean/scattering_2d" in f:
            img = f["xpcs/temporal_mean/scattering_2d"][...]
            if img.ndim == 3:
                shape = (int(img.shape[1]), int(img.shape[2]))
            else:
                shape = (int(img.shape[0]), int(img.shape[1]))
        elif "xpcs/qmap/static_roi_map" in f:
            rm = f["xpcs/qmap/static_roi_map"]
            shape = (int(rm.shape[0]), int(rm.shape[1]))
        else:
            raise ValueError("Provide shape=(H,W) or ensure scattering_2d or static_roi_map exists.")
    H, W = shape

    # --------- helpers to read scalars robustly
    def _read_first_existing(paths: list[str]) -> float:
        for p in paths:
            if p in f:
                v = f[p][...]
                # v might be scalar array
                return float(np.asarray(v).reshape(-1)[0])
        raise KeyError(f"None of these paths exist: {paths}")

    # beam center (pixels)
    # Prefer xpcs/qmap; fall back to entry/instrument if you ever pass the raw/metadata handle instead
    cx = _read_first_existing([
        "xpcs/qmap/beam_center_x",
        "entry/instrument/detector_1/beam_center_x",
        "entry/instrument/detector_1/beam_center_position_x",
    ])
    cy = _read_first_existing([
        "xpcs/qmap/beam_center_y",
        "entry/instrument/detector_1/beam_center_y",
        "entry/instrument/detector_1/beam_center_position_y",
    ])

    # detector distance (meters)
    dist_m = _read_first_existing([
        "xpcs/qmap/detector_distance",
        "entry/instrument/detector_1/distance",
    ])

    # pixel size (meters) – assume square if only one is provided
    # Your results.hdf shows xpcs/qmap/pixel_size exists.
    pix_m = _read_first_existing([
        "xpcs/qmap/pixel_size",
        "entry/instrument/detector_1/x_pixel_size",
    ])
    # If you ever want anisotropic pixels, we can extend to pix_x/pix_y.

    # energy (keV)
    E_keV = _read_first_existing([
        "xpcs/qmap/energy",
        "entry/instrument/incident_beam/incident_energy",
        "entry/instrument/monochromator/energy",
    ])

    # --------- physics: q = (4*pi/lambda) * sin(theta), where theta is half scattering angle
    # lambda [Angstrom] = 12.3984193 / E_keV
    lam_A = 12.3984193 / float(E_keV)
    k_Ainv = 2.0 * np.pi / lam_A

    # detector pixel grid
    # x increases to the right, y increases downward (array index convention)
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]

    dx_px = xx - float(cx)
    dy_px = yy - float(cy)

    # radial distance on detector face (meters)
    r_m = np.sqrt(dx_px * dx_px + dy_px * dy_px) * float(pix_m)

    # scattering angle: 2theta = arctan(r / dist)
    two_theta = np.arctan2(r_m, float(dist_m))
    theta = 0.5 * two_theta

    Q_map = 2.0 * k_Ainv * np.sin(theta)  # Å^-1

    # phi in degrees
    Phi_map_deg = np.degrees(np.arctan2(dy_px, dx_px))  # (-180, 180]
    if phi_offset_deg:
        Phi_map_deg = Phi_map_deg + float(phi_offset_deg)
        # keep it tidy
        Phi_map_deg = (Phi_map_deg + 180.0) % 360.0 - 180.0

    return Q_map.astype(np.float64), Phi_map_deg.astype(np.float64)

def exec_integrated_intensities_plot():

    integrated_intensities_plot(
        h5_file=h5_file,

        # --- data layout ---
        phi_fast_axis=True,  # True if flat index = iq*nphi + iphi
        # Set False if you ever discover iphi*nq + iq

        # --- display scaling ---
        map_scale="log",  # "log" or "linear"
        # Log is almost always what you want for scattering

        # --- robust color scaling for maps ---
        vmin_pct=1.0,  # lower percentile for color scaling
        vmax_pct=99.8,  # upper percentile (prevents Bragg peak blowout)

        # --- variability map scaling ---
        relstd_vmax=0.5,  # cap relative std map (None = auto 99.5%)
    )

    res = integrated_intensities_peak_stability(
        h5_file,
        q_range=(1.09, 1.13),
        phi_range_deg=(-10.0, 10.0),
        weight_power=1.0,  # keep tails physically important
        use_log_for_similarity=True,  # good dynamic range
        show_plots=True,
    )

    print("q drift:", res["drift_q"])
    print("phi drift (deg):", res["drift_phi_deg"])
    print("phi skew per segment:", res["skew_phi"])

def compare_q_skew_two_methods(
    I_mean_qphi: np.ndarray,
    I_seg_qphi: np.ndarray | None,
    q: np.ndarray,
    phi: np.ndarray,
    *,
    iphi: int | None = None,
    choose_phi: str = "argmax",  # "argmax" or "closest0"
    eps: float = 1e-12,
) -> dict:
    """
    Compare two q-skewness definitions:
      A) single-phi lineout:      w(q) = I(q, phi_fixed)
      B) phi-averaged lineout:    w(q) = mean_phi I(q, phi)

    Uses intensity-weighted central moments:
      q_mean = sum(w*q)/sum(w)
      mu2 = sum(w*(q-q_mean)^2)/sum(w)
      mu3 = sum(w*(q-q_mean)^3)/sum(w)
      skew = mu3 / (mu2^(3/2) + eps)

    Parameters
    ----------
    I_mean_qphi : (nq, nphi)
    I_seg_qphi  : (nseg, nq, nphi) or None
    q           : (nq,)
    phi         : (nphi,) in degrees
    iphi        : optional explicit phi index. If None, it is chosen by choose_phi.
    choose_phi  : if iphi is None:
                    - "argmax": pick phi index at global argmax intensity in mean map
                    - "closest0": pick phi index closest to 0 degrees
    """

    def _weighted_skew_1d(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
        x = np.asarray(x, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()

        w = np.clip(w, 0.0, None)
        sw = float(np.sum(w))
        if not np.isfinite(sw) or sw <= eps:
            return float("nan"), float("nan"), float("nan"), float("nan")

        mu = float(np.sum(w * x) / sw)
        dx = x - mu
        mu2 = float(np.sum(w * dx * dx) / sw)
        mu3 = float(np.sum(w * dx * dx * dx) / sw)
        sig = float(np.sqrt(max(mu2, 0.0)))
        skew = float(mu3 / (mu2 ** 1.5 + eps))
        return mu, sig, skew, sw

    I_mean_qphi = np.asarray(I_mean_qphi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()

    nq, nphi = I_mean_qphi.shape
    if q.size != nq or phi.size != nphi:
        raise ValueError(f"Shape mismatch: I_mean_qphi={I_mean_qphi.shape}, q={q.shape}, phi={phi.shape}")

    # ---- choose iphi if not provided ----
    if iphi is None:
        choose_phi = str(choose_phi).strip().lower()
        if choose_phi == "closest0":
            iphi = int(np.argmin(np.abs(phi - 0.0)))
        elif choose_phi == "argmax":
            flat = int(np.nanargmax(I_mean_qphi))
            _, iphi = np.unravel_index(flat, I_mean_qphi.shape)
        else:
            raise ValueError("choose_phi must be 'argmax' or 'closest0'")

    if not (0 <= int(iphi) < nphi):
        raise ValueError(f"iphi out of range: iphi={iphi}, nphi={nphi}")

    # ---- Method A: single-phi slice ----
    wA = I_mean_qphi[:, int(iphi)]
    q_mean_A, sigma_q_A, skew_q_A, swA = _weighted_skew_1d(q, wA)

    # ---- Method B: phi-averaged ----
    wB = np.nanmean(I_mean_qphi, axis=1)
    q_mean_B, sigma_q_B, skew_q_B, swB = _weighted_skew_1d(q, wB)

    print("q-skew comparison (mean map):")
    print(f"  phi index used: iphi={int(iphi)}  phi={phi[int(iphi)]:.3f} deg")
    print(f"  Method A (single-phi lineout):   q_mean={q_mean_A:.6f}  sigma_q={sigma_q_A:.6f}  skew_q={skew_q_A:.6f}")
    print(f"  Method B (phi-averaged lineout): q_mean={q_mean_B:.6f}  sigma_q={sigma_q_B:.6f}  skew_q={skew_q_B:.6f}")

    out = {
        "iphi_used": int(iphi),
        "phi_deg_used": float(phi[int(iphi)]),
        "mean_map": {
            "method_A_single_phi": {"q_mean": q_mean_A, "sigma_q": sigma_q_A, "skew_q": skew_q_A, "sum_w": swA},
            "method_B_phi_avg":    {"q_mean": q_mean_B, "sigma_q": sigma_q_B, "skew_q": skew_q_B, "sum_w": swB},
        },
    }

    # ---- per-segment comparison (optional) ----
    if I_seg_qphi is not None:
        I_seg_qphi = np.asarray(I_seg_qphi, dtype=np.float64)
        if I_seg_qphi.ndim != 3 or I_seg_qphi.shape[1:] != (nq, nphi):
            raise ValueError(f"Expected I_seg_qphi shape (nseg,{nq},{nphi}), got {I_seg_qphi.shape}")

        nseg = I_seg_qphi.shape[0]
        seg_rows = []
        for s in range(nseg):
            wA_s = I_seg_qphi[s, :, int(iphi)]
            wB_s = np.nanmean(I_seg_qphi[s, :, :], axis=1)

            q_mean_A_s, sigma_A_s, skew_A_s, _ = _weighted_skew_1d(q, wA_s)
            q_mean_B_s, sigma_B_s, skew_B_s, _ = _weighted_skew_1d(q, wB_s)

            seg_rows.append((s, q_mean_A_s, skew_A_s, q_mean_B_s, skew_B_s))

        print("\nq-skew comparison per segment:")
        print("  seg   q_mean(A)     skew(A)      q_mean(B)     skew(B)")
        for s, qmA, skA, qmB, skB in seg_rows:
            print(f"  {s:>3d}  {qmA:>10.6f}  {skA:>10.6f}   {qmB:>10.6f}  {skB:>10.6f}")

        out["per_segment"] = [
            {"seg": int(s), "q_mean_A": float(qmA), "skew_A": float(skA), "q_mean_B": float(qmB), "skew_B": float(skB)}
            for (s, qmA, skA, qmB, skB) in seg_rows
        ]

    return out

def _peak_anchor_from_map(I_mean_qphi: np.ndarray, q: np.ndarray, phi: np.ndarray) -> dict:
    """
    Find argmax anchor on the (nq, nphi) mean map.
    Returns iq0, iphi0, q0, phi0_deg, I0.
    """
    if I_mean_qphi.ndim != 2:
        raise ValueError(f"I_mean_qphi must be 2D (nq,nphi), got {I_mean_qphi.shape}")
    nq, nphi = I_mean_qphi.shape
    if q.size != nq or phi.size != nphi:
        raise ValueError(f"Axis mismatch: I_mean_qphi={I_mean_qphi.shape}, q={q.size}, phi={phi.size}")

    flat = int(np.nanargmax(I_mean_qphi))
    iq0, iphi0 = np.unravel_index(flat, I_mean_qphi.shape)
    q0 = float(q[iq0])
    phi0 = float(phi[iphi0])  # degrees
    I0 = float(I_mean_qphi[iq0, iphi0])
    return {"iq0": int(iq0), "iphi0": int(iphi0), "q0": q0, "phi0_deg": phi0, "I0": I0}


def _tail_asymmetry_about_q0(
    Iq: np.ndarray,
    q: np.ndarray,
    q0: float,
    *,
    p: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    Peak-anchored tail asymmetry A(p) about q0.
      A < 0 means heavier tail to LOWER q.
      A > 0 means heavier tail to HIGHER q.

    Uses weights w = max(Iq, 0).
    """
    Iq = np.asarray(Iq, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if Iq.size != q.size:
        raise ValueError(f"Iq and q must have same length, got {Iq.size} and {q.size}")

    w = np.clip(Iq, 0.0, None)

    dq = q - float(q0)
    left = dq < 0
    right = dq > 0

    if not np.any(left) or not np.any(right):
        return float("nan")

    dl = np.abs(dq[left]) ** float(p)
    dr = np.abs(dq[right]) ** float(p)

    ML = float(np.sum(w[left] * dl))
    MR = float(np.sum(w[right] * dr))

    denom = ML + MR
    if denom <= eps or not np.isfinite(denom):
        return float("nan")

    return float((MR - ML) / denom)


def _skew_peak_about_q0(
    Iq: np.ndarray,
    q: np.ndarray,
    q0: float,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Peak-anchored "moment skewness" about q0:
      skew_peak = sum(w*dq^3) / (sum(w*dq^2))^(3/2)
    Uses weights w = max(Iq, 0).
    """
    Iq = np.asarray(Iq, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if Iq.size != q.size:
        raise ValueError(f"Iq and q must have same length, got {Iq.size} and {q.size}")

    w = np.clip(Iq, 0.0, None)
    dq = q - float(q0)

    mu2 = float(np.sum(w * dq * dq))
    mu3 = float(np.sum(w * dq * dq * dq))

    if not np.isfinite(mu2) or mu2 <= eps:
        return float("nan")

    return float(mu3 / (mu2 ** 1.5 + eps))


def compare_peak_anchored_q_tail_metrics(
    I_mean_qphi: np.ndarray,
    I_seg_qphi: np.ndarray,
    q: np.ndarray,
    phi_deg: np.ndarray,
    *,
    p_list: tuple[float, ...] = (0.0, 1.0),
    include_skew_peak: bool = True,
) -> dict:
    """
    Compare peak-anchored q-tail metrics for:
      Method A: single-phi lineout at the argmax phi (iphi0)
      Method B: phi-averaged lineout

    Inputs
    ------
    I_mean_qphi : (nq, nphi)
    I_seg_qphi  : (nseg, nq, nphi)
    q           : (nq,)
    phi_deg     : (nphi,)  degrees

    Prints:
      - anchor info (iq0, iphi0, q0, phi0)
      - mean-map A(p) and (optional) skew_peak for A and B
      - per-segment table for the same quantities

    Returns
    -------
    dict with anchor + arrays of metrics.
    """
    I_mean_qphi = np.asarray(I_mean_qphi, dtype=np.float64)
    I_seg_qphi = np.asarray(I_seg_qphi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    phi_deg = np.asarray(phi_deg, dtype=np.float64)

    if I_mean_qphi.ndim != 2:
        raise ValueError(f"I_mean_qphi must be (nq,nphi), got {I_mean_qphi.shape}")
    if I_seg_qphi.ndim != 3:
        raise ValueError(f"I_seg_qphi must be (nseg,nq,nphi), got {I_seg_qphi.shape}")

    nq, nphi = I_mean_qphi.shape
    if q.size != nq or phi_deg.size != nphi:
        raise ValueError(
            f"Axis mismatch: I_mean_qphi={I_mean_qphi.shape}, I_seg_qphi={I_seg_qphi.shape}, "
            f"q={q.size}, phi={phi_deg.size}"
        )
    if I_seg_qphi.shape[1:] != (nq, nphi):
        raise ValueError(f"I_seg_qphi second/third dims must match mean map, got {I_seg_qphi.shape}")

    anchor = _peak_anchor_from_map(I_mean_qphi, q, phi_deg)
    iq0 = anchor["iq0"]
    iphi0 = anchor["iphi0"]
    q0 = anchor["q0"]
    phi0 = anchor["phi0_deg"]

    print("Peak anchor from mean map (argmax):")
    print(f"  iq0={iq0}, iphi0={iphi0}, q0={q0:.6f}, phi0={phi0:.3f} deg, I0={anchor['I0']:.6g}")

    # ---- mean-map lineouts ----
    Iq_A_mean = I_mean_qphi[:, iphi0]             # Method A
    Iq_B_mean = I_mean_qphi.mean(axis=1)          # Method B

    def _metrics_for_lineout(Iq: np.ndarray) -> dict:
        out = {}
        for p in p_list:
            out[f"A_p{p:g}"] = _tail_asymmetry_about_q0(Iq, q, q0, p=p)
        if include_skew_peak:
            out["skew_peak"] = _skew_peak_about_q0(Iq, q, q0)
        return out

    mA = _metrics_for_lineout(Iq_A_mean)
    mB = _metrics_for_lineout(Iq_B_mean)

    print("\nPeak-anchored q-tail metrics (mean map):")
    print(f"  Method A: single-phi lineout at iphi0={iphi0} (phi={phi0:.3f} deg)")
    for p in p_list:
        print(f"    A(p={p:g}) = {mA[f'A_p{p:g}']:+.6f}")
    if include_skew_peak:
        print(f"    skew_peak  = {mA['skew_peak']:+.6f}")

    print("  Method B: phi-averaged lineout")
    for p in p_list:
        print(f"    A(p={p:g}) = {mB[f'A_p{p:g}']:+.6f}")
    if include_skew_peak:
        print(f"    skew_peak  = {mB['skew_peak']:+.6f}")

    # ---- per-segment metrics ----
    nseg = int(I_seg_qphi.shape[0])

    A_A = {p: np.full((nseg,), np.nan, dtype=np.float64) for p in p_list}
    A_B = {p: np.full((nseg,), np.nan, dtype=np.float64) for p in p_list}
    skewA = np.full((nseg,), np.nan, dtype=np.float64) if include_skew_peak else None
    skewB = np.full((nseg,), np.nan, dtype=np.float64) if include_skew_peak else None

    for s in range(nseg):
        Iseg = I_seg_qphi[s]
        Iq_A = Iseg[:, iphi0]
        Iq_B = Iseg.mean(axis=1)

        for p in p_list:
            A_A[p][s] = _tail_asymmetry_about_q0(Iq_A, q, q0, p=p)
            A_B[p][s] = _tail_asymmetry_about_q0(Iq_B, q, q0, p=p)

        if include_skew_peak:
            skewA[s] = _skew_peak_about_q0(Iq_A, q, q0)
            skewB[s] = _skew_peak_about_q0(Iq_B, q, q0)

    print("\nPer-segment peak-anchored q-tail metrics:")
    header = "  seg"
    for p in p_list:
        header += f"   A(p={p:g})_A    A(p={p:g})_B"
    if include_skew_peak:
        header += "    skewA      skewB"
    print(header)

    for s in range(nseg):
        row = f"  {s:>3d}"
        for p in p_list:
            row += f"   {A_A[p][s]:+10.6f}  {A_B[p][s]:+10.6f}"
        if include_skew_peak:
            row += f"   {skewA[s]:+9.6f}  {skewB[s]:+9.6f}"
        print(row)

    return {
        "anchor": anchor,
        "p_list": tuple(float(p) for p in p_list),
        "mean": {"methodA": mA, "methodB": mB},
        "per_segment": {
            "A_methodA": {float(p): A_A[p] for p in p_list},
            "A_methodB": {float(p): A_B[p] for p in p_list},
            "skew_methodA": skewA,
            "skew_methodB": skewB,
        },
    }

def compare_peak_anchored_phi_tail_metrics(
    I_mean_qphi,
    I_seg_qphi,
    q,
    phi_deg,
    *,
    p_list=(0.0, 1.0),
    include_skew_peak=True,
):
    """
    Peak-anchored phi-tail asymmetry, fully analogous to q Method A.

    Uses a single-q lineout at iq0 (argmax in mean map).
    """

    Iq = I_mean_qphi.mean(axis=1)
    iq0 = int(np.argmax(Iq))

    # Mean-map lineout at the peak q
    I_phi = I_mean_qphi[iq0, :]
    iphi0 = int(np.argmax(I_phi))

    phi0 = float(phi_deg[iphi0])

    # Signed angular distance from peak (wrapped to [-180,180])
    dphi = (phi_deg - phi0 + 180.0) % 360.0 - 180.0

    def tail_asymmetry(x, I, p):
        left = I[x < 0]
        right = I[x > 0]
        xl = np.abs(x[x < 0])
        xr = np.abs(x[x > 0])

        Il = np.sum(left * xl**p)
        Ir = np.sum(right * xr**p)

        denom = Il + Ir + 1e-15
        return (Ir - Il) / denom

    out = {
        "iphi0": iphi0,
        "phi0_deg": phi0,
        "skew_peak": {},
        "per_segment": {},
    }

    # Mean map
    for p in p_list:
        out["skew_peak"][p] = tail_asymmetry(dphi, I_phi, p)

    # Per segment
    nseg = I_seg_qphi.shape[0]
    for s in range(nseg):
        Iphi_s = I_seg_qphi[s, iq0, :]
        out["per_segment"][s] = {
            p: tail_asymmetry(dphi, Iphi_s, p) for p in p_list
        }

    return out

def scroll_segments_ax0(*, log_eps: float = 1e-12, cmap: str = "magma"):
    with h5py.File(h5_file, "r") as f:
        Iseg = np.asarray(f["xpcs/temporal_mean/scattering_1d_segments"][...], dtype=np.float64)
        q = np.asarray(f["xpcs/qmap/static_v_list_dim0"][...], dtype=np.float64)
        phi = np.asarray(f["xpcs/qmap/static_v_list_dim1"][...], dtype=np.float64)

    # Sanity, reshape with phi as fast axis: flat index = iq*nphi + iphi
    nseg = int(Iseg.shape[0])
    nq = int(q.size)
    nphi = int(phi.size)
    if Iseg.ndim != 2 or Iseg.shape[1] != nq * nphi:
        raise ValueError(f"Expected Iseg shape (nseg, {nq*nphi}), got {Iseg.shape}")

    I_seg_qphi = Iseg.reshape(nseg, nq, nphi)  # (seg, q, phi)

    # For stable color scaling across segments
    I_all_pos = I_seg_qphi[I_seg_qphi > 0]
    if I_all_pos.size == 0:
        raise RuntimeError("No positive intensities found.")
    vmin = np.percentile(np.log10(I_all_pos + log_eps), 1.0)
    vmax = np.percentile(np.log10(I_all_pos + log_eps), 99.7)

    state = {"s": 0}

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.8))

    def seg_to_image(s: int):
        # ax0-style: x=q, y=phi, so image must be (phi, q)
        Iphiq = I_seg_qphi[s].T  # (phi, q)
        return np.log10(np.clip(Iphiq, 0.0, None) + log_eps)

    im = ax.imshow(
        seg_to_image(0),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[q.min(), q.max(), phi.min(), phi.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("q (Å$^{-1}$)")
    ax.set_ylabel("φ (deg)")
    title = ax.set_title(f"scattering_1d_segments, log10(I+eps), segment 0/{nseg-1}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(I + eps)")

    def redraw():
        s = state["s"]
        im.set_data(seg_to_image(s))
        title.set_text(f"scattering_1d_segments, log10(I+eps), segment {s}/{nseg-1}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["s"] = (state["s"] + 1) % nseg
            redraw()
        elif event.key in ("left", "a"):
            state["s"] = (state["s"] - 1) % nseg
            redraw()
        elif event.key in ("home",):
            state["s"] = 0
            redraw()
        elif event.key in ("end",):
            state["s"] = nseg - 1
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()



# ---- google_sheet_upload ----
def _as_1d(x) -> np.ndarray:
    return np.asarray(x).squeeze().reshape(-1)


def execute_with_backoff(request, tries: int = 6, base_delay: float = 1.0):
    for attempt in range(tries):
        try:
            return request.execute()
        except (HttpError, ConnectionResetError, TimeoutError) as e:
            if attempt == tries - 1:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.random()
            print(f"Upload failed ({type(e).__name__}), retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


# ============================================================
# Sheets scanning helpers
# ============================================================

def get_ids_for_position(ws, position_name: str, *, id_col: int = 1, position_col: int = 3, header_rows: int = 1):
    """
    Return a list of (row_number, sample_id) for rows where position_col == position_name.
    """
    col_ids = ws.col_values(id_col)
    col_pos = ws.col_values(position_col)

    n = min(len(col_ids), len(col_pos))
    out = []
    for i in range(header_rows, n):
        if col_pos[i].strip() == position_name:
            out.append((i + 1, col_ids[i].strip()))
    return out


def get_position_for_sample(ws, sample_id: str, *, id_col: int = 1, position_col: int = 3, header_rows: int = 1) -> str:
    """
    Look up the position name (e.g. 'A5') for a given sample_id (e.g. 'A013').
    """
    col_ids = ws.col_values(id_col)
    col_pos = ws.col_values(position_col)
    n = min(len(col_ids), len(col_pos))

    for i in range(header_rows, n):
        if col_ids[i].strip() == sample_id:
            return col_pos[i].strip()

    raise ValueError(f"Sample ID {sample_id} not found in spreadsheet")


def find_results_hdf_optional(base_dir: Path, sample_id: str) -> Path | None:
    pattern = f"{sample_id}_*_results.hdf"
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        # Fall back to local override path
        override = BASE_DIR_OVERRIDES.get(sample_id)
        if override is not None:
            alt_dir = override / "Twotime_PostExpt_01"
            matches = sorted(alt_dir.glob(pattern))
    return matches[0] if matches else None


# ============================================================
# Google auth + API clients
# ============================================================

def get_creds(token_path: str, creds_path: str, scopes: list[str]) -> Credentials:
    creds = None
    if Path(token_path).exists():
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, scopes)
            creds = flow.run_local_server(port=0)
            Path(token_path).write_text(creds.to_json())

    return creds


def get_ws_and_drive(creds: Credentials, spreadsheet_id: str, tab_name: str):
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(tab_name)

    authed_http = AuthorizedHttp(creds, http=httplib2.Http(timeout=300))
    drive = build("drive", "v3", http=authed_http, cache_discovery=False)

    print("Opened spreadsheet:", sh.title, "| worksheet:", ws.title)
    return ws, drive


# ============================================================
# Upload + local save helpers
# ============================================================

def upload_fig_to_cell(
    ws,
    drive,
    fig,
    cell: str,
    upload_name: str,
    *,
    upload_folder_id: str,
    dpi: int = 300,
):
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        media = MediaIoBaseUpload(buf, mimetype="image/png", resumable=True)
        req = drive.files().create(
            body={"name": upload_name, "parents": [upload_folder_id]},
            media_body=media,
            fields="id",
        )
        created = execute_with_backoff(req)
        file_id = created["id"]

        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
        formula = f'=IMAGE("{image_url}")'

        ws.update(
            range_name=cell,
            values=[[formula]],
            value_input_option="USER_ENTERED",
        )
        return file_id
    finally:
        buf.close()


def save_fig_local(fig, out_dir: Path, position_name: str, fig_key: str, sample_id: str, *, figtype_dir: dict, dpi: int):
    subdir = out_dir / position_name / figtype_dir[fig_key]
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{sample_id}.png"
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    return out_path


# ============================================================
# HDF loading + mask selection
# ============================================================

@dataclass(frozen=True)
class CommonData:
    roi_map: np.ndarray
    scat2d: np.ndarray
    g2: np.ndarray
    q_list: np.ndarray
    phi_list: np.ndarray
    stride: int


def load_common_data(hdf_path: Path) -> CommonData:
    with h5py.File(hdf_path, "r") as f:
        roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
        scat = f["xpcs/temporal_mean/scattering_2d"][...]
        scat2d = scat[0, :, :] if scat.ndim == 3 else scat

        g2 = f["xpcs/twotime/normalized_g2"][...]
        q_list = _as_1d(f["xpcs/qmap/dynamic_v_list_dim0"][...])
        phi_list = _as_1d(f["xpcs/qmap/dynamic_v_list_dim1"][...])

    stride = int(len(phi_list)) if len(phi_list) else 30
    return CommonData(roi_map=roi_map, scat2d=scat2d, g2=g2, q_list=q_list, phi_list=phi_list, stride=stride)


def load_c2_map(hdf_path: Path, mask_idx: int) -> np.ndarray:
    ttc_tree = f"xpcs/twotime/correlation_map/c2_00{int(mask_idx):03d}"
    with h5py.File(hdf_path, "r") as f:
        return f[ttc_tree][...]


# find_brightest_mask_by_integrated_intensity is defined once in the
# correlation_analysis section below (returns (label, best_sum)).


def neighborhood_offsets(n: int, *, stride: int) -> list[int]:
    """
    Offsets for an n×n neighborhood around a center index in a flattened (q,phi) grid.

    Ordering matches your process_position convention:
        [-29,   1,  31,
         -30,   0,  30,
         -31,  -1,  29]
    for n=3, stride=30.
    """
    if n % 2 == 0:
        raise ValueError("n must be odd (e.g., 3, 5).")

    r = n // 2
    offsets: list[int] = []
    for dx in range(r, -r - 1, -1):     # +phi to -phi (top to bottom)
        for dy in range(-r, r + 1):     # -q to +q (left to right)
            offsets.append(dy * int(stride) + dx)
    return offsets


def compute_neighborhood_indices(
    roi_map: np.ndarray,
    scat2d: np.ndarray,
    *,
    n_masks: int = 0,   # kept for API compat; ignored (labels discovered from roi_map)
    grid_n: int,
    stride: int,
) -> tuple[int, list[int]]:
    """
    Returns (center_mask, idxs) where center_mask is the brightest mask.
    """
    center, _ = find_brightest_mask_by_integrated_intensity(roi_map, scat2d)
    offs = neighborhood_offsets(grid_n, stride=stride)
    idxs = [center + o for o in offs]
    return center, idxs


# ============================================================
# q/phi geometry helpers for q-dependent TTC labels
# ============================================================

def qphi_for_mask(mask_idx: int, q_list: np.ndarray, phi_list: np.ndarray, *, stride: int) -> tuple[float, float, int, int]:
    """
    Flattened mask -> (iq, iphi) -> (q, phi).
    """
    stride = int(stride)
    iq = int(mask_idx) // stride
    iphi = int(mask_idx) % stride

    if iq < 0 or iq >= len(q_list):
        raise IndexError(f"iq={iq} out of range for q_list (len={len(q_list)})")
    if iphi < 0 or iphi >= len(phi_list):
        raise IndexError(f"iphi={iphi} out of range for phi_list (len={len(phi_list)})")

    return float(q_list[iq]), float(phi_list[iphi]), iq, iphi


def infer_steps_from_axis_lists(q_list: np.ndarray, phi_list: np.ndarray) -> tuple[float, float]:
    uq = np.sort(np.unique(np.asarray(q_list, float)))
    up = np.sort(np.unique(np.asarray(phi_list, float)))

    uq = uq[np.isfinite(uq)]
    up = up[np.isfinite(up)]

    if uq.size < 2:
        raise ValueError("Not enough q values to infer dq step.")
    if up.size < 2:
        raise ValueError("Not enough phi values to infer dphi step.")

    dq_step = float(np.median(np.diff(uq)))
    dphi_step_deg = float(np.median(np.diff(up)))
    return dq_step, dphi_step_deg


def _minmax_from_corners(xlo: float, xhi: float, ylo: float, yhi: float) -> tuple[float, float]:
    """
    Min/max of sqrt(x^2+y^2) over rectangle corners.
    """
    vals = [
        float(np.hypot(xlo, ylo)),
        float(np.hypot(xlo, yhi)),
        float(np.hypot(xhi, ylo)),
        float(np.hypot(xhi, yhi)),
    ]
    return min(vals), max(vals)


def length_scale_nm_from_qinvA(q_invA: float) -> float:
    """
    ℓ = 2π/q. Convert Å -> nm by *0.1.
    """
    q = float(q_invA)
    if not np.isfinite(q) or q <= 0:
        return np.nan
    return (2.0 * np.pi / q) * 0.1


def label_ranges_for_mask(
    mask_idx: int,
    *,
    center_mask: int,
    q_list: np.ndarray,
    phi_list: np.ndarray,
    stride: int,
    dq_step: float,
    dphi_step_deg: float,
) -> dict:
    """
    Build ranges for:
      - dq_radial in Å^-1 (relative to center)
      - dq_tangential in Å^-1 (≈ q0*Δphi_rad, relative to center)
      - |Δq| magnitude in Å^-1 (pythagorean, using corner extremes)
      - length-scale range nm via ℓ = 2π/|Δq|
    """
    q0, phi0, iq0, iphi0 = qphi_for_mask(center_mask, q_list, phi_list, stride=stride)
    q_m, phi_m, iq_m, iphi_m = qphi_for_mask(mask_idx, q_list, phi_list, stride=stride)

    # Center offsets in axis units
    dq_center = q_m - q0
    dphi_center_deg = phi_m - phi0

    # Each mask occupies a half-bin in q and phi
    dq_lo = dq_center - 0.5 * dq_step
    dq_hi = dq_center + 0.5 * dq_step

    dphi_lo_deg = dphi_center_deg - 0.5 * dphi_step_deg
    dphi_hi_deg = dphi_center_deg + 0.5 * dphi_step_deg

    # Tangential q shift: q0 * Δphi (radians)
    dqt_lo = q0 * np.deg2rad(dphi_lo_deg)
    dqt_hi = q0 * np.deg2rad(dphi_hi_deg)

    # magnitude of Δq over rectangle corners
    dmag_lo, dmag_hi = _minmax_from_corners(dq_lo, dq_hi, dqt_lo, dqt_hi)

    # Convert to length scales: ℓ = 2π/|Δq|
    # dmag_lo can be 0 near center => ℓ_hi = inf
    if dmag_hi <= 0 or not np.isfinite(dmag_hi):
        ell_lo_nm = np.nan
    else:
        ell_lo_nm = length_scale_nm_from_qinvA(dmag_hi)  # smallest length at largest |Δq|

    if dmag_lo <= 0 or not np.isfinite(dmag_lo):
        ell_hi_nm = np.inf
    else:
        ell_hi_nm = length_scale_nm_from_qinvA(dmag_lo)

    return dict(
        q0=q0,
        phi0=phi0,
        q=q_m,
        phi=phi_m,
        iq=iq_m,
        iphi=iphi_m,
        dq_lo=dq_lo, dq_hi=dq_hi,
        dqt_lo=dqt_lo, dqt_hi=dqt_hi,
        dmag_lo=dmag_lo, dmag_hi=dmag_hi,
        ell_lo_nm=ell_lo_nm, ell_hi_nm=ell_hi_nm,
    )


# ============================================================
# Plot builders
# ============================================================

def make_overview_fig(
    sample_id: str,
    roi_map: np.ndarray,
    scat2d: np.ndarray,
    idxs: list[int],
    *,
    title: str,
    half_crop: int = 200,
):
    combined_mask = np.isin(roi_map, idxs)

    fig, ax = plt.subplots()

    # Boost neighborhood masks
    I = scat2d.astype(float, copy=False).copy()
    I[combined_mask] *= 10.0

    # Crop around neighborhood centroid
    ys, xs = np.where(combined_mask)
    cy = int(np.round(ys.mean())) if ys.size else I.shape[0] // 2
    cx = int(np.round(xs.mean())) if xs.size else I.shape[1] // 2

    ymin = max(cy - half_crop, 0)
    ymax = min(cy + half_crop, I.shape[0])
    xmin = max(cx - half_crop, 0)
    xmax = min(cx + half_crop, I.shape[1])

    Icrop = I[ymin:ymax, xmin:xmax]
    Mcrop = roi_map[ymin:ymax, xmin:xmax]

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    Ishow = Icrop.copy()
    Ishow[Ishow <= 0] = 0.0
    Ishow_ma = np.ma.masked_less_equal(Ishow, 0.0)
    vmax = float(Ishow_ma.max()) if Ishow_ma.count() else 1.0

    im = ax.imshow(
        Ishow_ma,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_facecolor("black")

    # Single-pass borders around masks in idxs
    M = Mcrop
    in_neigh = np.isin(M, idxs)

    boundary = np.zeros_like(M, dtype=bool)
    dv = (M[1:, :] != M[:-1, :])
    tv = in_neigh[1:, :] | in_neigh[:-1, :]
    boundary[1:, :] |= dv & tv

    dh = (M[:, 1:] != M[:, :-1])
    th = in_neigh[:, 1:] | in_neigh[:, :-1]
    boundary[:, 1:] |= dh & th

    overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=float)
    overlay[boundary] = (0.0, 0.0, 0.0, 1.0)  # black
    ax.imshow(overlay, origin="upper", interpolation="nearest")

    ax.set_title(f"{sample_id} {title}")
    ax.axis("off")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig


def make_g2s_fig(sample_id: str, g2: np.ndarray, idxs: list[int], *, title: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    x = np.arange(g2.shape[0])

    for mi in idxs:
        j = mi - 1  # your convention
        if 0 <= j < g2.shape[1]:
            ax.semilogx(x, g2[:, j], label=f"M{mi}")

    ax.set_title(f"{sample_id} {title}")
    ax.set_ylabel("g2(q,τ)")
    ax.set_xlabel("Delay time τ (index)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def make_twotime_grid_fig(
    sample_id: str,
    hdf_path: Path,
    idxs: list[int],
    *,
    grid_n: int,
    figsize: tuple[float, float],
    clip_hi_percentile: float = 99.9,
    textbox_fontsize: int = 10,
    suptitle: str = "",
):
    """
    Plain TTC grid with textbox per panel, no per-panel titles.
    """
    fig, axes = plt.subplots(grid_n, grid_n, figsize=figsize)
    axes = np.array(axes).reshape(grid_n, grid_n)

    for k, ax in enumerate(axes.flat):
        mi = int(idxs[k])
        C = load_c2_map(hdf_path, mi)
        C = symmetrize_ttc(C)
        Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))

        ax.imshow(Cplot, origin="lower", cmap="plasma", interpolation="nearest")
        ax.axis("off")

        txt = f"M{mi}\nmin={np.nanmin(C):.3g}\nmax={np.nanmax(C):.3g}"
        ax.text(
            0.04, 0.96,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=textbox_fontsize,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    if suptitle:
        fig.suptitle(f"{sample_id} {suptitle}", fontsize=18)
    fig.tight_layout()
    return fig


# ============================================================
# NEW: q-dependent TTC grid plot
# ============================================================

def plot_q_dependent_ttc(
    *,
    sample_id: str,
    base_dir: Path,
    out_dir: Path,
    ws,                      # used to map sample_id -> position folder structure
    grid_n: int = 5,          # 3 or 5
    n_masks: int = 300,
    clip_hi_percentile: float = 99.9,
    textbox_fontsize: int = 10,
    dpi: int = 200,
    out_name: str | None = None,
) -> Path:
    """
    Make a TTC grid (3x3 or 5x5) around the brightest mask, with textbox labels showing:
      - mask index
      - Δq_radial range [Å^-1]
      - Δq_tangential range [Å^-1]  (q0*Δphi)
      - |Δq| range [Å^-1] (pythagorean/corner bounds)
      - length-scale range [nm] via ℓ=2π/|Δq|
    """
    if grid_n not in (3, 5):
        raise ValueError("grid_n must be 3 or 5")

    position_name = get_position_for_sample(ws, sample_id)
    hdf_path = find_results_hdf_optional(base_dir, sample_id)
    if hdf_path is None:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")

    common = load_common_data(hdf_path)
    stride = int(common.stride)

    # brightest mask + neighborhood (ordering matches process_position)
    center_mask, idxs = compute_neighborhood_indices(
        common.roi_map,
        common.scat2d,
        n_masks=n_masks,
        grid_n=grid_n,
        stride=stride,
    )

    # offset = -60
    # center_mask = center_mask + offset
    # new_idxs = [i + offset for i in idxs]
    # idxs = new_idxs
    # print(center_mask)
    # print(idxs)

    dq_step, dphi_step_deg = infer_steps_from_axis_lists(common.q_list, common.phi_list)

    # build grid
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(12, 12) if grid_n == 5 else (7.2, 7.2))
    axes = np.array(axes).reshape(grid_n, grid_n)

    # Center mask q0/phi0 for reference (only used implicitly in labels)
    q0, phi0, *_ = qphi_for_mask(center_mask, common.q_list, common.phi_list, stride=stride)

    for k, ax in enumerate(axes.flat):
        mi = int(idxs[k])

        # load TTC
        C = load_c2_map(hdf_path, mi)
        C = symmetrize_ttc(C)
        Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))

        ax.imshow(Cplot, origin="lower", cmap="plasma", interpolation="nearest")
        ax.axis("off")

        # label ranges
        rr = label_ranges_for_mask(
            mi,
            center_mask=center_mask,
            q_list=common.q_list,
            phi_list=common.phi_list,
            stride=stride,
            dq_step=dq_step,
            dphi_step_deg=dphi_step_deg,
        )

        # pretty formatting
        def fmt_nm(x):
            if x == np.inf:
                return "∞"
            if not np.isfinite(x):
                return "nan"
            return f"{x:.3g}"

        txt = (
            f"q={rr['q']:.4g} Å⁻¹\n"
            f"φ={rr['phi']:.4g}°\n"
            f"ℓ=[{fmt_nm(rr['ell_lo_nm'])},{fmt_nm(rr['ell_hi_nm'])}]nm"
        )

        ax.text(
            0.04, 0.96,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=textbox_fontsize,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    # one small overall title only (optional). If you want *none*, comment this out.
    fig.suptitle(
        f"{sample_id} q-dependent TTC grid (center=M{center_mask}, q0={q0:.4g} Å⁻¹, φ0={phi0:.4g}°)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()

    # save
    out_base = out_dir / "q_dependent_ttc" / f"position_{position_name}" / sample_id
    out_base.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = f"{sample_id}_qdep_ttc_ctx{grid_n}x{grid_n}.png"

    out_path = out_base / out_name
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.show()
    # plt.close(fig)

    return out_path


# ============================================================
# Plot selection / filtering helpers
# ============================================================

def normalize_keys(keys: Optional[Iterable[str]], all_keys: set[str]) -> set[str]:
    return set(all_keys) if keys is None else set(keys)


def should_generate(key: str, generate_keys: Optional[Iterable[str]], all_keys: set[str]) -> bool:
    return key in normalize_keys(generate_keys, all_keys)


def should_upload(
    key: str,
    upload_enabled: bool,
    upload_keys: Optional[Iterable[str]],
    generate_keys: Optional[Iterable[str]],
    all_keys: set[str],
) -> bool:
    if not upload_enabled:
        return False
    if not should_generate(key, generate_keys, all_keys):
        return False
    if upload_keys is None:
        return True
    return key in set(upload_keys)


# ============================================================
# Main processing pipeline
# ============================================================

def process_one_scan(
    sample_id: str,
    row: int,
    hdf_path: Path,
    ws,
    drive,
    *,
    position_name: str,
    out_dir: Path,
    generate_keys=None,
    upload_enabled: bool = True,
    upload_keys=None,
    # config dicts
    figtype_dir: dict[str, str],
    plot_cols: dict[str, str],
    dpi_by_plot: dict[str, int],
    all_plot_keys: set[str],
    upload_folder_id: str,
):
    gen = normalize_keys(generate_keys, all_plot_keys)
    common = load_common_data(hdf_path)

    need_9 = any(k in gen for k in ("overview_9", "g2s_9", "twotime_9"))
    need_25 = any(k in gen for k in ("overview_25", "g2s_25", "twotime_25"))

    idxs_9 = idxs_25 = None
    if need_9:
        _, idxs_9 = compute_neighborhood_indices(common.roi_map, common.scat2d, n_masks=300, grid_n=3, stride=common.stride)
    if need_25:
        _, idxs_25 = compute_neighborhood_indices(common.roi_map, common.scat2d, n_masks=300, grid_n=5, stride=common.stride)

    figs = {}

    # 9-mask set
    if "overview_9" in gen:
        figs["overview_9"] = make_overview_fig(sample_id, common.roi_map, common.scat2d, idxs_9, title="9-mask overview")
    if "g2s_9" in gen:
        figs["g2s_9"] = make_g2s_fig(sample_id, common.g2, idxs_9, title="9-mask g2")
    if "twotime_9" in gen:
        figs["twotime_9"] = make_twotime_grid_fig(
            sample_id, hdf_path, idxs_9,
            grid_n=3, figsize=(7, 7),
            suptitle="9-mask TTC",
            textbox_fontsize=10,
        )

    # 25-mask set
    if "overview_25" in gen:
        figs["overview_25"] = make_overview_fig(sample_id, common.roi_map, common.scat2d, idxs_25, title="25-mask overview")
    if "g2s_25" in gen:
        figs["g2s_25"] = make_g2s_fig(sample_id, common.g2, idxs_25, title="25-mask g2")
    if "twotime_25" in gen:
        figs["twotime_25"] = make_twotime_grid_fig(
            sample_id, hdf_path, idxs_25,
            grid_n=5, figsize=(12, 12),
            suptitle="25-mask TTC",
            textbox_fontsize=10,
        )

    try:
        for key, fig in figs.items():
            dpi = int(dpi_by_plot[key])

            # local save
            local_path = save_fig_local(fig, out_dir, position_name, key, sample_id, figtype_dir=figtype_dir, dpi=dpi)

            # upload
            if should_upload(key, upload_enabled, upload_keys, generate_keys, all_plot_keys):
                cell = f"{plot_cols[key]}{row}"
                upload_name = f"{sample_id}_{key}.png"
                upload_fig_to_cell(ws, drive, fig, cell, upload_name, upload_folder_id=upload_folder_id, dpi=dpi)
                print(f"Saved + uploaded: {local_path}  →  {ws.title}!{cell}")
            else:
                print(f"Saved local: {local_path}")

    finally:
        for fig in figs.values():
            plt.close(fig)


def process_position(
    position_name: str,
    base_dir: Path,
    ws,
    drive,
    *,
    out_dir: Path,
    generate_keys=None,
    upload_enabled: bool = True,
    upload_keys=None,
    start_sample_id: str | None = None,
    start_row: int | None = None,
    start_index: int = 0,
    # config dicts
    figtype_dir: dict[str, str],
    plot_cols: dict[str, str],
    dpi_by_plot: dict[str, int],
    all_plot_keys: set[str],
    upload_folder_id: str,
):
    rows_and_ids = get_ids_for_position(ws, position_name)
    print(f"Found {len(rows_and_ids)} scans at position {position_name}")

    # decide where to start
    if start_row is not None:
        rows_and_ids = [(r, sid) for (r, sid) in rows_and_ids if r >= start_row]
    elif start_sample_id is not None:
        start_pos = None
        for i, (r, sid) in enumerate(rows_and_ids):
            if sid == start_sample_id:
                start_pos = i
                break
        if start_pos is None:
            raise ValueError(f"start_sample_id={start_sample_id} not found in position {position_name}")
        rows_and_ids = rows_and_ids[start_pos:]
    else:
        rows_and_ids = rows_and_ids[start_index:]

    print(f"Starting from: {rows_and_ids[0] if rows_and_ids else 'nothing to do'}")

    for row, sample_id in rows_and_ids:
        hdf_path = find_results_hdf_optional(base_dir, sample_id)
        if hdf_path is None:
            print(f"SKIP: no HDF file found for {sample_id}")
            continue

        print(f"Processing {sample_id} (row {row})")
        process_one_scan(
            sample_id=sample_id,
            row=row,
            hdf_path=hdf_path,
            ws=ws,
            drive=drive,
            position_name=position_name,
            out_dir=out_dir,
            generate_keys=generate_keys,
            upload_enabled=upload_enabled,
            upload_keys=upload_keys,
            figtype_dir=figtype_dir,
            plot_cols=plot_cols,
            dpi_by_plot=dpi_by_plot,
            all_plot_keys=all_plot_keys,
            upload_folder_id=upload_folder_id,
        )


# ============================================================
# plot_single_mask_scan (kept structure as you wanted)
# ============================================================

def plot_single_mask_scan(
    *,
    sample_id: str,
    mask_n: int,
    base_dir: Path,
    out_dir: Path,
    ws,                 # used to map sample_id -> position
    grid_n: int = 5,     # 3 or 5
    n_masks: int = 300,
    dpi: int = 250,
    figsize=(18, 5.5),
    stride: int | None = None,
    border_width: float = 1.5,
    half_crop: int = 220,
    out_name: str | None = None,
    # highlight controls
    neigh_boost: float = 10.0,
    other_dim: float = 0.35,
    highlight_boost: float = 25.0,
    highlight_outline: bool = True,
    outline_rgba=(1.0, 1.0, 1.0, 1.0),
    ttc_cbar_size: str | float = "6%",
):
    """
    One combined figure for a single scan + single mask:
      [overview with neighborhood borders (selected mask highlighted) | g2 for mask_n | TTC for mask_n]

    ttc_cbar_size : str or float
        Width of the TTC colorbar, e.g. "4%" or "8%" (string) or 0.06 (fraction of axes width). Default "4%".

    Saves to:
      out_dir/individual_scan_plots/position_<POS>/<SAMPLE_ID>/<SAMPLE_ID>_mask_<mask>_ctxNxN.png
    """
    if grid_n not in (3, 5):
        raise ValueError("grid_n must be 3 or 5")

    position_name = get_position_for_sample(ws, sample_id)

    hdf_path = find_results_hdf_optional(base_dir, sample_id)
    if hdf_path is None:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")

    common = load_common_data(hdf_path)
    stride_eff = int(stride) if stride is not None else int(common.stride)

    # context neighborhood around brightest mask (for borders)
    _, idxs = compute_neighborhood_indices(common.roi_map, common.scat2d, n_masks=n_masks, grid_n=grid_n, stride=stride_eff)

    # load TTC for this mask
    C = load_c2_map(hdf_path, mask_n)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.10, 1.15], wspace=0.35)

    # ----------------------------
    # (1) Overview
    # ----------------------------
    ax0 = fig.add_subplot(gs[0])
    I = common.scat2d.astype(float, copy=False).copy()

    neigh = np.isin(common.roi_map, idxs)
    sel = (common.roi_map == mask_n)

    I[neigh] *= neigh_boost
    I[neigh & ~sel] *= other_dim
    I[sel] *= highlight_boost

    ys, xs = np.where(neigh)
    if ys.size == 0 or xs.size == 0:
        cy, cx = I.shape[0] // 2, I.shape[1] // 2
    else:
        cy = int(np.round(ys.mean()))
        cx = int(np.round(xs.mean()))

    ymin = max(cy - half_crop, 0)
    ymax = min(cy + half_crop, I.shape[0])
    xmin = max(cx - half_crop, 0)
    xmax = min(cx + half_crop, I.shape[1])

    Icrop = I[ymin:ymax, xmin:xmax]
    Mcrop = common.roi_map[ymin:ymax, xmin:xmax]

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    Ishow = np.ma.masked_less_equal(Icrop, 0.0)
    vmax = float(Ishow.max()) if Ishow.count() else 1.0

    im0 = ax0.imshow(
        Ishow,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=vmax),
        interpolation="nearest",
    )
    ax0.set_facecolor("black")

    in_neigh = np.isin(Mcrop, idxs)
    boundary = np.zeros_like(Mcrop, dtype=bool)
    boundary[1:, :] |= (Mcrop[1:, :] != Mcrop[:-1, :]) & (in_neigh[1:, :] | in_neigh[:-1, :])
    boundary[:, 1:] |= (Mcrop[:, 1:] != Mcrop[:, :-1]) & (in_neigh[:, 1:] | in_neigh[:, :-1])

    if border_width and border_width > 1:
        N = int(round(border_width)) - 1
        b = boundary.copy()
        for _ in range(N):
            b2 = b.copy()
            b2[1:, :] |= b[:-1, :]
            b2[:-1, :] |= b[1:, :]
            b2[:, 1:] |= b[:, :-1]
            b2[:, :-1] |= b[:, 1:]
            b = b2
        boundary = b

    overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=float)
    overlay[boundary] = (0.0, 0.0, 0.0, 1.0)
    ax0.imshow(overlay, origin="upper", interpolation="nearest")

    if highlight_outline:
        sel_crop = (Mcrop == mask_n)
        sel_b = np.zeros_like(sel_crop, dtype=bool)
        sel_b[1:, :] |= (sel_crop[1:, :] != sel_crop[:-1, :])
        sel_b[:, 1:] |= (sel_crop[:, 1:] != sel_crop[:, :-1])

        if border_width and border_width > 1:
            N = int(round(border_width)) - 1
            b = sel_b.copy()
            for _ in range(N):
                b2 = b.copy()
                b2[1:, :] |= b[:-1, :]
                b2[:-1, :] |= b[1:, :]
                b2[:, 1:] |= b[:, :-1]
                b2[:, :-1] |= b[:, 1:]
                b = b2
            sel_b = b

        sel_overlay = np.zeros((sel_b.shape[0], sel_b.shape[1], 4), dtype=float)
        sel_overlay[sel_b] = outline_rgba
        ax0.imshow(sel_overlay, origin="upper", interpolation="nearest")

    ax0.set_title(f"{sample_id} overview (M{mask_n} highlighted, ctx {grid_n}×{grid_n})")
    ax0.axis("off")

    div0 = make_axes_locatable(ax0)
    cax0 = div0.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(im0, cax=cax0)

    # ----------------------------
    # (2) g2 for mask_n
    # ----------------------------
    ax1 = fig.add_subplot(gs[1])
    tau = np.arange(common.g2.shape[0])
    j = mask_n - 1
    if 0 <= j < common.g2.shape[1]:
        ax1.semilogx(tau, common.g2[:, j], lw=2)
        # map mask to q/phi:
        try:
            q_m, phi_m, *_ = qphi_for_mask(mask_n, common.q_list, common.phi_list, stride=stride_eff)
            ax1.set_title(f"g2 for M{mask_n}\nq={q_m:.3f} Å⁻¹, φ={phi_m:.3f}°")
        except Exception:
            ax1.set_title(f"g2 for M{mask_n}")
    else:
        ax1.set_title(f"g2 for M{mask_n} (out of range)")

    ax1.set_xlabel("Delay time τ (index)")
    ax1.set_ylabel("g2(τ)", labelpad=10)
    ax1.grid(True, alpha=0.3)

    # ----------------------------
    # (3) TTC for mask_n
    # ----------------------------
    ax2 = fig.add_subplot(gs[2])
    C = symmetrize_ttc(C)
    Cplot = clip_ttc(C, p_hi=99.9)
    cmin, cmax = np.nanmin(Cplot), np.nanmax(Cplot)
    if cmax > cmin and np.isfinite(cmin) and np.isfinite(cmax):
        Cplot = (Cplot - cmin) / (cmax - cmin)
    else:
        Cplot = np.full_like(Cplot, 0.5)

    im2 = ax2.imshow(Cplot, origin="lower", cmap="plasma", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_title(f"TTC for M{mask_n}")
    ax2.set_xlabel("t₁")
    ax2.set_ylabel("t₂")

    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size=ttc_cbar_size, pad=0.8)
    fig.colorbar(im2, cax=cax2)
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_tick_params(labelleft=True, labelright=False)
    line_width = 1.5
    for spine in cax2.spines.values():
        spine.set_linewidth(line_width)
    cax2.tick_params(axis="y", width=line_width, length=4, labelsize=30)

    ax2.text(
        0.05, 0.95,
        f"M{mask_n}\nmin {np.nanmin(C):.2f}\nmax {np.nanmax(C):.2f}",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=12,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none"),
    )

    # save
    out_base = out_dir / "individual_scan_plots" / f"position_{position_name}" / sample_id
    out_base.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = f"{sample_id}_mask_{mask_n:03d}_ctx{grid_n}x{grid_n}.png"

    out_path = out_base / out_name
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.show()
    # plt.close(fig)

    return out_path


# ============================================================
# Bragg peak center (argmax) + T from filename for t_dep_xrd
# ============================================================

def temperature_k_from_filename(hdf_path: Path) -> Optional[int]:
    """
    Extract temperature in Kelvin from results HDF filename (e.g. ..._260K_... → 260).
    """
    match = re.search(r"(\d+)K", hdf_path.name)
    return int(match.group(1)) if match else None


def bragg_peak_center_argmax(hdf_path: Path, *, phi_fast_axis: bool = True) -> tuple[float, float]:
    """
    Find Bragg peak (q0, phi0) from temporal-mean scattering_1d using argmax,
    same method as integrated_intensities_plot() in analysis_for_aps_08-ide-2025-1006.py.
    Returns (q0, phi0_deg).
    """
    with h5py.File(hdf_path, "r") as f:
        I1d = np.asarray(f["xpcs/temporal_mean/scattering_1d"][...])
        q = _as_1d(f["xpcs/qmap/static_v_list_dim0"][...])
        phi = _as_1d(f["xpcs/qmap/static_v_list_dim1"][...])
    nq, nphi = int(q.size), int(phi.size)
    if I1d.ndim != 2 or I1d.shape[0] != 1 or nq * nphi != int(I1d.shape[1]):
        raise ValueError(
            f"scattering_1d shape {I1d.shape} does not match q.size={nq} * phi.size={nphi}"
        )
    if phi_fast_axis:
        I_mean_qphi = I1d[0].reshape(nq, nphi)
    else:
        I_mean_phiq = I1d[0].reshape(nphi, nq)
        I_mean_qphi = np.transpose(I_mean_phiq, (1, 0))
    iq0, iphi0 = np.unravel_index(int(np.nanargmax(I_mean_qphi)), I_mean_qphi.shape)
    q0 = float(q[iq0])
    phi0 = float(phi[iphi0])
    return q0, phi0


def bragg_peak_center_argmax_second_moment_uncertainty(
    hdf_path: Path,
    *,
    phi_fast_axis: bool = True,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """
    Bragg peak (q0, phi0) from argmax; uncertainties from intensity-weighted second moments
    on the mean (q, φ) map. Center = argmax; σ_q, σ_φ = sqrt(weighted variance) with w = I.
    Returns (q0, phi0, sigma_q, sigma_phi).
    """
    with h5py.File(hdf_path, "r") as f:
        I1d = np.asarray(f["xpcs/temporal_mean/scattering_1d"][...])
        q = _as_1d(f["xpcs/qmap/static_v_list_dim0"][...])
        phi = _as_1d(f["xpcs/qmap/static_v_list_dim1"][...])
    nq, nphi = int(q.size), int(phi.size)
    if I1d.ndim != 2 or I1d.shape[0] != 1 or nq * nphi != int(I1d.shape[1]):
        raise ValueError(
            f"scattering_1d shape {I1d.shape} does not match q.size={nq} * phi.size={nphi}"
        )
    if phi_fast_axis:
        I_mean_qphi = I1d[0].reshape(nq, nphi)
    else:
        I_mean_phiq = I1d[0].reshape(nphi, nq)
        I_mean_qphi = np.transpose(I_mean_phiq, (1, 0))

    iq0, iphi0 = np.unravel_index(int(np.nanargmax(I_mean_qphi)), I_mean_qphi.shape)
    q0 = float(q[iq0])
    phi0 = float(phi[iphi0])

    w = np.clip(np.asarray(I_mean_qphi, dtype=np.float64), 0.0, None)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= eps:
        uq = np.sort(np.unique(np.asarray(q, dtype=np.float64)))
        uphi = np.sort(np.unique(np.asarray(phi, dtype=np.float64)))
        dq = float(np.min(np.diff(uq))) / 2.0 if uq.size >= 2 else 1e-5
        dphi = float(np.min(np.diff(uphi))) / 2.0 if uphi.size >= 2 else 0.5
        return q0, phi0, dq, dphi

    q_2d = np.asarray(q, dtype=np.float64)[:, np.newaxis]  # (nq, nphi)
    phi_2d = np.asarray(phi, dtype=np.float64)[np.newaxis, :]  # (nq, nphi)
    q_bar = float(np.sum(w * q_2d) / sw)
    phi_bar = float(np.sum(w * phi_2d) / sw)
    var_q = float(np.sum(w * (q_2d - q_bar) ** 2) / sw)
    var_phi = float(np.sum(w * (phi_2d - phi_bar) ** 2) / sw)
    sigma_q = np.sqrt(max(var_q, 0.0))
    sigma_phi = np.sqrt(max(var_phi, 0.0))

    uq = np.sort(np.unique(np.asarray(q, dtype=np.float64)))
    uphi = np.sort(np.unique(np.asarray(phi, dtype=np.float64)))
    dq_half = float(np.min(np.diff(uq))) / 2.0 if uq.size >= 2 else 1e-5
    dphi_half = float(np.min(np.diff(uphi))) / 2.0 if uphi.size >= 2 else 0.5
    sigma_q = max(sigma_q, dq_half)
    sigma_phi = max(sigma_phi, dphi_half)
    return q0, phi0, sigma_q, sigma_phi


# ============================================================
# Execution functions
# ============================================================

def exec_google_sheet_upload():
    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    # Examples:
    # process_position("A6", BASE_DIR, ws, drive, out_dir=OUT_DIR, ...)
    # process_position(POSITION_NAME, BASE_DIR, ws, drive, out_dir=OUT_DIR, start_sample_id="A031", ...)

    process_position(
        POSITION_NAME,
        RESULTS_BASE_DIR,
        ws,
        drive,
        out_dir=OUT_DIR,
        generate_keys=GENERATE_KEYS,
        upload_enabled=UPLOAD_TO_SHEETS,
        upload_keys=UPLOAD_KEYS,
        figtype_dir=FIGTYPE_DIR,
        plot_cols=PLOT_COLS,
        dpi_by_plot=DPI_BY_PLOT,
        all_plot_keys=ALL_PLOT_KEYS,
        upload_folder_id=UPLOAD_FOLDER_ID,
    )


def exec_single_mask_plot_save():
    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, _drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    plot_single_mask_scan(
        sample_id=SAMPLE_ID,
        mask_n=MASK_N,
        base_dir=RESULTS_BASE_DIR,
        out_dir=OUT_DIR,
        ws=ws,
        grid_n=5,
        border_width=1,
        dpi=250,
    )


def mask_mesh_around_bright_peak(
    *,
    sample_id: str | None = None,
    base_dir: Path | None = None,
    half_crop: int = 150,
    border_width: float = 1.0,
    dpi: int = 150,
    out_path: Path | str | None = None,
) -> Path | None:
    """
    Plot a 300×300-pixel crop around the brightest part of the scattering image,
    with all mask boundaries outlined (no neighborhood filter).

    Similar to the first subfigure of exec_single_mask_plot_save(), but:
      - Crop is centered on the brightest mask (same logic).
      - All masks in the crop are outlined (not just a 5×5 neighborhood).

    Parameters
    ----------
    sample_id : str | None
        Sample/scan ID used to find the results HDF. Defaults to SAMPLE_ID.
    base_dir : Path | None
        Base directory to search for <sample_id>_*_results.hdf. Defaults to RESULTS_BASE_DIR.
    half_crop : int
        Half-width of the square crop in pixels (default 150 → 300×300).
    border_width : float
        Line width for mask boundaries (default 1.0).
    dpi : int
        Figure DPI for saved image (default 150).
    out_path : Path | str | None
        If set, save the figure to this path; otherwise show interactively.

    Returns
    -------
    Path | None
        out_path if the figure was saved, else None.
    """
    if sample_id is None:
        sample_id = SAMPLE_ID
    if base_dir is None:
        base_dir = RESULTS_BASE_DIR

    hdf_path = find_results_hdf_optional(base_dir, sample_id)
    if hdf_path is None:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")

    common = load_common_data(hdf_path)
    center_mask, _ = find_brightest_mask_by_integrated_intensity(common.roi_map, common.scat2d)

    I = common.scat2d.astype(float, copy=False).copy()
    roi_map = common.roi_map

    ys, xs = np.where(roi_map == center_mask)
    if ys.size == 0 or xs.size == 0:
        cy, cx = I.shape[0] // 2, I.shape[1] // 2
    else:
        cy = int(np.round(ys.mean()))
        cx = int(np.round(xs.mean()))

    ymin = max(cy - half_crop, 0)
    ymax = min(cy + half_crop, I.shape[0])
    xmin = max(cx - half_crop, 0)
    xmax = min(cx + half_crop, I.shape[1])

    Icrop = I[ymin:ymax, xmin:xmax]
    Mcrop = roi_map[ymin:ymax, xmin:xmax]

    cmap = plt.cm.plasma.copy()
    cmap.set_under("black")
    cmap.set_bad("black")

    Ishow = np.ma.masked_less_equal(Icrop, 0.0)
    vmax = float(Ishow.max()) if Ishow.count() else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        Ishow,
        origin="upper",
        cmap=cmap,
        norm=LogNorm(vmin=0.1, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_facecolor("black")

    in_crop = (Mcrop > 0)
    boundary = np.zeros_like(Mcrop, dtype=bool)
    boundary[1:, :] |= (Mcrop[1:, :] != Mcrop[:-1, :]) & (in_crop[1:, :] | in_crop[:-1, :])
    boundary[:, 1:] |= (Mcrop[:, 1:] != Mcrop[:, :-1]) & (in_crop[:, 1:] | in_crop[:, :-1])

    if border_width and border_width > 1:
        N = int(round(border_width)) - 1
        b = boundary.copy()
        for _ in range(N):
            b2 = b.copy()
            b2[1:, :] |= b[:-1, :]
            b2[:-1, :] |= b[1:, :]
            b2[:, 1:] |= b[:, :-1]
            b2[:, :-1] |= b[:, 1:]
            b = b2
        boundary = b

    overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=float)
    overlay[boundary] = (0.0, 0.0, 0.0, 1.0)
    ax.imshow(overlay, origin="upper", interpolation="nearest")

    ax.set_title(f"{sample_id} mask mesh (M{center_mask} center, {2*half_crop}×{2*half_crop} px)")
    ax.axis("off")

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.show()
    return None


def exec_mask_mesh_around_bright_peak(
    *,
    half_crop: int = 250,
    border_width: float = 1.0,
    dpi: int = 150,
    out_path: Path | str | None = None,
) -> Path | None:
    """
    Entrypoint: run mask_mesh_around_bright_peak with SAMPLE_ID and RESULTS_BASE_DIR.
    Options (half_crop, border_width, dpi, out_path) can be overridden.
    """
    return mask_mesh_around_bright_peak(
        sample_id=SAMPLE_ID,
        base_dir=RESULTS_BASE_DIR,
        half_crop=half_crop,
        border_width=border_width,
        dpi=dpi,
        out_path=out_path,
    )


def _g2_stretched_model(tau: np.ndarray, g2_inf: float, beta: float, tau0: float, gamma: float) -> np.ndarray:
    """g2(τ) = g2_inf + β exp(-2(τ/τ0)^γ). g2_inf = baseline (~1.2); stretched if γ < 1, compressed if γ > 1."""
    return g2_inf + beta * np.exp(-2.0 * (tau / tau0) ** gamma)


def g2_fitting_plot(
    *,
    sample_id: str | None = None,
    base_dir: Path | None = None,
    mask_n: int | None = None,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Load g2 for a mask (same source as exec_single_mask_plot_save middle subfigure)
    and plot g2(τ) vs τ (log x).

    Parameters
    ----------
    sample_id : str | None
        Sample/scan ID. Defaults to SAMPLE_ID.
    base_dir : Path | None
        Base directory for results HDF. Defaults to RESULTS_BASE_DIR.
    mask_n : int | None
        Mask number (1-indexed). Defaults to MASK_N.
    out_path : Path | str | None
        If set, save figure; otherwise show.
    dpi : int
        DPI when saving (default 150).

    Returns
    -------
    Path | None
        out_path if saved, else None.
    """
    if sample_id is None:
        sample_id = SAMPLE_ID
    if base_dir is None:
        base_dir = RESULTS_BASE_DIR
    if mask_n is None:
        mask_n = MASK_N

    hdf_path = find_results_hdf_optional(base_dir, sample_id)
    if hdf_path is None:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")

    common = load_common_data(hdf_path)
    stride_eff = int(common.stride)

    tau = np.arange(common.g2.shape[0])
    j = mask_n - 1

    fig, ax = plt.subplots(figsize=(6, 4))
    y_max = None
    if 0 <= j < common.g2.shape[1]:
        y = common.g2[:, j]
        tau = tau[1:-1]
        y = y[1:-1]
        ax.semilogx(tau, y, lw=2, label="g2", color="C0")
        y_max = float(np.nanmax(y))
        # Stretched/compressed exponential fit: g2(τ) = g2_inf + β exp(-2(τ/τ0)^γ)
        mask_fit = np.isfinite(tau) & np.isfinite(y) & (tau > 0)
        if np.sum(mask_fit) >= 5:
            tau_f = tau[mask_fit].astype(float)
            y_f = y[mask_fit].astype(float)
            g2_inf_init = 1.2  # baseline (long-τ) from data
            beta_init = max(0.01, float(np.nanmax(y_f)) - g2_inf_init)
            tau0_init = 130.0
            p0 = (g2_inf_init, beta_init, tau0_init, 1.0)
            lb = (1.0, 1e-6, 1.0, 0.2)
            ub = (1.5, 1.0, float(np.max(tau_f)) * 1.5, 3.0)
            try:
                popt, _ = curve_fit(
                    _g2_stretched_model,
                    tau_f,
                    y_f,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=5000,
                )
                tau_smooth = np.logspace(np.log10(max(1e-6, float(tau.min()))), np.log10(float(tau.max())), 200)
                y_fit = _g2_stretched_model(tau_smooth, *popt)
                ax.semilogx(tau_smooth, y_fit, "--", lw=1.5, label=f"fit g2∞={popt[0]:.2f}, β={popt[1]:.3f}, τ0={popt[2]:.1f}, γ={popt[3]:.3f}", color="C1")
            except Exception:
                pass
        try:
            q_m, phi_m, *_ = qphi_for_mask(mask_n, common.q_list, common.phi_list, stride=stride_eff)
            ax.set_title(f"{sample_id}: g2 for M{mask_n}\nq={q_m:.3f} Å⁻¹, φ={phi_m:.3f}°")
        except Exception:
            ax.set_title(f"{sample_id}: g2 for M{mask_n}")
    else:
        ax.set_title(f"{sample_id}: g2 for M{mask_n} (out of range)")


    ax.set_xlabel("Delay time τ (index)")
    ax.set_ylabel("g2(τ)", labelpad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if y_max is not None and np.isfinite(y_max):
        ax.set_autoscaley_on(False)
        ax.set_ylim(0, y_max * 1.10)
        ax.margins(y=0)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.show()
    return None


def exec_g2_fitting(
    *,
    mask_n: int | None = None,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Entrypoint: plot g2 for config mask (SAMPLE_ID, RESULTS_BASE_DIR, MASK_N).
    Override mask_n, out_path, or dpi as needed.
    """
    return g2_fitting_plot(
        sample_id=SAMPLE_ID,
        base_dir=RESULTS_BASE_DIR,
        mask_n=mask_n if mask_n is not None else MASK_N,
        out_path=out_path,
        dpi=dpi,
    )


def exec_q_dependent_ttc_plot():
    """
    New entrypoint for the q-dependent TTC grid.
    """
    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, _drive = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)

    out_path = plot_q_dependent_ttc(
        sample_id=SAMPLE_ID,
        base_dir=RESULTS_BASE_DIR,
        out_dir=OUT_DIR,
        ws=ws,
        grid_n=5,              # 3 or 5
        n_masks=300,
        clip_hi_percentile=99.9,
        textbox_fontsize=10,
        dpi=200,
    )
    print("Saved:", out_path)


def t_dep_xrd_argmax(
    *,
    position_name: str | None = None,
    base_dir: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """
    For all samples at a position: find Bragg peak (q0, phi0) via argmax on scattering_1d,
    uncertainties from intensity-weighted second moments (σ_q, σ_φ), get temperature
    from the results HDF filename, then plot q0 and phi0 vs T with error bars.
    Uses module CONFIG (POSITION_NAME, RESULTS_BASE_DIR, OUT_DIR) when arguments are None.
    """
    base_dir = base_dir or RESULTS_BASE_DIR
    position_name = position_name or POSITION_NAME
    out_dir = out_dir or OUT_DIR

    creds = get_creds(TOKEN_PATH, CREDS_PATH, SCOPES)
    ws, _ = get_ws_and_drive(creds, SPREADSHEET_ID, TAB_NAME)
    rows_and_ids = get_ids_for_position(ws, position_name)
    if not rows_and_ids:
        print(f"No sample IDs found for position {position_name}")
        return

    T_list: list[int] = []
    q0_list: list[float] = []
    phi0_list: list[float] = []
    q0_err_list: list[float] = []
    phi0_err_list: list[float] = []
    sample_ids_used: list[str] = []

    for _row, sample_id in rows_and_ids:
        hdf_path = find_results_hdf_optional(base_dir, sample_id)
        if hdf_path is None:
            print(f"SKIP: no HDF for {sample_id}")
            continue
        T = temperature_k_from_filename(hdf_path)
        if T is None:
            print(f"SKIP: no temperature in filename for {sample_id} ({hdf_path.name})")
            continue
        try:
            q0, phi0, q0_err, phi0_err = bragg_peak_center_argmax_second_moment_uncertainty(hdf_path)
        except Exception as e:
            print(f"SKIP: {sample_id} bragg_peak_center_argmax_second_moment_uncertainty failed: {e}")
            continue
        T_list.append(T)
        q0_list.append(q0)
        phi0_list.append(phi0)
        q0_err_list.append(q0_err)
        phi0_err_list.append(phi0_err)
        sample_ids_used.append(sample_id)

    if not T_list:
        print("No data to plot.")
        return

    # sort by T for clean curve
    order = np.argsort(T_list)
    T_arr = np.array(T_list)[order]
    q0_arr = np.array(q0_list)[order]
    phi0_arr = np.array(phi0_list)[order]
    q0_err_arr = np.array(q0_err_list)[order]
    phi0_err_arr = np.array(phi0_err_list)[order]

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.errorbar(T_arr[0:-1], q0_arr[0:-1], yerr=q0_err_arr[0:-1], fmt="o-", color="C0", capsize=3)
    ax1.set_ylabel("q₀ (Å⁻¹)")
    ax1.set_xlabel("T (K)")
    ax1.set_title("Bragg peak q vs temperature")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(T_arr[0:-1], phi0_arr[0:-1], yerr=phi0_err_arr[0:-1], fmt="s-", color="C1", capsize=3)
    ax2.set_ylabel("φ₀ (deg)")
    ax2.set_xlabel("T (K)")
    ax2.set_title("Bragg peak φ vs temperature")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Position {position_name}", y=0.98)
    fig.tight_layout()

    out_base = out_dir / "t_dep_xrd"
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = out_base / f"bragg_vs_T_{position_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved:", out_path)

    plt.show()


# ================================================================== #
# ---- correlation_analysis ----                                      #
# Merged from correlation_analysis.py                                 #
# ================================================================== #

# ---- Data loading (correlation-specific) ----

@dataclass
class XPCSData:
    dynamic_roi_map: np.ndarray
    scattering_2d: np.ndarray
    ttc: np.ndarray
    g2: np.ndarray


def find_results_hdf_direct(base_dir: Path, sample_id: str) -> Path:
    """
    Find <sample_id>_*_results.hdf directly inside base_dir (no subdirectory).
    Falls back to BASE_DIR_OVERRIDES[sample_id]/Twotime_PostExpt_01/ if the
    primary base_dir has no match (e.g. volume disconnected, data is local).
    """
    pattern = f"{sample_id}_*_results.hdf"
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        # Try override path (local data)
        override = BASE_DIR_OVERRIDES.get(sample_id)
        if override is not None:
            alt_dir = override / "Twotime_PostExpt_01"
            matches = sorted(alt_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No results HDF found for {sample_id} in {base_dir}")
    return matches[0]


def load_xpcs_arrays(
    sample_id: str,
    base_dir: Path,
    *,
    mask_n: int,
    scattering_first_frame_only: bool = True,
) -> XPCSData:
    """
    Load 4 arrays from a processed results HDF:
        dynamic_roi_map, scattering_2d, ttc, g2
    Returns XPCSData dataclass.
    """
    hdf_path = find_results_hdf_direct(base_dir, sample_id)
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


# ---- Spatial scaling calculator ----

def spatial_scaling_calculator(
    center_mask: int,
    relative_mask: int,
    xray_energy_keV: float,
    *,
    q_step: float = 0.004,
    phi_step_rad: float = 0.01216,
    stride: int = 30,
    print_result: bool = True,
) -> tuple[float, float]:
    """
    Calculate the min/max spatial scales (in nm) probed by a relative mask
    compared to a central mask.

    The calculation uses the small-angle approximation dq = (2π/λ) × Δθ for
    the tangential component, and direct Δq for the radial component.

    Parameters
    ----------
    center_mask : int
        Central mask number (1-indexed label, 1-300)
    relative_mask : int
        Relative mask number (1-indexed label, 1-300)
    xray_energy_keV : float
        X-ray energy in keV
    q_step : float
        Q-bin spacing in Å⁻¹ (default 0.004)
    phi_step_rad : float
        Phi-bin spacing in radians (default 0.01216, i.e. ~0.697°)
    stride : int
        Number of phi bins per q row (default 30)
    print_result : bool
        If True, print the results to stdout

    Returns
    -------
    tuple[float, float]
        (d_min_nm, d_max_nm) - min and max spatial scales in nm
    """
    center_idx = center_mask - 1
    rel_idx = relative_mask - 1

    iq_center = center_idx // stride
    iphi_center = center_idx % stride

    iq_rel = rel_idx // stride
    iphi_rel = rel_idx % stride

    delta_iq = iq_rel - iq_center
    delta_iphi = iphi_rel - iphi_center

    delta_q_center = delta_iq * q_step
    delta_phi_center = delta_iphi * phi_step_rad

    q_half = q_step / 2
    phi_half = phi_step_rad / 2

    hc = 12.39842  # keV·Å
    wavelength = hc / xray_energy_keV

    dq_radial_min = max(0.0, abs(delta_q_center) - q_half)
    dphi_min = max(0.0, abs(delta_phi_center) - phi_half)
    dq_tan_min = (2 * np.pi / wavelength) * dphi_min
    dq_min = np.sqrt(dq_radial_min**2 + dq_tan_min**2)

    dq_radial_max = abs(delta_q_center) + q_half
    dphi_max = abs(delta_phi_center) + phi_half
    dq_tan_max = (2 * np.pi / wavelength) * dphi_max
    dq_max = np.sqrt(dq_radial_max**2 + dq_tan_max**2)

    if dq_min > 0:
        d_max_nm = (2 * np.pi / dq_min) / 10
    else:
        d_max_nm = np.inf
    d_min_nm = (2 * np.pi / dq_max) / 10

    if print_result:
        print(f"|Δq| range: {dq_min:.6f} to {dq_max:.6f} Å⁻¹")
        print(f"Spatial scale range: {d_min_nm:.1f} nm to {d_max_nm:.1f} nm")
        print(f"  Center mask: {center_mask} (iq={iq_center}, iphi={iphi_center})")
        print(f"  Relative mask: {relative_mask} (iq={iq_rel}, iphi={iphi_rel})")
        print(f"  Offset: Δiq={delta_iq}, Δiphi={delta_iphi}")

    return d_min_nm, d_max_nm


def spatial_scale_demo_plot(
    xray_energy_keV: float = 12.4,
    *,
    q_step: float = 0.004,
    phi_step_rad: float = 0.01216,
    stride: int = 30,
    d_max_cap_nm: float = 1000.0,
) -> None:
    """
    Create a 5x5 grid visualization showing mask offsets and their spatial scales.

    The central square represents the reference mask (M0). Surrounding squares
    show relative mask offsets (e.g., M-1, M+29) with their corresponding
    |Δq| ranges and spatial scale ranges.

    Parameters
    ----------
    xray_energy_keV : float
        X-ray energy in keV (default 12.4)
    q_step : float
        Q-bin spacing in Å⁻¹ (default 0.004)
    phi_step_rad : float
        Phi-bin spacing in radians (default 0.01216)
    stride : int
        Number of phi bins per q row (default 30)
    d_max_cap_nm : float
        Cap for maximum spatial scale in nm (default 1000 = 1 μm)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Spatial Scale Grid (E = {xray_energy_keV} keV)", fontsize=14, fontweight="bold")

    center_mask = 150
    hc = 12.39842
    wavelength = hc / xray_energy_keV
    q_half = q_step / 2
    phi_half = phi_step_rad / 2

    for row in range(5):
        for col in range(5):
            x = col
            y = 4 - row

            iq_offset = col - 2
            iphi_offset = -(row - 2)
            mask_offset = iq_offset * stride + iphi_offset

            rect = mpatches.FancyBboxPatch(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)

            if mask_offset == 0:
                label = "M0"
                dq_min = 0.0
                dq_max = np.sqrt(q_half**2 + ((2 * np.pi / wavelength) * phi_half)**2)
                d_min_nm = (2 * np.pi / dq_max) / 10
                d_max_nm = d_max_cap_nm
                q_text = f"|Δq|: 0 – {dq_max:.3g} Å⁻¹"
                d_text = f"d: {d_min_nm:.3g} nm – 1 μm"
            else:
                if mask_offset > 0:
                    label = f"M+{mask_offset}"
                else:
                    label = f"M{mask_offset}"

                relative_mask = center_mask + mask_offset
                delta_q_center = iq_offset * q_step
                delta_phi_center = iphi_offset * phi_step_rad

                dq_radial_min = max(0.0, abs(delta_q_center) - q_half)
                dphi_min = max(0.0, abs(delta_phi_center) - phi_half)
                dq_tan_min = (2 * np.pi / wavelength) * dphi_min
                dq_min = np.sqrt(dq_radial_min**2 + dq_tan_min**2)

                dq_radial_max = abs(delta_q_center) + q_half
                dphi_max = abs(delta_phi_center) + phi_half
                dq_tan_max = (2 * np.pi / wavelength) * dphi_max
                dq_max = np.sqrt(dq_radial_max**2 + dq_tan_max**2)

                d_min_nm = (2 * np.pi / dq_max) / 10
                if dq_min > 0:
                    d_max_nm = (2 * np.pi / dq_min) / 10
                else:
                    d_max_nm = d_max_cap_nm

                q_text = f"|Δq|: {dq_min:.3g} – {dq_max:.3g} Å⁻¹"
                d_text = f"d: {d_min_nm:.3g} – {d_max_nm:.3g} nm"

            ax.text(x, y + 0.25, label, ha="center", va="center", fontsize=12, fontweight="bold")
            ax.text(x, y, q_text, ha="center", va="center", fontsize=8)
            ax.text(x, y - 0.22, d_text, ha="center", va="center", fontsize=8)

    ax.text(2, -0.8, "← lower q       higher q →", ha="center", va="center", fontsize=10)
    ax.text(-0.8, 2, "← lower φ       higher φ →", ha="center", va="center", fontsize=10, rotation=90)

    plt.tight_layout()
    plt.show()


# ---- TTC preprocessing + utilities ----

def despike_patch_with_local_median(C: np.ndarray, center: tuple[int, int], halfwidth: int = 1) -> np.ndarray:
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


def arrow_endpoint_to_edge(start_x: int, start_y: int, dx: int, dy: int, n: int) -> tuple[int, int]:
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


# ---- Lineout extraction helpers ----

def extract_antidiagonal_lineout(
    ttc: np.ndarray,
    *,
    start_idx: int,
    dt_s: float,
    clip_percentile: float | None = 99.9,
):
    """Extract anti-diagonal lineout through (i,i) and return (t, y)."""
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


def extract_antidiagonal_lineout_y_only(Csym: np.ndarray, start_idx: int, *, drop_first: int = 0):
    """Extract y along anti-diagonal through (i,i). Returns y (optionally dropping first points)."""
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


def extract_horizontal_lineout_y_only(
    C: np.ndarray,
    start_idx: int,
    *,
    drop_first: int = 0,
) -> np.ndarray:
    """Horizontal lineout at fixed t2=i, from t1=0..i (ends at the diagonal)."""
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    i = int(start_idx)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"TTC must be square, got {C.shape}")
    if not (0 <= i < n):
        raise ValueError(f"start_idx must be in [0, {n-1}]")
    y = C[i, 0:i + 1]
    if drop_first > 0:
        y = y[int(drop_first):]
    return y.astype(np.float64)


def _extract_antidiagonal_y_only_corr(
    C: np.ndarray,
    start_idx: int,
    *,
    drop_first: int = 0,
) -> np.ndarray:
    """Anti-diagonal through (t1=i, t2=i): (t1=i-k, t2=i+k), k>=0. C indexed as C[t2, t1]."""
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
    y = C[t2, t1]
    if drop_first > 0:
        y = y[int(drop_first):]
    return y.astype(np.float64)


# ---- FFT / spectral helpers ----

def estimate_fft(t: np.ndarray, y: np.ndarray, *, drop_first: int = 0, detrend: bool = True, window: bool = True):
    """Returns freqs (Hz), power, f_peak (Hz), period (s)."""
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
        power[0] = 0.0
    k = int(np.argmax(power)) if len(power) else 0
    f_peak = float(freqs[k]) if len(freqs) else 0.0
    period = (1.0 / f_peak) if f_peak > 0 else np.inf
    return freqs, power, f_peak, period


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
    Returns freqs, power, f_peak, period_s, p_peak.
    """
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if N < 8:
        raise ValueError("Need at least ~8 points for a meaningful FFT")

    x = y - np.mean(y)
    if detrend:
        t = np.arange(N, dtype=np.float64)
        a, b = np.polyfit(t, x, 1)
        x = x - (a * t + b)

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
    F = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=dt_s)
    W = np.sum(w**2)
    power = (np.abs(F) ** 2) / W

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


def segment_indices(n: int, seg_len: int, overlap: float) -> list[tuple[int, int]]:
    step = max(1, int(round(seg_len * (1 - overlap))))
    idx = []
    for s in range(0, n - seg_len + 1, step):
        idx.append((s, s + seg_len))
    return idx


def corr_periodogram(y: np.ndarray, dt: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    n = len(y)
    w = window_fn(window, n)
    yw = (y - np.mean(y)) * w
    Y = np.fft.rfft(yw)
    P = (np.abs(Y) ** 2)
    f = np.fft.rfftfreq(n, d=dt)
    return f, P


def peak_frequency_from_psd(f: np.ndarray, P: np.ndarray, fmin: float, fmax: float) -> float:
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        raise ValueError("No frequencies in band")
    i = np.argmax(P[m])
    return f[m][i]


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
    Returns f_peak, f_lo, f_hi, period, period_lo, period_hi, df.
    """
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if N < 8:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    freqs, power, f_peak, period_s, p_peak = fft_peak_from_lineout(
        y, dt_fft, detrend=detrend, window=window, fmin=fmin, fmax=fmax,
    )

    df = 1.0 / (N * dt_fft)
    f_lo = f_peak - 0.5 * df
    f_hi = f_peak + 0.5 * df

    if fmin is not None:
        f_lo = max(f_lo, fmin)
        f_hi = max(f_hi, fmin)
    if fmax is not None:
        f_lo = min(f_lo, fmax)
        f_hi = min(f_hi, fmax)

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
    """Bootstrap CI for peak frequency. Returns f_hat, f_lo, f_hi, f_samples."""
    y = np.asarray(y, float)
    if detrend:
        y = detrend_linear(y)
    n = len(y)
    if seg_len is None:
        seg_len = max(64, n // 4)
    seg_len = min(seg_len, n)

    idx = segment_indices(n, seg_len, overlap)
    if len(idx) < 3:
        f, P = corr_periodogram(y, dt, window=window)
        f0 = peak_frequency_from_psd(f, P, fmin, fmax)
        return f0, f0, f0, np.array([f0])

    f_ref = None
    Ps = []
    for (a, b) in idx:
        f, P = corr_periodogram(y[a:b], dt, window=window)
        if f_ref is None:
            f_ref = f
        Ps.append(P)
    Ps = np.stack(Ps, axis=0)
    f_ref = np.asarray(f_ref)

    rng = np.random.default_rng(rng_seed)
    K = Ps.shape[0]
    f_samp = np.empty(n_boot, float)
    for i in range(n_boot):
        picks = rng.integers(0, K, size=K)
        Pmean = Ps[picks].mean(axis=0)
        f_samp[i] = peak_frequency_from_psd(f_ref, Pmean, fmin, fmax)

    f_hat = float(np.median(f_samp))
    alpha = (1 - ci) / 2
    f_lo = float(np.quantile(f_samp, alpha))
    f_hi = float(np.quantile(f_samp, 1 - alpha))
    return f_hat, f_lo, f_hi, f_samp


def bootstrap_peak_frequency_fixedbin(
    y: np.ndarray,
    dt: float,
    *,
    fmin: float,
    fmax: float,
    peak_halfwidth_bins: int = 2,
    seg_len: int | None = None,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: bool = True,
    n_boot: int = 2000,
    ci: float = 0.68,
    rng_seed: int = 0,
):
    """Bootstrap CI for peak frequency, restricted to neighborhood of global peak."""
    y = np.asarray(y, float)
    if detrend:
        y = detrend_linear(y)
    n = len(y)
    if seg_len is None:
        seg_len = max(64, n // 4)
    seg_len = min(seg_len, n)

    idx = segment_indices(n, seg_len, overlap)
    if len(idx) < 3:
        f, P = corr_periodogram(y, dt, window=window)
        m = (f >= fmin) & (f <= fmax)
        if not np.any(m):
            raise ValueError("No frequencies in band")
        k0 = np.argmax(P[m])
        f0 = float(f[m][k0])
        return f0, f0, f0, np.array([f0])

    f_full, P_full = corr_periodogram(y, dt, window=window)
    band = (f_full >= fmin) & (f_full <= fmax)
    if not np.any(band):
        raise ValueError("No frequencies in band")
    band_idxs = np.flatnonzero(band)
    k_band_peak = band_idxs[np.argmax(P_full[band])]
    k0 = int(k_band_peak)

    k_lo = max(1, k0 - int(peak_halfwidth_bins))
    k_hi = min(len(f_full) - 1, k0 + int(peak_halfwidth_bins))
    neigh = np.arange(k_lo, k_hi + 1)

    Ps = []
    for (a, b) in idx:
        f_seg, P_seg = corr_periodogram(y[a:b], dt, window=window)
        Ps.append(P_seg)
    Ps = np.stack(Ps, axis=0)

    rng = np.random.default_rng(rng_seed)
    K = Ps.shape[0]
    f_samp = np.empty(n_boot, float)
    for i in range(n_boot):
        picks = rng.integers(0, K, size=K)
        Pmean = Ps[picks].mean(axis=0)
        kk = neigh[np.argmax(Pmean[neigh])]
        f_samp[i] = float(f_full[kk])

    f_hat = float(np.median(f_samp))
    alpha = (1.0 - float(ci)) / 2.0
    f_lo = float(np.quantile(f_samp, alpha))
    f_hi = float(np.quantile(f_samp, 1.0 - alpha))
    return f_hat, f_lo, f_hi, f_samp


def _rolling_smooth_with_band(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling smooth: y_smooth (mean), lo_smooth (min), hi_smooth (max)."""
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


# ---- Damped cosine fitting ----

def fit_damped_cosine_with_linear_baseline(
    t: np.ndarray,
    y: np.ndarray,
    omega_grid: np.ndarray,
    tau_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Fit: y(t) = C + b*t + exp(-t/tau)*(a*cos(omega*t) + s*sin(omega*t))
    For fixed (omega, tau), (C, b, a, s) are solved by weighted least squares.
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

    best: dict = {"sse": np.inf}
    for tau in tau_grid:
        tau = float(tau)
        if tau <= 0:
            continue
        e = np.exp(-t / tau)
        for omega in omega_grid:
            omega = float(omega)
            coswt = np.cos(omega * t)
            sinwt = np.sin(omega * t)
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
                phi = float(np.arctan2(-s, a))
                best = dict(C=float(C), b=float(b), a=float(a), s=float(s),
                            R=R, phi=phi, omega=omega, tau=tau, yhat=yhat, sse=sse)
    return best


def evaluate_model_damped(t: np.ndarray, C: float, b: float, omega: float, tau: float, a: float, s: float) -> np.ndarray:
    t = np.asarray(t, float)
    e = np.exp(-t / tau)
    return C + b * t + e * (a * np.cos(omega * t) + s * np.sin(omega * t))


# ---- Lineout + fitting plots ----

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
    """Left: symmetrized TTC with arrows for lineouts. Right: lineout curves."""
    C = symmetrize_ttc(data.ttc)
    n = C.shape[0]
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"TTC must be square, got shape {C.shape}")
    if not (0 <= start < n):
        raise ValueError(f"start must be in [0, {n-1}]")

    if despike_at_start:
        C = despike_patch_with_local_median(C, center=(start, start), halfwidth=despike_halfwidth)

    x_h = np.arange(start, n)
    y_h = C[start, start:n]
    x_v = np.arange(start, -1, -1)
    y_v = C[start::-1, start]
    x_d = np.arange(start, n)
    y_d = np.diag(C)[start:n]

    lineouts = [
        dict(x=x_h, y=y_h, color="tab:blue",   label="horizontal (→)"),
        dict(x=x_v, y=y_v, color="tab:orange", label="vertical (↓)"),
        dict(x=x_d, y=y_d, color="tab:green",  label="diag x=y (↗)"),
    ]

    if add_antidiag_se:
        kmax = min(n - 1 - start, start)
        xs = start + np.arange(0, kmax + 1)
        ys = start - np.arange(0, kmax + 1)
        y_ad = C[ys, xs]
        lineouts.append(dict(x=xs, y=y_ad, color="tab:purple", label="anti-diag (↘)"))

    Cplot = C.copy()
    if clip_percentile is not None:
        Cplot = clip_ttc(Cplot, p_hi=float(clip_percentile))

    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title("TTC with lineouts")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    arrows = [
        ("horizontal", (1, 0),  "tab:blue"),
        ("vertical",   (0, -1), "tab:orange"),
        ("diag x=y",   (1, 1),  "tab:green"),
    ]
    if add_antidiag_se:
        arrows.append(("anti-diag", (1, -1), "tab:purple"))

    for _, (dx, dy), col in arrows:
        ex, ey = arrow_endpoint_to_edge(start, start, dx, dy, n)
        ax0.annotate("", xy=(ex, ey), xytext=(start, start),
                     arrowprops=dict(arrowstyle="->", lw=3, color=col))

    ax0.plot([start], [start], marker="o", markersize=4, color="white", zorder=5)
    fig.colorbar(im, ax=ax0, fraction=0.046)

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
) -> dict:
    """Extract anti-diagonal lineout + fit damped cosine + plot TTC and fit."""
    data = load_xpcs_arrays(sample_id, base_dir, mask_n=mask_n)

    C = symmetrize_ttc(data.ttc)
    if despike_at_start:
        C = despike_patch_with_local_median(C, center=(start_time_idx, start_time_idx), halfwidth=despike_halfwidth)
    C = clip_ttc(C, p_hi=clip_hi_percentile)

    N = C.shape[0]
    i = int(start_time_idx)
    if not (0 <= i < N):
        raise ValueError(f"start_time_idx must be in [0, {N-1}]")

    kmax = min(i, N - 1 - i)
    ks = np.arange(0, kmax + 1, dtype=int)
    t1 = i - ks
    t2 = i + ks
    y = C[t2, t1]
    tau_idx = 2 * ks
    t = tau_idx.astype(np.float64) * float(dt_s)

    omega_grid = np.linspace(float(omega_min), float(omega_max), int(n_omega))

    if tau_min is None:
        tau_min = max(2.0 * dt_s, 0.02 * (t.max() if t.size else 1.0))
    if tau_max is None:
        tau_max = max(5.0 * tau_min, 2.0 * (t.max() if t.size else 1.0))
    tau_grid = np.geomspace(float(tau_min), float(tau_max), int(n_tau))

    dt_fft = 2.0 * dt_s
    freqs, power, f_peak, period_s, p_peak = fft_peak_from_lineout(
        y, dt_fft, detrend=True, window="hann", fmin=1 / 1000, fmax=1 / 10,
    )
    print(f"FFT peak: f = {f_peak:.4g} Hz  (period ≈ {period_s:.2f} s)")

    weights = None
    if use_weights and t.size:
        weights = np.exp(-t / (0.7 * t.max()))

    fit = fit_damped_cosine_with_linear_baseline(t, y, omega_grid, tau_grid, weights=weights)

    period = (2 * np.pi / fit["omega"]) if fit["omega"] != 0 else np.inf
    print("Fit parameters (baseline + damped cosine):")
    print(f"  C      = {fit['C']:.6g}")
    print(f"  b      = {fit['b']:.6g}")
    print(f"  tau    = {fit['tau']:.6g} s")
    print(f"  omega  = {fit['omega']:.6g} rad/s  (period ≈ {period:.2f} s)")
    print(f"  R      = {fit['R']:.6g}")
    print(f"  phi    = {fit['phi']:.6g} rad")
    print(f"  SSE    = {fit['sse']:.6g}")

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.35}
    )

    im = ax0.imshow(C, origin="lower", cmap="plasma", interpolation="nearest")
    ax0.set_title(f"{sample_id}  M{mask_n}  TTC")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    end_t1 = i - kmax
    end_t2 = i + kmax
    ax0.annotate("", xy=(end_t2, end_t1), xytext=(i, i),
                 arrowprops=dict(arrowstyle="->", lw=2.8, color="tab:purple"))
    ax0.plot(i, i, "o", color="white", ms=6)
    fig.colorbar(im, ax=ax0, fraction=0.046)

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
    """Left: TTC with anti-diagonal arrow. Right-top: lineout. Right-bottom: FFT power."""
    C = symmetrize_ttc(data.ttc)
    Cplot = clip_ttc(C, p_hi=clip_hi_percentile)

    t, y = extract_antidiagonal_lineout(
        data.ttc, start_idx=start_idx, dt_s=dt_s, clip_percentile=clip_hi_percentile,
    )
    N = C.shape[0]
    i = int(start_idx)
    kmax = min(i, N - 1 - i)

    freqs, power, f_peak, period = estimate_fft(
        t, y, drop_first=drop_first, detrend=detrend, window=window
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], wspace=0.35, hspace=0.35)

    ax0 = fig.add_subplot(gs[:, 0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title(f"TTC, start t1=t2={start_idx}")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")

    end_t1 = i - kmax
    end_t2 = i + kmax
    ax0.add_patch(FancyArrowPatch(
        (i, i), (end_t2, end_t1),
        arrowstyle="->", linewidth=3, mutation_scale=14, color=arrow_color,
    ))
    ax0.plot([i], [i], marker="o", markersize=5, color="white", zorder=5)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, y, lw=2, color=arrow_color, label="anti-diagonal lineout")
    ax1.set_title("Anti-diagonal lineout")
    ax1.set_xlabel("τ (s)")
    ax1.set_ylabel("TTC value")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(freqs, power, lw=1.8, color="black")
    ax2.set_title(f"FFT power (peak period ≈ {period:.2f} s)" if np.isfinite(period) else "FFT power")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_yscale("log")
    ax2.set_xlim([-0.001, 0.01])
    ax2.grid(True, alpha=0.3)
    if f_peak > 0:
        ax2.axvline(f_peak, lw=1.5, color=arrow_color, alpha=0.9)

    plt.show()
    return {"t": t, "y": y, "freqs": freqs, "power": power, "f_peak_hz": f_peak, "period_s": period}


# ---- Period vs diagonal start plots ----

def plot_period_vs_diagonal_start(
    data,
    *,
    dt_s: float,
    start_idxs: np.ndarray | None = None,
    clip_hi_percentile: float = 99.9,
    drop_first_lineout: int = 0,
    drop_first_horizontal: int = 0,
    half_window: int = 5,
    band_ci: float = 0.68,
    fmin: float | None = 1/1000,
    fmax: float | None = 1/10,
    detrend: bool = True,
    window: str = "hann",
    cmap: str = "plasma",
    figsize=(13, 5.5),
):
    """Left: TTC with diagonal start positions. Right: smoothed peak period vs start time."""
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

    dt_fft = 2.0 * float(dt_s)

    raw_period = np.full(start_idxs.shape, np.nan, dtype=float)
    for k, i in enumerate(start_idxs):
        y = extract_antidiagonal_lineout_y_only(C, int(i), drop_first=drop_first_lineout)
        if y.size < 16:
            continue
        f_peak, f_lo, f_hi, period, period_lo, period_hi, df = fft_peak_with_bin_uncertainty(
            y, dt_fft, fmin=fmin, fmax=fmax, detrend=detrend, window=window,
        )
        if np.isfinite(period) and period > 0:
            raw_period[k] = period

    half_window = int(max(0, half_window))
    smooth_period = np.full_like(raw_period, np.nan)
    band_lo = np.full_like(raw_period, np.nan)
    band_hi = np.full_like(raw_period, np.nan)
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
        smooth_period[k] = float(np.median(window_vals))
        band_lo[k] = float(np.quantile(window_vals, q_lo))
        band_hi[k] = float(np.quantile(window_vals, q_hi))

    starts_s = start_idxs.astype(float) * float(dt_s)
    m = np.isfinite(smooth_period)
    starts_s_m = starts_s[m]
    smooth_m = smooth_period[m]
    blo_m = band_lo[m]
    bhi_m = band_hi[m]
    start_idxs_used = start_idxs[m]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_title(SAMPLE_ID + " mask " + str(MASK_N) + " TTC + diagonal start positions")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")
    ax0.plot(start_idxs_used, start_idxs_used, "o", ms=3.5, color="C2", alpha=0.9)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(starts_s_m, smooth_m, lw=2.4, color="C2", label=f"median over ±{half_window}")
    mm = np.isfinite(blo_m) & np.isfinite(bhi_m)
    ax1.fill_between(starts_s_m[mm], blo_m[mm], bhi_m[mm],
                     color="C2", alpha=0.25, linewidth=0, label=f"{int(band_ci*100)}% window band")

    ax1.set_title(SAMPLE_ID + " M" + str(MASK_N) + " peak period vs diagonal start time")
    ax1.set_xlabel("Diagonal start time  (t = start_idx · dt_s)  [s]")
    ax1.set_ylabel("Peak period  [s]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    return {
        "start_idx": start_idxs_used, "start_time_s": starts_s_m,
        "period_raw_s": raw_period, "period_smooth_s": smooth_m,
        "band_lo_s": blo_m, "band_hi_s": bhi_m,
        "dt_fft_used_s": dt_fft, "half_window": half_window, "band_ci": band_ci,
    }


def plot_period_vs_diagonal_start_both_lineouts(
    data,
    *,
    dt_s: float,
    start_idxs: np.ndarray | None = None,
    clip_hi_percentile: float = 99.9,
    drop_first_antidiag: int = 0,
    drop_first_horizontal: int = 0,
    half_window: int = 5,
    band_ci: float = 0.68,
    fmin: float | None = 1 / 1000,
    fmax: float | None = 1 / 10,
    detrend: bool = True,
    window: str = "hann",
    cmap: str = "plasma",
    figsize=(13.5, 6.0),
):
    """Period vs diagonal start for both anti-diagonal and horizontal lineouts."""
    C = symmetrize_ttc(data.ttc)
    Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))

    def block_average(C, out_n=300):
        n = C.shape[0]
        m = n // out_n
        C = C[:out_n * m, :out_n * m]
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

    dt_fft_anti = 2.0 * float(dt_s)
    dt_fft_horz = 1.0 * float(dt_s)

    raw_period_anti = np.full(start_idxs.shape, np.nan, dtype=float)
    raw_period_horz = np.full(start_idxs.shape, np.nan, dtype=float)

    for k, i in enumerate(start_idxs):
        try:
            y_a = extract_antidiagonal_lineout_y_only(C, int(i), drop_first=drop_first_antidiag)
        except Exception:
            continue
        if y_a.size >= 16:
            f_peak, f_lo, f_hi, period, period_lo, period_hi, df = fft_peak_with_bin_uncertainty(
                y_a, dt_fft_anti, fmin=fmin, fmax=fmax, detrend=detrend, window=window,
            )
            if np.isfinite(period) and period > 0:
                raw_period_anti[k] = float(period)
        try:
            y_h = extract_horizontal_lineout_y_only(C, int(i), drop_first=drop_first_horizontal)
        except Exception:
            continue
        if y_h.size >= 16:
            f_peak, f_lo, f_hi, period, period_lo, period_hi, df = fft_peak_with_bin_uncertainty(
                y_h, dt_fft_horz, fmin=fmin, fmax=fmax, detrend=detrend, window=window,
            )
            if np.isfinite(period) and period > 0:
                raw_period_horz[k] = float(period)

    half_window = int(max(0, half_window))

    def smooth_with_quantile_band(raw: np.ndarray):
        smooth = np.full_like(raw, np.nan, dtype=float)
        blo = np.full_like(raw, np.nan, dtype=float)
        bhi = np.full_like(raw, np.nan, dtype=float)
        alpha = (1.0 - float(band_ci)) / 2.0
        q_lo = alpha
        q_hi = 1.0 - alpha
        for kk in range(len(start_idxs)):
            a = max(0, kk - half_window)
            b = min(len(start_idxs), kk + half_window + 1)
            wv = raw[a:b]
            wv = wv[np.isfinite(wv)]
            if wv.size < 3:
                continue
            smooth[kk] = float(np.median(wv))
            blo[kk] = float(np.quantile(wv, q_lo))
            bhi[kk] = float(np.quantile(wv, q_hi))
        return smooth, blo, bhi

    smooth_anti, blo_anti, bhi_anti = smooth_with_quantile_band(raw_period_anti)
    smooth_horz, blo_horz, bhi_horz = smooth_with_quantile_band(raw_period_horz)

    starts_s = start_idxs.astype(float) * float(dt_s)
    m_any = np.isfinite(smooth_anti) | np.isfinite(smooth_horz)
    start_idxs_used = start_idxs[m_any]
    starts_s_used = starts_s[m_any]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")
    ax0.set_title(f"{SAMPLE_ID} mask {MASK_N} TTC + diagonal start positions")
    ax0.plot(start_idxs_used, start_idxs_used, "o", ms=3.5, color="C2", alpha=0.9)
    fig.colorbar(im, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[1])
    mA = np.isfinite(smooth_anti) & np.isfinite(blo_anti) & np.isfinite(bhi_anti)
    ax1.plot(starts_s[mA], smooth_anti[mA], lw=2.4, color="C2", label=f"anti-diag median over ±{half_window}")
    ax1.fill_between(starts_s[mA], blo_anti[mA], bhi_anti[mA],
                     color="C2", alpha=0.22, linewidth=0, label=f"anti-diag {int(band_ci*100)}% window band")

    mH = np.isfinite(smooth_horz) & np.isfinite(blo_horz) & np.isfinite(bhi_horz)
    ax1.plot(starts_s[mH], smooth_horz[mH], lw=2.4, color="C0", label=f"horizontal median over ±{half_window}")
    ax1.fill_between(starts_s[mH], blo_horz[mH], bhi_horz[mH],
                     color="C0", alpha=0.18, linewidth=0, label=f"horizontal {int(band_ci*100)}% window band")

    ax1.set_title(f"{SAMPLE_ID} M{MASK_N} peak periods vs diagonal start time")
    ax1.set_xlabel("Diagonal start time  (t = start_idx · dt_s)  [s]")
    ax1.set_ylabel("Peak period  [s]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    return {
        "start_idx": start_idxs_used, "start_time_s": starts_s_used,
        "anti_period_raw_s": raw_period_anti, "horz_period_raw_s": raw_period_horz,
        "anti_period_smooth_s": smooth_anti[m_any], "anti_band_lo_s": blo_anti[m_any],
        "anti_band_hi_s": bhi_anti[m_any],
        "horz_period_smooth_s": smooth_horz[m_any], "horz_band_lo_s": blo_horz[m_any],
        "horz_band_hi_s": bhi_horz[m_any],
        "dt_fft_anti_used_s": dt_fft_anti, "dt_fft_horz_used_s": dt_fft_horz,
        "half_window": half_window, "band_ci": band_ci,
    }


# ---- 2D FFT of TTC ----

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
    """Side-by-side: symmetrized TTC and log-power 2D FFT."""
    C = symmetrize_ttc(ttc)
    Cplot = clip_ttc(C, p_hi=float(clip_hi_percentile))
    N = C.shape[0]

    X = C.astype(np.float64)
    if remove_mean:
        X = X - np.nanmean(X)
    X = np.nan_to_num(X, nan=0.0)
    if window:
        w = np.hanning(N)
        W = w[:, None] * w[None, :]
        X = X * W

    F = np.fft.fftshift(np.fft.fft2(X))
    P = np.abs(F) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt_s))

    fmax = 0.01
    m = (f >= -fmax) & (f <= fmax)
    f_zoom = f[m]
    P_zoom = P[np.ix_(m, m)]

    eps = 1e-12
    axis_mask = (np.abs(f_zoom[:, None]) < eps) | (np.abs(f_zoom[None, :]) < eps)
    F1, F2 = np.meshgrid(f_zoom, f_zoom, indexing="ij")
    diag_mask = np.abs(F1 + F2) < (2 * (f_zoom[1] - f_zoom[0]))
    axis_power = np.sum(P_zoom[axis_mask])
    diag_power = np.sum(P_zoom[diag_mask])
    print(f"Axis power     : {axis_power:.3e}")
    print(f"Diagonal power : {diag_power:.3e}")
    print(f"Ratio A_s/A_d ≈ {axis_power / diag_power:.3f}")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(Cplot, origin="lower", cmap=cmap_ttc, interpolation="nearest", aspect="equal")
    ax0.set_title("TTC (symmetrized)")
    ax0.set_xlabel("t₁ index")
    ax0.set_ylabel("t₂ index")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(
        np.log10(P_zoom + 1e-12), origin="lower",
        extent=[f_zoom[0], f_zoom[-1], f_zoom[0], f_zoom[-1]],
        cmap=cmap_fft, aspect="equal",
    )
    ax1.set_title("2D FFT power  (log scale)")
    ax1.set_xlabel("f₁  [Hz]")
    ax1.set_ylabel("f₂  [Hz]")
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    plt.tight_layout()
    plt.show()
    return {"ttc_sym": C, "fft_power": P, "freqs_hz": f}


# ---- TTC model fitting (2D cosine model) ----

def maybe_window_2d(X: np.ndarray, window: bool) -> np.ndarray:
    if not window:
        return X
    n = X.shape[0]
    w = np.hanning(n)
    W = w[:, None] * w[None, :]
    return X * W


def make_time_axes(n: int, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return t1, t2 grids in seconds, shape (n, n)."""
    t = np.arange(n, dtype=np.float64) * float(dt_s)
    t1 = t[None, :]
    t2 = t[:, None]
    return t1, t2


def model_ttc(
    n: int, dt_s: float, *, C0: float, A_d: float, A_s: float,
    omega_d: float, omega_s: float,
) -> np.ndarray:
    t1, t2 = make_time_axes(n, dt_s)
    return C0 + A_d * np.cos(omega_d * (t2 - t1)) + A_s * np.cos(omega_s * t1) * np.cos(omega_s * t2)


def fit_amplitudes_linear(
    C: np.ndarray, dt_s: float, *, omega_d: float, omega_s: float,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Given omega_d, omega_s, solve for (C0, A_d, A_s) by weighted least squares."""
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    t1, t2 = make_time_axes(n, dt_s)
    Bd = np.cos(omega_d * (t2 - t1))
    Bs = np.cos(omega_s * t1) * np.cos(omega_s * t2)
    y = C.reshape(-1)
    A = np.column_stack([np.ones_like(y), Bd.reshape(-1), Bs.reshape(-1)])

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
    C: np.ndarray, dt_s: float, omega_d: float, omega_s: float,
    *, weights: Optional[np.ndarray] = None,
) -> tuple[float, dict]:
    """Compute SSE after solving amplitudes at (omega_d, omega_s)."""
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


def fft2d_power(
    C: np.ndarray, dt_s: float, *, remove_mean: bool = True, window: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f_hz, P, F): frequency axis, 2D power, complex 2D FFT (all fftshifted)."""
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
    C: np.ndarray, dt_s: float, *, fmax_hz: float = 0.02,
    diag_band_hz: Optional[float] = None,
) -> float:
    """Estimate omega_d from anti-diagonal ridge in 2D FFT."""
    f, P, _ = fft2d_power(C, dt_s, remove_mean=True, window=True)
    n = len(f)
    m = (f >= -fmax_hz) & (f <= fmax_hz)
    idx = np.flatnonzero(m)
    if idx.size < 8:
        raise ValueError("FFT zoom band too small for this N/dt_s")
    Pz = P[np.ix_(idx, idx)]
    fz = f[idx]
    nz = fz.size
    anti = np.array([Pz[i, nz - 1 - i] for i in range(nz)], dtype=np.float64)
    center = nz // 2
    anti[center] = 0.0
    k = int(np.argmax(anti))
    f_peak = float(abs(fz[k]))
    return float(2.0 * np.pi * f_peak)


def guess_omega_s_from_axis_ridge(
    C: np.ndarray, dt_s: float, *, fmax_hz: float = 0.02,
) -> float:
    """Estimate omega_s from axis ridge in 2D FFT."""
    f, P, _ = fft2d_power(C, dt_s, remove_mean=True, window=True)
    n = len(f)
    m = (f >= -fmax_hz) & (f <= fmax_hz)
    idx = np.flatnonzero(m)
    if idx.size < 8:
        raise ValueError("FFT zoom band too small for this N/dt_s")
    Pz = P[np.ix_(idx, idx)]
    fz = f[idx]
    nz = fz.size
    c = nz // 2
    band = 2
    r0 = max(0, c - band)
    r1 = min(nz, c + band + 1)
    axis_profile = np.mean(Pz[r0:r1, :], axis=0)
    axis_profile[c] = 0.0
    k = int(np.argmax(axis_profile))
    f_peak = float(abs(fz[k]))
    return float(2.0 * np.pi * f_peak)


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
    fmax_guess_hz: float = 0.02,
    fd_span_hz: float = 0.004,
    fs_span_hz: float = 0.004,
    n_fd: int = 45,
    n_fs: int = 45,
    refine_rounds: int = 2,
) -> FitResult:
    """Fit (A_d, A_s, omega_d, omega_s) + C0 via FFT-seeded coarse-to-fine grid search."""
    C = symmetrize_ttc(ttc)
    if downsample > 1:
        C = C[::downsample, ::downsample]
    C = np.nan_to_num(C, nan=float(np.nanmean(C)))

    w_d0 = guess_omega_d_from_diag_ridge(C, dt_s * downsample, fmax_hz=fmax_guess_hz)
    w_s0 = guess_omega_s_from_axis_ridge(C, dt_s * downsample, fmax_hz=fmax_guess_hz)

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
                        C0=amps["C0"], A_d=amps["A_d"], A_s=amps["A_s"],
                        omega_d=omega_d, omega_s=omega_s, sse=sse,
                    )
        f_d0 = best.omega_d / (2.0 * np.pi)
        f_s0 = best.omega_s / (2.0 * np.pi)
        fd_span *= 0.35
        fs_span *= 0.35
    return best


def plot_measured_vs_model_ttc(
    ttc: np.ndarray,
    fit: FitResult,
    *,
    dt_s: float = 1.0,
    clip_hi_percentile: float = 99.9,
    cmap: str = "plasma",
    figsize: tuple[float, float] = (12.5, 5.8),
):
    C_meas = symmetrize_ttc(ttc)
    C_mod = model_ttc(
        C_meas.shape[0], dt_s, C0=fit.C0, A_d=fit.A_d, A_s=fit.A_s,
        omega_d=fit.omega_d, omega_s=fit.omega_s,
    )
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


def demo_with_random():
    """Synthetic sanity check for TTC fitting."""
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


# ---- Multi-scan TTC panel plots ----

def _safe_decode(x):
    try:
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return x


def _get_temperature_from_results_hdf(hdf_path):
    """Best-effort temperature fetcher. Returns float or None."""
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
        for p in candidate_paths:
            if p in f:
                try:
                    v = f[p][()]
                    v = np.asarray(v).squeeze()
                    if v.size == 1:
                        return float(v)
                except Exception:
                    pass
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
    """Pick the mask label with the largest integrated intensity in scattering_2d."""
    lab = np.asarray(dynamic_roi_map)
    img = np.asarray(scattering_2d)
    if lab.ndim != 2 or img.ndim != 2 or lab.shape != img.shape:
        raise ValueError(f"Shape mismatch: roi_map {lab.shape}, scattering_2d {img.shape}")
    labels = np.unique(lab)
    labels = labels[labels != 0]
    best_label = None
    best_sum = -np.inf
    for k in labels:
        m = (lab == k)
        s = float(np.nansum(img[m]))
        if s > best_sum:
            best_sum = s
            best_label = int(k)
    if best_label is None:
        raise ValueError("No nonzero ROI labels found in dynamic_roi_map")
    return best_label, best_sum


def temperature_str_from_filename(hdf_path):
    """Extract e.g. '080K' from ..._080K_..._results.hdf."""
    name = Path(hdf_path).name
    m = re.search(r"_([0-9]{2,4}K)_", name)
    return m.group(1) if m else None


def _load_ttc_for_mask_corr(hdf_path: Path, mask_n: int) -> np.ndarray:
    ttc_path = f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}"
    with h5py.File(hdf_path, "r") as f:
        if ttc_path not in f:
            raise KeyError(f"Missing TTC path {ttc_path} in {hdf_path}")
        return f[ttc_path][...]


def read_sample_temperature(f_meta: h5py.File) -> float | None:
    """Try common temperature fields and return value in K if found."""
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
    """One wide figure: N TTC panels in a row, each using the brightest mask."""
    sample_ids = list(sample_ids)
    n_panels = len(sample_ids)

    fig_w = float(figsize_per_panel[0]) * n_panels
    fig_h = float(figsize_per_panel[1])
    fig, axs = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)
    axs = axs[0]

    Cplots = []
    panel_meta = []

    for sid in sample_ids:
        hdf_path = find_results_hdf_direct(Path(base_dir), sid)
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

        Csym = symmetrize_ttc(C)
        cmin = float(np.nanmin(Csym))
        cmax = float(np.nanmax(Csym))
        Cplot = clip_ttc(Csym, p_hi=float(clip_hi_percentile))
        Cplots.append(Cplot)
        T = _get_temperature_from_results_hdf(hdf_path)
        panel_meta.append((sid, mask_n, T, cmin, cmax))

    ims = []
    for ax, Cplot, meta in zip(axs, Cplots, panel_meta):
        sid, mask_n, T, cmin, cmax = meta
        im = ax.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest", aspect="equal")
        ims.append(im)
        hdf_path = find_results_hdf_direct(Path(base_dir), sid)
        temp_str = temperature_str_from_filename(hdf_path)
        if temp_str is None:
            ax.set_title(f"{sid}", fontsize=title_fontsize)
        else:
            ax.set_title(f"{sid} | {temp_str}", fontsize=title_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        txt = f"M{int(mask_n)}\nmin={cmin:.3g}\nmax={cmax:.3g}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=textbox_fontsize,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.5"))

    plt.tight_layout()
    plt.show()
    return panel_meta


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
    label_fontsize: int = 16,
    dt_s: float = 0.5,
    n_ticks: int = 5,
):
    """3×N TTC figure: row 0 = brightest mask, row 1 = brightest-1, row 2 = brightest-2. N = 4 or 5."""
    if base_dir is None:
        base_dir = RESULTS_BASE_DIR
    base_dir = Path(base_dir)

    sample_ids = list(sample_ids)
    n_cols = len(sample_ids)
    if n_cols not in (4, 5):
        raise ValueError("sample_ids must have length 4 or 5 for a 3×N grid")

    def _brightest_mask_like_A4(hdf_path: Path) -> int:
        with h5py.File(hdf_path, "r") as f:
            roi_map = f["xpcs/qmap/dynamic_roi_map"][...]
            scat = f["xpcs/temporal_mean/scattering_2d"][...]
            if scat.ndim == 3:
                scat = scat[0, :, :]
        mask_n, _ = find_brightest_mask_by_integrated_intensity(roi_map, scat)
        return int(mask_n)

    def _load_ttc_local(hdf_path: Path, mask_n: int) -> np.ndarray:
        ttc_path = f"xpcs/twotime/correlation_map/c2_00{int(mask_n):03d}"
        with h5py.File(hdf_path, "r") as f:
            if ttc_path not in f:
                raise KeyError(f"Missing TTC path {ttc_path} in {hdf_path}")
            return f[ttc_path][...]

    hdf_paths: dict[str, Path] = {}
    brightest_by_sid: dict[str, int] = {}
    for sid in sample_ids:
        hdf_path = find_results_hdf_direct(base_dir, sid)
        hdf_paths[sid] = hdf_path
        brightest_by_sid[sid] = _brightest_mask_like_A4(hdf_path)

    fig, axes = plt.subplots(3, n_cols, figsize=figsize)
    axes = np.asarray(axes)
    meta: list[tuple[str, int]] = []

    for r, add in enumerate((0, -1, -2)):
        for c, sid in enumerate(sample_ids):
            ax = axes[r, c]
            mask0 = brightest_by_sid[sid]
            mask_n = int(mask0 + add)
            C = _load_ttc_local(hdf_paths[sid], mask_n)
            Csym = symmetrize_ttc(C)
            Cplot = clip_ttc(Csym, p_hi=float(clip_hi_percentile))
            vmin = float(np.nanmin(Cplot))
            vmax = float(np.nanmax(Cplot))
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
                vmin, vmax = None, None
            ax.imshow(Cplot, origin="lower", cmap=cmap, interpolation="nearest",
                      aspect="equal", vmin=vmin, vmax=vmax)
            n1, n2 = C.shape[1], C.shape[0]
            tick_fontsize = max(8, label_fontsize - 2)
            if r == 2:
                # t1 ticks and labels for bottom row (time in minutes, 1 index = dt_s)
                tick_idx = np.linspace(0, n1 - 1, min(n_ticks, n1), dtype=int)
                tick_mins = tick_idx * dt_s / 60.0
                ax.set_xticks(tick_idx)
                ax.set_xticklabels([f"{t:.0f}" for t in tick_mins], fontsize=tick_fontsize)
            else:
                ax.set_xticks([])
            if c == 0:
                # t2 ticks and labels for left column (time in minutes, 1 index = dt_s)
                tick_idx = np.linspace(0, n2 - 1, min(n_ticks, n2), dtype=int)
                tick_mins = tick_idx * dt_s / 60.0
                ax.set_yticks(tick_idx)
                ax.set_yticklabels([f"{t:.0f}" for t in tick_mins], fontsize=tick_fontsize)
            else:
                ax.set_yticks([])
            ax.set_frame_on(False)
            # ax.text(0.02, 0.98, f"{sid} | M{mask_n:03d}\nmin={np.nanmin(Cplot):.3g}\nmax={np.nanmax(Cplot):.3g}",
            #         transform=ax.transAxes,ha="left", va="top", fontsize=label_fontsize, color="white",
            #         bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none"))
            # ax.text(0.04, 0.96, f"min={np.nanmin(Cplot):.3g}\nmax={np.nanmax(Cplot):.3g}",
            #         transform=ax.transAxes, ha="left", va="top", fontsize=label_fontsize, color="white",
            #         bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none"))
            meta.append((sid, mask_n))

    for c in range(n_cols):
        axes[2, c].set_xlabel(r"$\mathit{t}_1$ (mins)", fontsize=label_fontsize)
    for r in range(3):
        axes[r, 0].set_ylabel(r"$\mathit{t}_2$ (mins)", fontsize=label_fontsize)

    plt.tight_layout(pad=0.5)

    if save:
        if out_path is None:
            out_path = Path.cwd() / "ttc_3x5_brightest_plus_0_1_2.png"
        else:
            out_path = Path(out_path)
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()
    return meta


# ---- Correlation analysis execution wrappers ----
# These use RESULTS_BASE_DIR (= Twotime_PostExpt_01 path) since
# load_xpcs_arrays / find_results_hdf_direct expects the HDF parent directory.

def corr_cosine_fitting_test():
    DT_S = 1.0
    START_DIAG_IDX = 2500
    fit = extract_fit_antidiagonal_with_ttc_plot(
        SAMPLE_ID, RESULTS_BASE_DIR, mask_n=MASK_N,
        start_time_idx=START_DIAG_IDX, dt_s=DT_S,
        omega_min=2 * np.pi / 500.0, omega_max=2 * np.pi / 20.0,
        n_omega=260, n_tau=120,
        despike_at_start=True, despike_halfwidth=1, use_weights=True,
    )


def corr_plot_of_lineout_directions():
    START_DIAG_IDX = 2500
    data = load_xpcs_arrays(SAMPLE_ID, RESULTS_BASE_DIR, mask_n=MASK_N)
    plot_ttc_with_lineouts(data, start=START_DIAG_IDX, add_antidiag_se=True,
                           despike_at_start=True, despike_halfwidth=1)


def corr_plot_of_period_vs_diagonal_start():
    DT_S = 1.0
    data = load_xpcs_arrays(SAMPLE_ID, RESULTS_BASE_DIR, mask_n=MASK_N)
    out = plot_period_vs_diagonal_start(
        data, dt_s=DT_S, drop_first_lineout=5, half_window=5, band_ci=0.68,
        fmin=1 / 1000, fmax=1 / 10, detrend=True, window="hann",
    )


def corr_plot_of_single_fft_antidiagonal_lineout():
    START_DIAG_IDX = 2500
    DT_S = 1.0
    data = load_xpcs_arrays(SAMPLE_ID, RESULTS_BASE_DIR, mask_n=MASK_N)
    out = plot_ttc_lineout_fft(
        data, start_idx=START_DIAG_IDX, dt_s=DT_S,
        drop_first=5, detrend=True, window=True,
    )


def corr_plot_of_period_vs_diagonal_start_both_lineouts():
    DT_S = 1.0
    data = load_xpcs_arrays(SAMPLE_ID, RESULTS_BASE_DIR, mask_n=MASK_N)
    out = plot_period_vs_diagonal_start_both_lineouts(
        data, dt_s=DT_S, drop_first_antidiag=5, drop_first_horizontal=0,
        half_window=5, fmin=1 / 1000, fmax=1 / 10, detrend=True, window="hann",
    )


def corr_fft_2d_plot():
    data = load_xpcs_arrays(SAMPLE_ID, RESULTS_BASE_DIR, mask_n=MASK_N)
    plot_ttc_and_2dfft(data.ttc, dt_s=1.0, clip_hi_percentile=99.9)


def corr_fft_2d_fitting_and_parameter_extraction():
    data = load_xpcs_arrays(SAMPLE_ID, RESULTS_BASE_DIR, mask_n=MASK_N)
    ttc = data.ttc
    fit = fit_ttc_four_params(ttc, dt_s=1.0, downsample=5)
    plot_measured_vs_model_ttc(ttc, fit, dt_s=1.0, clip_hi_percentile=99.5)


def corr_plot_A4_17scan_central_brightest_ttcs():
    scans = [
        "A010","A017","A023","A029","A036","A042","A048","A053","A063",
        "A073","A078","A083","A088","A093","A098","A101","A104",
    ]
    return plot_A4_17scan_central_brightest_ttcs(
        base_dir=RESULTS_BASE_DIR, sample_ids=scans,
        clip_hi_percentile=99.9, cmap="plasma",
        figsize_per_panel=(2.0, 2.25), title_fontsize=9, textbox_fontsize=8,
    )


def corr_exec_plot_3x5_brightest_plus_offsets_ttcs():
    return plot_3x5_brightest_plus_offsets_ttcs(
        base_dir=RESULTS_BASE_DIR,
        sample_ids=["A036", "A048", "A073", "A088", "A101"],
        clip_hi_percentile=99.9, cmap="plasma", figsize=(12.0, 7.2),
        save=True, out_path=FIGURES_DIR / "misc" / "A4_3x5_brightest_plus012.png",
    )


def corr_exec_plot_3x4_brightest_plus_offsets_ttcs():
    return plot_3x5_brightest_plus_offsets_ttcs(
        base_dir=RESULTS_BASE_DIR,
        sample_ids=["A048", "A073", "A088", "A101"],
        clip_hi_percentile=99.9, cmap="plasma", figsize=(9.6, 7.2),
        save=True, out_path=FIGURES_DIR / "misc" / "A4_3x4_brightest_plus012.png",
    )
