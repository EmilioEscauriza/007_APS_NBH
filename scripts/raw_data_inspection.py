from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Callable

import h5py
import hdf5plugin
import numpy as np

import matplotlib as mpl
mpl.use("macosx")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


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
        raise FileNotFoundError(f"No raw run directory found in {data_dir} starting with {sample_id}_")
    return matches[0]


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
    export_path: str = "../figures_export/movies/masked_movie.mp4",
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
                bitrate=2000,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"],
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
    # tw_path = Path("/Users/emilioescauriza/Documents/repos/006_APS_8IDE/emilio_scripts/python_scripts/twotime.py")
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

def mask_roi_viewer():

    run = load_run_data(BASE_DIR, SAMPLE_ID, mask_n=MASK_N)

    launch_masked_raw_viewer(run, mask_n="peak", start_frame=0, clip_percentile_init=99.9, crop_size=300, mp4_export=True)

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
        out_path=Path("A073_M146_raw_vs_processed_vs_diff.png"),
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



# ============================================================
# User parameters / entry point
# ============================================================

# BASE_DIR = Path("/Volumes/EmilioSD4TB/APS_08-IDEI-2025-1006")
BASE_DIR = Path("/Users/emilioescauriza/Desktop")
SAMPLE_ID = "A073"
MASK_N = 144
CONTROL_MASK_N = 176

if __name__ == "__main__":

    # data_structure_viewer()
    mask_roi_viewer()
    # raw_mask_oscillation_inspector()
    # comparison_of_corr_and_g_ttc_plot_methods()
    # compare_existing_vs_corr_entrypoint()
    # compare_existing_ttc_and_cgpt_ttc_from_raw()
    # compare_existing_ttc_and_ttc_from_raw()
    # ttc_with_custom_mask()



    pass
