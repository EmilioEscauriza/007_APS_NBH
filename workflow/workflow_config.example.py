# ============================================================
# Shared config for workflow_run.py and workflow_lib.py
# Copy this to workflow_config.py and edit paths for your machine.
# ============================================================

from pathlib import Path

# ---- repo root (derived from this file's location) ----
_REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = _REPO_ROOT / "figures"

# ---- raw_data_inspection ----
# Default base directory (external drive) — parent of data/ and Twotime_PostExpt_01/
BASE_DIR = Path("/Volumes/YOUR_DRIVE/APS_08-IDEI-2025-1006")

SAMPLE_ID = "A073"

# Per-scan overrides: raw data that lives somewhere other than BASE_DIR/data/
BASE_DIR_OVERRIDES = {
    "A073": _REPO_ROOT / "data" / "A073",  # A073 raw+processed data lives in repo
}
MASK_N = 144
CONTROL_MASK_N = 176

# ---- analysis (APS 08-IDE): results HDF ----
FILE_ID = "A073"
# Explicit h5 path (used when FILE_ID is A013 or A073); else glob under H5_BASE_DIR
H5_FILE = _REPO_ROOT / "data" / "A073" / "Twotime_PostExpt_01" / "A073_IPA_NBH_1_att0100_260K_001_results.hdf"
H5_BASE_DIR = Path("/Volumes/YOUR_DRIVE/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")

# ---- google_sheet_upload ----
POSITION_NAME = "A4"
RESULTS_BASE_DIR = Path("/Volumes/YOUR_DRIVE/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
OUT_DIR = FIGURES_DIR
SPREADSHEET_ID = "YOUR_SPREADSHEET_ID"
TAB_NAME = "IPA NBH"
TOKEN_PATH = _REPO_ROOT / "config" / "token.json"
CREDS_PATH = _REPO_ROOT / "config" / "client_secret_XXXX.apps.googleusercontent.com.json"
UPLOAD_FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID"

FIGTYPE_DIR = {
    "overview_9": "9_mask_overview",
    "g2s_9": "9_mask_g2",
    "twotime_9": "9_mask_twotime",
    "overview_25": "25_mask_overview",
    "g2s_25": "25_mask_g2",
    "twotime_25": "25_mask_twotime",
}
PLOT_COLS = {
    "overview_9": "AJ", "g2s_9": "AK", "twotime_9": "AL",
    "overview_25": "AM", "g2s_25": "AN", "twotime_25": "AO",
}
ALL_PLOT_KEYS = {
    "overview_9", "g2s_9", "twotime_9",
    "overview_25", "g2s_25", "twotime_25",
}
DPI_BY_PLOT = {
    "overview_9": 300, "g2s_9": 300, "twotime_9": 150,
    "overview_25": 250, "g2s_25": 250, "twotime_25": 80,
}
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

GENERATE_KEYS = None
UPLOAD_TO_SHEETS = True
UPLOAD_KEYS = None
