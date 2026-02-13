# ============================================================
# Example config — copy to workflow_config.py and fill in your paths.
# workflow_config.py is gitignored (contains local paths & credentials).
# ============================================================

from pathlib import Path

# ---- raw_data_inspection ----
BASE_DIR = Path("/Volumes/YOUR_DRIVE/APS_08-IDEI-2025-1006")
SAMPLE_ID = "A073"
BASE_DIR_OVERRIDES = {
    # "A073": Path("/Users/you/Desktop"),
}
MASK_N = 144
CONTROL_MASK_N = 176

# ---- analysis (APS 08-IDE): results HDF ----
FILE_ID = "A073"
H5_FILE = Path("/path/to/results.hdf")
H5_BASE_DIR = Path("/Volumes/YOUR_DRIVE/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")

# ---- google_sheet_upload ----
POSITION_NAME = "A4"
RESULTS_BASE_DIR = Path("/Volumes/YOUR_DRIVE/APS_08-IDEI-2025-1006/Twotime_PostExpt_01")
OUT_DIR = Path("/path/to/007_APS_NBH/figures")
SPREADSHEET_ID = "YOUR_SPREADSHEET_ID"
TAB_NAME = "IPA NBH"
TOKEN_PATH = "token.json"
CREDS_PATH = "client_secret_XXXXX.apps.googleusercontent.com.json"
UPLOAD_FOLDER_ID = "YOUR_FOLDER_ID"

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
