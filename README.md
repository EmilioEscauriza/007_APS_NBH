# 007_APS_NBH

XPCS analysis of APS 8-ID-E beamline data (IPA / NBH samples).

## Structure

```
lib/              Reusable analysis & upload code
config/           Workflow configuration (local paths, credentials)
scripts/          Entry points and standalone analysis scripts
notebooks/        Jupyter notebooks for exploration
papers/           Paper drafts (one subfolder per paper)
figures/          Generated figures (gitignored, regenerate via workflow)
data/             Intermediate data files (gitignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate xpcs
cp config/config.example.py config/workflow_config.py
# Edit workflow_config.py with your local paths and credentials
```

## Usage

```bash
python scripts/run_workflow.py
```
