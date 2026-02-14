# 007_APS_NBH

XPCS analysis of APS 8-ID-E beamline data (IPA / NBH samples).

## Structure

```
workflow/         Main workflow: entry point, config, and analysis library
CLI/              CLI version of analysis script
source/           Standalone modules (twotime correlation)
modelling/        Modelling code
notebooks/        Jupyter notebooks for exploration
papers/           Paper drafts (one subfolder per paper)
figures/          Generated figures (gitignored, regenerate via workflow)
```

## Setup

```bash
pip install -r requirements.txt
# Edit workflow/workflow_config.py with your local paths and credentials
```

## Usage

```bash
python workflow/run_workflow.py
```
