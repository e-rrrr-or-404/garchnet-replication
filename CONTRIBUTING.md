# Setup Guide for Collaborators

Follow these steps once. Should take 30-45 minutes. After this you'll be able
to run all the code that's currently in the repo.

## Prerequisites

- **Windows, Mac, or Linux** — works on all three.
- **Anaconda or Miniconda** installed. If you don't have it, install Miniconda
  from https://docs.conda.io/projects/miniconda/en/latest/. (Anaconda also works
  but is heavier.)
- **Git** installed. Check by running `git --version` in a terminal. If not,
  install from https://git-scm.com/.

## Step 1: Clone the repo

Open a terminal (PowerShell on Windows, Terminal on Mac/Linux). Navigate to
wherever you keep code projects (NOT inside OneDrive-synced folders on Windows;
OneDrive will lock files and break git).
cd C:\dev          # Windows; create C:\dev first if it doesn't exist
or
cd ~/projects      # Mac/Linux

Then:
git clone https://github.com/e-rrrr-or-404/garchnet-replication

cd garchnet-replication

If git asks for credentials, you'll need a Personal Access Token from GitHub
(not your password). Create one at github.com → Settings → Developer settings →
Personal access tokens → Tokens (classic) → Generate new (classic) → check the
`repo` scope. Copy it and use as the password when git prompts.

## Step 2: Create the Python environment
conda create -n garchnet python=3.11 -y

conda activate garchnet

Your prompt should now show `(garchnet)` at the start. If not, the activation
failed — close the terminal and reopen it before retrying.

## Step 3: Install dependencies

From inside the repo folder, with `(garchnet)` active:
pip install -r requirements.txt

This installs ~15 packages and takes 3-5 minutes. If `torch` fails or hangs,
use the CPU-only install:
pip install torch --index-url https://download.pytorch.org/whl/cpu

## Step 4: Verify everything works

Four checks. All must pass before you write any new code.

### Check 1: data loads
python -c "import pandas as pd; [print(name, pd.read_parquet(f'data/processed/{name}_returns.parquet').shape) for name in ['wig20','spx','ftse']]"

Expected: three lines, each showing (~4500, 7) or (~4500, 8). If you get
"file not found," the data isn't in the repo yet (it should be — re-pull main).

### Check 2: skewed-t module works
python tests\test_skewed_t.py

Expected: four lines saying `[OK]` and "All skewed-t tests passed." If any fail,
report it in the group chat with the full error.

### Check 3: GARCH baseline runs
python -c "

import pandas as pd, sys

sys.path.insert(0, '.')

from src.data_loader import get_window

from src.garch_baseline import fit_garch, forecast_var
df = pd.read_parquet('data/processed/spx_returns.parquet')

train, test, dates = get_window(df, '2005-01-01')

res = fit_garch(train, dist='skewt')

print('VaR(2.5%):', forecast_var(res, alpha=0.025))

"

Expected: a number around -0.06 to -0.08 (i.e. -6% to -8% — this is the SPX
test window starting late 2008, so the band is wide for good reasons).

### Check 4: PyTorch installed
python -c "import torch; print('PyTorch', torch.version); print('CUDA available:', torch.cuda.is_available())"

Expected: a version string and `CUDA available: False` (unless you have an
NVIDIA GPU, which is rare on student laptops). False is fine — we use Colab
for GPU training, not local hardware.


## Workflow once you're set up

### Pulling latest changes
git pull origin main

If git complains about uncommitted local changes, commit or stash them first:
git stash                # save your local changes temporarily

git pull origin main

git stash pop            # restore your local changes

### Pushing your changes
git add <files you changed>

git commit -m "What you did"

git push origin main


